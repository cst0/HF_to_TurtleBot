import copy
import math
import os
import sys
import time

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
from movement_utils.srv import (
    GetPosition,
    GetPositionRequest,
    GetPositionResponse,
    GoToRelative,
    GoToRelativeRequest,
    GoToRelativeResponse,
    ResetOdom,
    ResetOdomRequest,
    ResetOdomResponse,
)
import numpy as np
from qr_state_reader.srv import (
    ReadEnvironment,
    ReadEnvironmentRequest,
    ReadEnvironmentResponse,
)
import rospy


# import pybullet as p

# REWARD_STEP = -1
# REWARD_DONE = 5000
# REWARD_BREAK = 300
# angle_increment = np.pi/60
# half_beams = 60
# number_of_episodes = 50000
# time_per_episode = 600


class TurtleBotRosNode(object):
    def __init__(self, timeout_seconds=15):
        rospy.init_node("TurtleBotCurriculumNav", anonymous=False)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("Starting up turtlebot ROS node.")

        srv_str_get_position = "/movement_wrapper/get_position"
        srv_str_goto_relative = "/movement_wrapper/goto_relative"
        srv_str_reset_odom = "/movement_wrapper/reset_odom"
        srv_str_read_env = "/qr_state_reader/read_environment"

        try:
            rospy.loginfo("Attempting to connect to movement wrapper services.")
            rospy.wait_for_service(srv_str_get_position, timeout=timeout_seconds)
            self.service_get_position = rospy.ServiceProxy(
                srv_str_get_position, GetPosition
            )
            self.service_goto_position = rospy.ServiceProxy(
                srv_str_goto_relative, GoToRelative
            )
            self.service_reset_odom = rospy.ServiceProxy(srv_str_reset_odom, ResetOdom)
        except rospy.ROSException:
            rospy.logerr(
                "Tried accessing a movement_wrapper service but failed. Exiting."
            )
            sys.exit(1)
        rospy.loginfo("Connected to movement wrapper services successfully.")

        try:
            rospy.loginfo("Attempting to connect to qr service.")
            rospy.wait_for_service(srv_str_read_env, timeout=timeout_seconds)
            self.service_read_env = rospy.ServiceProxy(
                srv_str_read_env, ReadEnvironment
            )
        except rospy.ROSException:
            rospy.logerr(
                "Tried accessing the qr_state_reader service but failed. Exiting."
            )
            sys.exit(1)
        rospy.loginfo("Connected to qr service successfully. Ready to go!")

        self.reset_odom()

    def __del__(self):
        self.service_get_position.close()
        self.service_goto_position.close()
        self.service_reset_odom.close()
        self.service_read_env.close()

    def shutdown(self):
        self.halt()
        del self

    def get_position(self) -> GetPositionResponse:
        try:
            req = GetPositionRequest()
            return self.service_get_position(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:" + str(e))
            return GetPositionResponse()

    def goto_relative(self, req: GoToRelativeRequest) -> GoToRelativeResponse:
        resp = GoToRelativeResponse()
        try:
            resp = self.service_goto_position(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:" + str(e))

        return resp

    def goto_cwise(self):
        req = GoToRelativeRequest()
        req.movement = req.CWISE
        self.goto_relative(req)

    def goto_ccwise(self):
        req = GoToRelativeRequest()
        req.movement = req.CCWISE
        self.goto_relative(req)

    def goto_forward(self):
        req = GoToRelativeRequest()
        req.movement = req.FORWARD
        self.goto_relative(req)

    def halt(self):
        req = GoToRelativeRequest()
        req.movement = req.STOP
        self.goto_relative(req)

    def reset_odom(self) -> ResetOdomResponse:
        try:
            req = ResetOdomRequest()
            self.service_reset_odom(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:" + str(e))

        return ResetOdomResponse()

    def read_environment(self) -> ReadEnvironmentResponse:
        try:
            req = ReadEnvironmentRequest()
            return self.service_read_env(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:" + str(e))
            return ReadEnvironmentResponse()


class TurtleBotV0Env(gym.Env):
    def __init__(
        self,
        map_width=None,
        map_height=None,
        items_id=None,
        items_quantity=None,
        initial_inventory=None,
        goal_env=None,
        is_final=False,
    ):
        self.RosNode = TurtleBotRosNode()

        self.width = map_width
        self.height = map_height
        self.object_types = [
            0,
            1,
            2,
            3,
        ]  # we have 4 objects: wall, tree, rock, and craft table

        self.reward_step = -1
        self.reward_done = 1000
        if is_final == True:
            self.reward_break = 0
        else:
            self.reward_break = 50

        self.reward_hit_wall = -10
        self.reward_extra_inventory = 0

        self.half_beams = 10
        self.angle_increment = np.pi / 10
        self.angle_increment_deg = 18

        self.time_per_episode = 300
        self.sense_range = 5.7

        low = np.zeros(self.half_beams * 2 * len(self.object_types) + 3)
        high = np.ones(self.half_beams * 2 * len(self.object_types) + 3)
        # inventory_array = np.array([5,2])
        # high = np.append(high_array, inventory_array)
        self.observation_space = spaces.Box(low, high, dtype=float)
        self.action_space = spaces.Discrete(5)
        self.num_envs = 1
        self.reset_time = 0

        self.n_trees_org = items_quantity["tree"]
        self.n_rocks_org = items_quantity["rock"]
        self.n_crafting_table = items_quantity["crafting_table"]
        self.starting_trees = initial_inventory["tree"]
        self.starting_rocks = initial_inventory["rock"]
        self.goal_env = goal_env

    def reset(self):

        # print("reset called: ", self.reset_time)
        self.reset_time += 1
        # p.resetSimulation()
        # p.setGravity(0,0,-10)
        # p.setTimeStep(0.01)

        self.env_step_counter = 0
        offset = [0, 0, 0]
        # self.plane = p.loadURDF("plane.urdf")
        # self.turtle = p.loadURDF("turtlebot.urdf",offset)

        self.agent_loc = [0, 0]
        self.agent_orn = np.pi / 2

        self.trees = []
        self.rocks = []
        self.table = []
        self.n_trees = self.n_trees_org
        self.n_rocks = self.n_rocks_org
        self.n_table = self.n_crafting_table
        # x_rand = np.random.rand(self.n_trees + self.n_rocks + self.n_table, 1)
        # y_rand = np.random.rand(self.n_trees + self.n_rocks + self.n_table, 1)
        x_rand = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.48])
        y_rand = np.array([0.3, 0.1, 0.62, 0.41, 0.9, 0.25, 0.1])
        self.x_pos = []
        self.y_pos = []
        dummy_req = ReadEnvironmentResponse()
        self.objects_listing = [
            dummy_req.TREE1,
            dummy_req.TREE2,
            dummy_req.TREE3,
            dummy_req.TREE4,
            dummy_req.ROCK1,
            dummy_req.ROCK2,
            dummy_req.CRAFTING_TABLE,
        ]

        for i in range(self.n_trees):  # Instantiate the trees
            self.x_pos.append(-self.width / 2 + self.width * x_rand[i])
            self.y_pos.append(-self.height / 2 + self.height * y_rand[i])
            # self.trees.append(p.loadURDF("trees.urdf", basePosition=[-self.width/2+self.width*x_rand[i], -self.height/2+self.height*y_rand[i], 0], useFixedBase = 1))

        for i in range(self.n_rocks):  # Instantiate the rocks
            self.x_pos.append(-self.width / 2 + self.width * x_rand[i + self.n_trees])
            self.y_pos.append(-self.height / 2 + self.height * y_rand[i + self.n_trees])
            # self.rocks.append(p.loadURDF("rocks.urdf", basePosition=[-self.width/2+self.width*x_rand[i+self.n_trees], -self.height/2+self.height*y_rand[i+self.n_trees], 0], useFixedBase = 1))

        for i in range(self.n_table):
            if (
                abs(
                    -self.width / 2
                    + self.width * x_rand[i + self.n_trees + self.n_rocks]
                )
                < 0.3
                and abs(
                    -self.height / 2
                    + self.height * y_rand[i + self.n_trees + self.n_rocks]
                )
                < 0.3
            ):
                self.x_pos.append(self.width / 2 - 0.05)
                self.y_pos.append(self.height / 2 - 0.05)
                # self.table.append(p.loadURDF("table.urdf", basePosition=[self.width/2-0.05,self.height/2-0.05, 0], useFixedBase = 1))
            else:
                self.x_pos.append(
                    -self.width / 2
                    + self.width * x_rand[i + self.n_trees + self.n_rocks]
                )
                self.y_pos.append(
                    -self.height / 2
                    + self.height * y_rand[i + self.n_trees + self.n_rocks]
                )
                # self.table.append(p.loadURDF("table.urdf", basePosition=[-self.width/2+self.width*x_rand[i+self.n_trees+self.n_rocks], -self.height/2+self.height*y_rand[i+self.n_rocks+self.n_trees], 0], useFixedBase = 1))

        self.inventory = dict(
            [("wood", self.starting_trees), ("stone", self.starting_rocks), ("pogo", 0)]
        )
        self.x_low = [i - 0.15 for i in self.x_pos]
        self.x_high = [i + 0.15 for i in self.x_pos]
        self.y_low = [i - 0.15 for i in self.y_pos]
        self.y_high = [i + 0.15 for i in self.y_pos]

        obs = self.get_observation()
        self.x_pos_copy = copy.deepcopy(self.x_pos)
        self.y_pos_copy = copy.deepcopy(self.y_pos)

        print("X_POS", self.x_pos)
        print("Y_POS", self.y_pos)

        return obs

    def step(self, action):
        basePos = copy.deepcopy(self.agent_loc)
        baseOrn = copy.deepcopy(self.agent_orn)

        reward = self.reward_step
        done = False

        forward = 0
        object_removed = 0
        index_removed = 0

        if action == 0:  # Turn right
            baseOrn -= 20 * np.pi / 180
            self.RosNode.goto_cwise()

        elif action == 1:  # Turn left
            baseOrn += 20 * np.pi / 180
            self.RosNode.goto_ccwise()

        elif action == 2:  # Move forward
            x_new = basePos[0] + 0.25 * float(np.cos(baseOrn))
            y_new = basePos[1] + 0.25 * float(np.sin(baseOrn))
            forward = 1
            for i in range(self.n_trees + self.n_rocks + self.n_table):
                if abs(self.x_pos[i] - x_new) < 0.15:
                    if abs(self.y_pos[i] - y_new) < 0.15:
                        forward = 0

            if (abs(abs(x_new) - abs(self.width / 2)) < 0.05) or (
                abs(abs(y_new) - abs(self.height / 2)) < 0.05
            ):
                reward = self.reward_hit_wall
                forward = 0

            if forward == 1:
                basePos[0] = x_new
                basePos[1] = y_new
                self.RosNode.goto_forward()
                point = self.RosNode.get_position().point
                #basePos[0] = point.x
                #basePos[1] = point.y

        elif action == 3:  # Break
            x = basePos[0]
            y = basePos[1]
            print("Inventory: ", self.inventory)
            add_to_inventory = False
            for i in range(self.n_trees + self.n_rocks + self.n_table):
                if abs(self.x_pos[i] - x) < 0.3:
                    if abs(self.y_pos[i] - y) < 0.3:
                        if i < self.n_trees:
                            self.inventory["wood"] += 1
                            object_removed = 1
                            index_removed = i
                            add_to_inventory = True
                            # if self.inventory["wood"] <= 2:
                            #    reward = self.reward_break
                            # elif self.inventory["wood"] > 2:
                            #    reward = self.reward_extra_inventory
                        elif i > self.n_trees - 1 and i < self.n_trees + self.n_rocks:
                            self.inventory["stone"] += 1
                            object_removed = 2
                            index_removed = i
                            add_to_inventory = True
                            # if self.inventory["stone"] <= 1:
                            #    reward = self.reward_break
                            # elif self.inventory["stone"] > 1:
                            #    reward = self.reward_extra_inventory

            if add_to_inventory:
                print("object interaction with type ", object_removed, "index", index_removed)
                print("I think I'm at", x, ",", y)
                time.sleep(20)

            #            reading = self.RosNode.read_environment()
            #            if (
            #                reading.reading == reading.TREE1
            #                or reading.reading == reading.TREE2
            #                or reading.reading == reading.TREE3
            #                or reading.reading == reading.TREE4
            #            ):
            #                # this is a tree. which tree is it?
            #                object_removed = 1
            #                for n in range(0, len(self.objects_listing)):
            #                    if self.objects_listing[n] == reading.reading:
            #                        index_removed = n
            #
            #            if reading.reading == reading.ROCK1 or reading.reading == reading.ROCK2:
            #                # this is a rock. which rock is it?
            #                object_removed = 2
            #                for n in range(0, len(self.objects_listing)):
            #                    if self.objects_listing[n] == reading.reading:
            #                        index_removed = n

            if object_removed == 1:
                self.objects_listing.pop(index_removed)
                self.x_pos.pop(index_removed)
                self.y_pos.pop(index_removed)
                self.x_low.pop(index_removed)
                self.x_high.pop(index_removed)
                self.y_low.pop(index_removed)
                self.y_high.pop(index_removed)
                # p.removeBody(self.trees[index_removed])
                # self.trees.pop(index_removed)
                self.n_trees -= 1
                # print("tree removed")

                # print("obs: ",self.get_observation())

            if object_removed == 2:
                self.objects_listing.pop(index_removed)
                self.x_pos.pop(index_removed)
                self.y_pos.pop(index_removed)
                self.x_low.pop(index_removed)
                self.x_high.pop(index_removed)
                self.y_low.pop(index_removed)
                self.y_high.pop(index_removed)
                # p.removeBody(self.rocks[index_removed-self.n_trees])
                # self.rocks.pop(index_removed-self.n_trees)
                self.n_rocks -= 1
                # print("rock removed")
                # print("obs: ",self.get_observation())

        elif action == 4:  # Craft
            x = basePos[0]
            y = basePos[1]

            reading = self.RosNode.read_environment()
            at_crafting_table = reading.reading == reading.CRAFTING_TABLE

            if (
                at_crafting_table
                and self.inventory["wood"] >= 2
                and self.inventory["stone"] >= 1
            ):
                self.inventory["pogo"] += 1
                self.inventory["wood"] -= 2
                self.inventory["stone"] -= 1
                done = True
                reward = self.reward_done

        self.agent_loc = basePos
        self.agent_orn = baseOrn
        if self.goal_env == 0:
            x = basePos[0]
            y = basePos[1]
            for i in range(self.n_trees_org + self.n_rocks_org + self.n_table):
                if abs(self.x_pos_copy[i] - x) < 0.3:
                    if abs(self.y_pos_copy[i] - y) < 0.3:
                        reward = self.reward_done
                        done = True

        elif self.goal_env == 1:
            if (
                self.inventory["wood"] >= self.n_trees_org + self.starting_trees
                or self.inventory["wood"] >= 2
            ) and (
                self.inventory["stone"] >= self.n_rocks_org + self.starting_rocks
                or self.inventory["stone"] >= 1
            ):
                reward = self.reward_done
                done = True
                print("Inventory: ", self.inventory)

        self.env_step_counter += 1

        # if self.env_step_counter >= self.time_per_episode:
        # 	done = True

        obs = self.get_observation()

        # print(obs)
        return obs, reward, done, {}

    def get_observation(self):

        # print("here 0")
        num_obj_types = len(self.object_types)

        basePos = copy.deepcopy(self.agent_loc)
        baseOrn = copy.deepcopy(self.agent_orn)

        base = baseOrn
        rot_degree = base * 57.2958
        current_angle_deg = rot_degree
        current_angle = base
        lidar_readings = []
        index_temp = 0
        angle_temp = 0
        # print("base Pos x: ", basePos[0])
        # print("base Pos y: ", basePos[1])
        # print("base Ori x: ", eulerOrn[0])
        # print("base Ori y: ", eulerOrn[1])
        # print("base Ori z: ", eulerOrn[2])
        # print("\n")

        while True:

            beam_i = np.zeros(num_obj_types)

            for r in np.arange(0, self.sense_range, 0.1):

                flag = 0
                x = basePos[0] + r * np.cos(np.deg2rad(current_angle_deg))
                y = basePos[1] + r * np.sin(np.deg2rad(current_angle_deg))

                for i in range(self.n_trees + self.n_rocks + self.n_table):
                    if x > self.x_low[i] and x < self.x_high[i]:
                        if y > self.y_low[i] and y < self.y_high[i]:
                            flag = 1
                            sensor_value = float(self.sense_range - r) / float(
                                self.sense_range
                            )
                            if i < self.n_trees:
                                obj_type = 1  # Update object as tree
                                # p.addUserDebugLine([basePos[0],basePos[1],0],[x,y,0],[0,1,0],1.9, lifeTime = 4)

                            elif (
                                i > self.n_trees - 1 and i < self.n_trees + self.n_rocks
                            ):
                                obj_type = 2  # Update object as rocks
                                # p.addUserDebugLine([basePos[0],basePos[1],0],[x,y,0],[1,0,0],1.9, lifeTime = 4)

                            else:
                                obj_type = 3  # Update object as table
                                # p.addUserDebugLine([basePos[0],basePos[1],0],[x,y,0],[0,0,1],1.9, lifeTime = 4)

                            index_temp += 1
                            beam_i[obj_type] = sensor_value

                            break

                if flag == 1:
                    break

                if (
                    abs(self.width / 2) - abs(x) < 0.05
                    or abs(self.height / 2) - abs(y) < 0.05
                ):
                    sensor_value = float(self.sense_range - r) / float(self.sense_range)
                    # p.addUserDebugLine([basePos[0],basePos[1],0],[x,y,0],[0,0,0],0.9, lifeTime = 4)
                    index_temp += 1
                    beam_i[0] = sensor_value
                    break

            for k in range(0, len(beam_i)):
                lidar_readings.append(beam_i[k])

            current_angle += self.angle_increment
            angle_temp += 1

            # Commented for degree
            # if current_angle >= 2*np.pi + base:
            # 	break

            current_angle_deg += self.angle_increment_deg

            if current_angle_deg >= 343 + rot_degree:
                break

        while len(lidar_readings) < self.half_beams * 2 * num_obj_types:
            print("lidar readings appended")
            lidar_readings.append(0)

        while len(lidar_readings) > self.half_beams * 2 * num_obj_types:
            print("lidar readings popped")
            lidar_readings.pop()

        lidar_readings.append(self.inventory["wood"])
        lidar_readings.append(self.inventory["stone"])
        lidar_readings.append(self.inventory["pogo"])

        observations = np.asarray(lidar_readings)

        # print("here 1")
        return observations

    def close(self):
        return
