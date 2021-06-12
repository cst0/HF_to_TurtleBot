#!/usr/bin/env python3

import abc
import copy
from enum import Enum
import time
from typing import List, Tuple, Union
import rospy
import sys
import numpy as np

import gym
from gym import spaces
from movement_utils.srv  import GetPosition     , GetPositionRequest     , GetPositionResponse
from movement_utils.srv  import GoToRelative    , GoToRelativeRequest    , GoToRelativeResponse
from movement_utils.srv  import ResetOdom       , ResetOdomRequest       , ResetOdomResponse
from qr_state_reader.srv import ReadEnvironment , ReadEnvironmentRequest , ReadEnvironmentResponse


class TurtleBotRosNode:
    def __init__(self, timeout_seconds=15):
        rospy.init_node('TurtleBotCurriculumNav', anonymous=False)

        srv_str_get_position  = '/movement_wrapper/get_position'
        srv_str_goto_relative = '/movement_wrapper/goto_relative'
        srv_str_reset_odom    = '/movement_wrapper/reset_odom'
        srv_str_read_env      = '/qr_state_reader/read_environment'

        try:
            rospy.wait_for_service(srv_str_get_position, timeout=timeout_seconds)
            self.service_get_position  = rospy.ServiceProxy(srv_str_get_position,  GetPosition)
            self.service_goto_position = rospy.ServiceProxy(srv_str_goto_relative, GoToRelative)
            self.service_reset_odom    = rospy.ServiceProxy(srv_str_reset_odom,    ResetOdom)
        except rospy.ROSException:
            rospy.logerr('Tried accessing a movement_wrapper service but failed. Exiting.')
            sys.exit(1)

        try:
            rospy.wait_for_service(srv_str_read_env, timeout=timeout_seconds)
            self.service_read_env = rospy.ServiceProxy(srv_str_read_env, ReadEnvironment)
        except rospy.ROSException:
            rospy.logerr('Tried accessing the qr_state_reader service but failed. Exiting.')
            sys.exit(1)

        self.reset_odom()

    def get_position(self):
        try:
            return self.service_get_position()
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:" + str(e))

    def goto_relative(self, req:GoToRelativeRequest):
        try:
            self.service_goto_position(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:" + str(e))

    def reset_odom(self):
        try:
            self.service_reset_odom()
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:" + str(e))

    def read_environment(self):
        try:
            return self.service_read_env()
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:" + str(e))


class Action(Enum):
    STOP=0
    FORWARD=1
    CWISE=2
    CCWISE=3
    BREAK=4
    CRAFT=5


class Reading(Enum):
    TREE1=0
    TREE2=1
    TREE3=2
    TREE4=3
    ROCK1=4
    ROCK2=5
    NONE=6
    CRAFTING_TABLE=7


class Position:
    def __init__(self, x, y, deg):
        self.x   = x
        self.y   = y
        self.deg = deg


class EnvironmentHandler(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def take_action(self, action:Action) -> Tuple[float, bool]:  # returning reward, done
        raise NotImplementedError

    @abc.abstractmethod
    def get_position(self) -> Position:
        raise NotImplementedError

    @abc.abstractmethod
    def get_reading(self) -> Reading:
        raise NotImplementedError


class RosEnvironmentHandler(EnvironmentHandler):
    def __init__(self):
        self.node = TurtleBotRosNode()

    def take_action(self, action:Action) -> Tuple[float, bool]:  # returning reward, done
        req = GoToRelativeRequest()
        if action == Action.FORWARD:
            req.movement = req.FORWARD
            self.node.goto_relative(req)

        if action == Action.CWISE:
            req.movement = req.CWISE
            self.node.goto_relative(req)

        if action == Action.CCWISE:
            req.movement = req.CCWISE
            self.node.goto_relative(req)

        if action == Action.BREAK:
            reading = self.get_reading()

        if action == Action.CRAFT:
            reading = self.get_reading()

        return 0.0

    def get_position(self) -> Position:
        return Position(0,0,0)

    def get_reading(self) -> Reading:
        return Reading.NONE


class StandardEnvironmentHandler(EnvironmentHandler):
    def __init__(self, calling_super):
        self.calling_super = calling_super
        self.agent_loc = None
        self.agent_orn = None

    def take_action(self, action:Action):
        basePos = copy.deepcopy(self.calling_super.agent_loc)
        baseOrn = copy.deepcopy(self.calling_super.agent_orn)

        reward = self.calling_super.reward_step
        done = False

        forward = 0
        object_removed = 0
        index_removed = 0

        self.calling_super.map[int((self.calling_super.agent_loc[0] + self.calling_super.width / 2) * 10)][ # type: ignore
            int((self.calling_super.agent_loc[1] + self.calling_super.height / 2) * 10)
        ] = 0

        if action == 0:  # Turn right
            baseOrn -= 20 * np.pi / 180

        elif action == 1:  # Turn left
            baseOrn += 20 * np.pi / 180

        elif action == 2:  # Move forward
            x_new = basePos[0] + 0.25 * np.cos(baseOrn)  # type: ignore
            y_new = basePos[1] + 0.25 * np.sin(baseOrn)  # type: ignore
            forward = 1
            for i in range(self.calling_super.n_trees + self.calling_super.n_rocks + self.calling_super.n_table):
                if abs(self.calling_super.x_pos[i] - x_new) < 0.15:
                    if abs(self.calling_super.y_pos[i] - y_new) < 0.15:
                        forward = 0

            if (abs(abs(x_new) - abs(self.calling_super.width / 2)) < 0.25) or (
                abs(abs(y_new) - abs(self.calling_super.height / 2)) < 0.25
            ):
                reward = self.calling_super.reward_hit_wall
                forward = 0

            if forward == 1:
                basePos[0] = x_new
                basePos[1] = y_new

        elif action == 3:  # Break
            x = basePos[0]
            y = basePos[1]
            index_removed = self.get_reading().value
            if index_removed >= 0 and index_removed < 4:
                object_removed = 1
                print("Index Removed: ", index_removed)
                time.sleep(5.0)
                self.calling_super.inventory["wood"] += 1
                if self.calling_super.inventory["wood"] <= 2:
                    reward = self.calling_super.reward_break
            if index_removed > 3 and index_removed < 6:
                object_removed = 2
                self.calling_super.rocks_broken.append(index_removed)
                print("Index Removed: ", index_removed)
                time.sleep(5.0)
                self.calling_super.inventory["stone"] += 1
                if self.calling_super.inventory["stone"] <= 1:
                    reward = self.calling_super.reward_break

            if object_removed == 1:
                flag = 0
                for i in range(len(self.calling_super.trees_broken)):
                    if index_removed > self.calling_super.trees_broken[i]:
                        flag += 1
                self.calling_super.x_pos.pop(index_removed - flag)
                self.calling_super.y_pos.pop(index_removed - flag)
                self.calling_super.x_low.pop(index_removed - flag)
                self.calling_super.x_high.pop(index_removed - flag)
                self.calling_super.y_low.pop(index_removed - flag)
                self.calling_super.y_high.pop(index_removed - flag)
                self.calling_super.n_trees -= 1
                self.calling_super.trees_broken.append(index_removed)
                print("Object Broken:", object_removed)
                print("Index Broken:", index_removed)

            if object_removed == 2:
                flag = 0
                for i in range(len(self.calling_super.rocks_broken)):
                    if index_removed > self.calling_super.rocks_broken[i]:
                        flag += 1
                flag += len(self.calling_super.trees_broken)
                self.calling_super.x_pos.pop(index_removed - flag)
                self.calling_super.y_pos.pop(index_removed - flag)
                self.calling_super.x_low.pop(index_removed - flag)
                self.calling_super.x_high.pop(index_removed - flag)
                self.calling_super.y_low.pop(index_removed - flag)
                self.calling_super.y_high.pop(index_removed - flag)
                self.calling_super.n_rocks -= 1
                print("Object Broken:", object_removed)
                print("Index Broken:", index_removed)

        elif action == 4:  # Craft
            x = basePos[0]
            y = basePos[1]
            index_removed = self.get_reading().value
            if index_removed == 7:
                if self.calling_super.inventory["wood"] >= 2 and self.calling_super.inventory["stone"] >= 1:
                    self.calling_super.inventory["pogo"] += 1
                    self.calling_super.inventory["wood"] -= 2
                    self.calling_super.inventory["stone"] -= 1
                    done = True
                    reward = self.calling_super.reward_done

        return reward, done

    def get_position(self) -> Position:
        return Position(self.agent_loc[0], self.agent_loc[1], self.agent_orn)

    def get_reading(self) -> Reading:
        return Reading.NONE


class TurtleBotV0Env(gym.Env):
    def __init__(
        self,
        map_width=0,
        map_height=0,
        items_quantity=None,
        initial_inventory=None,
        goal_env=None,
        is_final=False,
        rosnode=True,
    ):

        self.EnvController:EnvironmentHandler
        if rosnode:
            self.EnvController:EnvironmentHandler = RosEnvironmentHandler()
        else:
            self.EnvController:EnvironmentHandler = StandardEnvironmentHandler(self)

        self.width = np.float64(map_width)
        self.height = np.float64(map_height)
        # we have 4 objects: wall, tree, rock, and craft table
        self.object_types = [0, 1, 2, 3]

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
        self.reset_time += 1

        self.env_step_counter = 0
        self.agent_loc = [0, 0]
        self.agent_orn = np.pi / 2

        self.trees = []
        self.rocks = []
        self.table = []
        self.n_trees = self.n_trees_org
        self.n_rocks = self.n_rocks_org
        self.n_table = self.n_crafting_table

        # First 4 are for trees, next 2 for rocks and the last for crafting table
        x_rand = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.48])
        y_rand = np.array([0.3, 0.1, 0.62, 0.41, 0.9, 0.25, 0.1])

        self.x_pos = []
        self.y_pos = []
        self.map = np.zeros((int(self.width * 10), int(self.height * 10)))
        self.rocks_broken = []
        self.trees_broken = []

        # Instantiate the trees
        for i in range(self.n_trees):
            # (Tree 1 will be at absolute location: -1.5 + 3*0.1 = -1.2)
            self.x_pos.append(-self.width / 2 + self.width * x_rand[i])
            self.y_pos.append(-self.height / 2 + self.height * y_rand[i])
            self.map[int(self.width * 10 * x_rand[i])][  # type: ignore (numpy stubs are incomplete)
                int(self.height * 10 * y_rand[i])
            ] = 1

        # Instantiate the rocks
        for i in range(self.n_rocks):
            self.x_pos.append(-self.width / 2 + self.width * x_rand[i + self.n_trees])
            self.y_pos.append(-self.height / 2 + self.height * y_rand[i + self.n_trees])
            self.map[int(self.width * 10 * x_rand[i + self.n_trees])][  # type: ignore (numpy stubs are incomplete)
                int(self.height * 10 * y_rand[i + self.n_trees])
            ] = 2

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
                self.map[int(self.width * 5)][int(self.height * 5)] = 3  # type: ignore (more numpy stuff)
            else:
                self.x_pos.append(
                    -self.width / 2
                    + self.width * x_rand[i + self.n_trees + self.n_rocks]
                )
                self.y_pos.append(
                    -self.height / 2
                    + self.height * y_rand[i + self.n_trees + self.n_rocks]
                )
                self.map[
                    int(self.width * 10 * x_rand[i + self.n_trees + self.n_rocks])  # type: ignore
                ][int(self.height * 10 * y_rand[i + self.n_trees + self.n_rocks])] = 3

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
        print("X Pos: ", self.x_pos)
        print("Y Pos: ", self.y_pos)

        return obs

    def step(self, action):
        # action if statement from here
        reward, done = self.EnvController.take_action(action=action)
        pos = self.EnvController.get_position()

        self.agent_loc = [pos.x, pos.y]
        self.agent_orn = pos.deg

        if self.goal_env == 0:
            x = pos.x
            y = pos.y
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

        obs = self.get_observation()
        self.map[int((self.agent_loc[0] + self.width / 2) * 10)][  # type: ignore
            int((self.agent_loc[1] + self.height / 2) * 10)
        ] = 567
        return obs, reward, done, {}

    def get_observation(self):
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

        while True:
            beam_i = np.zeros(num_obj_types)
            for r in np.arange(0, self.sense_range, 0.1):
                flag = 0
                x = basePos[0] + r * np.cos(np.deg2rad(current_angle_deg)) # type: ignore
                y = basePos[1] + r * np.sin(np.deg2rad(current_angle_deg)) # type: ignore

                for i in range(self.n_trees + self.n_rocks + self.n_table):
                    if x > self.x_low[i] and x < self.x_high[i]:
                        if y > self.y_low[i] and y < self.y_high[i]:
                            flag = 1
                            sensor_value = \
                                    float(self.sense_range - r) / float(self.sense_range) # type: ignore
                            if i < self.n_trees:
                                obj_type = 1  # Update object as tree

                            elif (
                                i > self.n_trees - 1 and i < self.n_trees + self.n_rocks
                            ):
                                obj_type = 2  # Update object as rocks

                            else:
                                obj_type = 3  # Update object as table

                            index_temp += 1
                            beam_i[obj_type] = sensor_value

                            break

                if flag == 1:
                    break

                if (
                    abs(self.width / 2) - abs(x) < 0.05
                    or abs(self.height / 2) - abs(y) < 0.05
                ):
                    sensor_value = float(self.sense_range - r) / float(self.sense_range) # type: ignore
                    index_temp += 1
                    beam_i[0] = sensor_value
                    break

            for k in range(0, len(beam_i)):
                lidar_readings.append(beam_i[k])

            current_angle += self.angle_increment
            angle_temp += 1
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

        observations = np.asarray(lidar_readings) # type: ignore

        return observations

    def close(self):
        return
