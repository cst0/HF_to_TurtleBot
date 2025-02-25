#!/usr/bin/env python3

import copy
import numpy as np

import gym
from gym import spaces

from TurtleBot_v0.envs.utils.EnvironmentHandlers import (
    EnvironmentHandler,
    RosEnvironmentHandler,
    StandardEnvironmentHandler,
)


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

        self.map_enable = False
        self.EnvController: EnvironmentHandler
        if rosnode:
            self.EnvController: EnvironmentHandler = RosEnvironmentHandler()
        else:
            self.EnvController: EnvironmentHandler = StandardEnvironmentHandler()
            self.map_enable = True

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
        if self.map_enable:
            self.map = np.zeros((int(self.width * 10), int(self.height * 10)))
        self.rocks_broken = []
        self.trees_broken = []

        # Instantiate the trees
        for i in range(self.n_trees):
            # (Tree 1 will be at absolute location: -1.5 + 3*0.1 = -1.2)
            self.x_pos.append(-self.width / 2 + self.width * x_rand[i])
            self.y_pos.append(-self.height / 2 + self.height * y_rand[i])
            if self.map_enable:
                self.map[int(self.width * 10 * x_rand[i])][  # type: ignore (numpy stubs are incomplete)
                   int(self.height * 10 * y_rand[i])
                ] = 1

        # Instantiate the rocks
        for i in range(self.n_rocks):
            self.x_pos.append(-self.width / 2 + self.width * x_rand[i + self.n_trees])
            self.y_pos.append(-self.height / 2 + self.height * y_rand[i + self.n_trees])
            if self.map_enable:
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
                if self.map_enable:
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
                if self.map_enable:
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
        self.agent_loc = [float(pos.x), float(pos.y)]
        self.agent_orn = float(pos.deg)

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
        if self.map_enable:
            a = int((self.agent_loc[0] + self.width  / 2) * 10)
            b = int((self.agent_loc[1] + self.height / 2) * 10)
            print('** ', a, ' ', b)
            self.map[a][b] = 567 # type: ignore
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
                x = basePos[0] + r * np.cos(np.deg2rad(current_angle_deg))  # type: ignore
                y = basePos[1] + r * np.sin(np.deg2rad(current_angle_deg))  # type: ignore

                for i in range(self.n_trees + self.n_rocks + self.n_table):
                    if x > self.x_low[i] and x < self.x_high[i]:
                        if y > self.y_low[i] and y < self.y_high[i]:
                            flag = 1
                            sensor_value = float(self.sense_range - r) / float(
                                self.sense_range
                            )  # type: ignore
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
                    sensor_value = float(self.sense_range - r) / float(self.sense_range)  # type: ignore
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

        observations = np.asarray(lidar_readings)  # type: ignore

        return observations

    def close(self):
        return
