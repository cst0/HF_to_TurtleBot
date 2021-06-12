import abc
import time
from enum import Enum
from movement_utils.srv import GetPosition, GetPositionRequest, GetPositionResponse
from movement_utils.srv import GoToRelative, GoToRelativeRequest, GoToRelativeResponse
from movement_utils.srv import ResetOdom, ResetOdomRequest, ResetOdomResponse
from qr_state_reader.srv import (
    ReadEnvironment,
    ReadEnvironmentRequest,
    ReadEnvironmentResponse,
)
from typing import List, Tuple, Union
import rospy
import sys


class TurtleBotRosNode:
    def __init__(self, timeout_seconds=15):
        rospy.init_node("TurtleBotCurriculumNav", anonymous=False)

        srv_str_get_position = "/movement_wrapper/get_position"
        srv_str_goto_relative = "/movement_wrapper/goto_relative"
        srv_str_reset_odom = "/movement_wrapper/reset_odom"
        srv_str_read_env = "/qr_state_reader/read_environment"

        try:
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

        try:
            rospy.wait_for_service(srv_str_read_env, timeout=timeout_seconds)
            self.service_read_env = rospy.ServiceProxy(
                srv_str_read_env, ReadEnvironment
            )
        except rospy.ROSException:
            rospy.logerr(
                "Tried accessing the qr_state_reader service but failed. Exiting."
            )
            sys.exit(1)

        self.reset_odom()

    def get_position(self):
        try:
            return self.service_get_position()
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:" + str(e))

    def goto_relative(self, req: GoToRelativeRequest):
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
    STOP = 0
    FORWARD = 1
    CWISE = 2
    CCWISE = 3
    BREAK = 4
    CRAFT = 5


class Reading(Enum):
    TREE1 = 0
    TREE2 = 1
    TREE3 = 2
    TREE4 = 3
    ROCK1 = 4
    ROCK2 = 5
    NONE = 6
    CRAFTING_TABLE = 7


class Position:
    def __init__(self, x, y, deg):
        self.x = x
        self.y = y
        self.deg = deg


class EnvironmentHandler(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def take_action(
        self, action: Action
    ) -> Tuple[float, bool]:  # returning reward, done
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

    def take_action(
        self, action: Action
    ) -> Tuple[float, bool]:  # returning reward, done
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
        return Position(0, 0, 0)

    def get_reading(self) -> Reading:
        return Reading.NONE


class StandardEnvironmentHandler(EnvironmentHandler):
    def __init__(self, calling_super):
        self.calling_super = calling_super
        self.agent_loc = None
        self.agent_orn = None

    def take_action(self, action: Action):
        basePos = copy.deepcopy(self.calling_super.agent_loc)
        baseOrn = copy.deepcopy(self.calling_super.agent_orn)

        reward = self.calling_super.reward_step
        done = False

        forward = 0
        object_removed = 0
        index_removed = 0

        self.calling_super.map[int((self.calling_super.agent_loc[0] + self.calling_super.width / 2) * 10)][  # type: ignore
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
            for i in range(
                self.calling_super.n_trees
                + self.calling_super.n_rocks
                + self.calling_super.n_table
            ):
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
                if (
                    self.calling_super.inventory["wood"] >= 2
                    and self.calling_super.inventory["stone"] >= 1
                ):
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
