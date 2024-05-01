# (c) 2024 chris paxton under MIT license

import time
import timeit
from threading import Lock
from typing import List

import click
import cv2
import numpy as np
import rclpy
import zmq

from home_robot.core.interfaces import ContinuousNavigationAction
from home_robot.core.robot import RobotClient
from home_robot.motion.robot import RobotModel
from home_robot.utils.image import Camera
from home_robot.utils.point_cloud import show_point_cloud


class HomeRobotZmqClient(RobotClient):
    def __init__(
        self,
        robot_ip: str = "192.168.1.15",
        recv_port: int = 4401,
        send_port: int = 4402,
        use_remote_computer: bool = True,
    ):
        self.recv_port = recv_port
        self.send_port = send_port
        self.reset()

        # Create ZMQ sockets
        self.context = zmq.Context()

        # Receive state information
        self.recv_socket = self.context.socket(zmq.SUB)
        self.recv_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.recv_socket.setsockopt(zmq.SNDHWM, 1)
        self.recv_socket.setsockopt(zmq.RCVHWM, 1)
        self.recv_socket.setsockopt(zmq.CONFLATE, 1)

        # SEnd actions back to the robot for execution
        self.send_socket = self.context.socket(zmq.PUB)
        self.send_socket.setsockopt(zmq.SNDHWM, 1)
        self.send_socket.setsockopt(zmq.RCVHWM, 1)
        action_send_address = "tcp://*:" + str(self.send_port)
        print(f"Publishing actions on {action_send_address}...")
        self.send_socket.bind(action_send_address)

        # Use remote computer or whatever
        if use_remote_computer:
            self.address = "tcp://" + robot_ip + ":" + str(self.recv_port)
        else:
            self.address = "tcp://" + "127.0.0.1" + ":" + str(self.recv_port)

        print(f"Connecting to {self.address} to receieve observations...")
        self.recv_socket.connect(self.address)
        print("...connected.")

        self._obs_lock = Lock()
        self._act_lock = Lock()

    def navigate_to(
        self, xyt: ContinuousNavigationAction, relative=False, blocking=False
    ):
        """Move to xyt in global coordinates or relative coordinates."""
        if isinstance(xyt, ContinuousNavigationAction):
            xyt = xyt.xyt
        assert len(xyt) == 3, "xyt must be a vector of size 3"
        with self._act_lock:
            self._next_action["xyt"] = xyt
            self._next_action["nav_relative"] = relative
            self._next_action["nav_blocking"] = blocking

    def reset(self):
        """Reset everything in the robot's internal state"""
        self._control_mode = None
        self._next_action = dict()

    def switch_to_navigation_mode(self):
        with self._act_lock:
            self._next_action["control_mode"] = "navigation"

    def switch_to_manipulation_mode(self):
        with self._act_lock:
            self._next_action["control_mode"] = "manipulation"

    def move_to_nav_posture(self):
        with self._act_lock:
            self._next_action["posture"] = "navigation"

    def move_to_manip_posture(self):
        with self._act_lock:
            self._next_action["posture"] = "manipulation"

    def in_manipulation_mode(self) -> bool:
        """is the robot ready to grasp"""
        return self._control_mode == "manipulation"

    def in_navigation_mode(self) -> bool:
        """Is the robot to move around"""
        return self._control_mode == "navigation"

    def last_motion_failed(self) -> bool:
        """Override this if you want to check to see if a particular motion failed, e.g. it was not reachable and we don't know why."""
        return False

    def start(self) -> bool:
        """Override this if there's custom startup logic that you want to add before anything else.

        Returns True if we actually should do anything (like update) after this."""
        return False

    def get_robot_model(self) -> RobotModel:
        """return a model of the robot for planning"""
        raise NotImplementedError()

    def _update_obs(self, obs):
        """Update observation internally with lock"""
        with self._obs_lock:
            self._obs = obs
            self._control_mode = obs["control_mode"]

    def execute_trajectory(
        self,
        trajectory: List[np.ndarray],
        pos_err_threshold: float = 0.2,
        rot_err_threshold: float = 0.75,
        spin_rate: int = 10,
        verbose: bool = False,
        per_waypoint_timeout: float = 10.0,
        relative: bool = False,
    ):
        """Open loop trajectory execution"""
        raise NotImplementedError()

    def blocking_spin(self, verbose: bool = False, visualize: bool = False):
        """this is just for testing"""
        sum_time = 0
        steps = 0
        t0 = timeit.default_timer()
        camera = None
        shown_point_cloud = visualize

        while True:
            output = self.recv_socket.recv_pyobj()
            if output is None:
                continue
            output["rgb"] = cv2.imdecode(output["rgb"], cv2.IMREAD_COLOR)
            compressed_depth = output["depth"]
            depth = cv2.imdecode(compressed_depth, cv2.IMREAD_UNCHANGED)
            output["depth"] = depth / 1000.0

            if camera is None:
                camera = Camera.from_K(
                    output["camera_K"], output["rgb_height"], output["rgb_width"]
                )

            output["xyz"] = camera.depth_to_xyz(output["depth"])

            if visualize and not shown_point_cloud:
                show_point_cloud(output["xyz"], output["rgb"] / 255.0, orig=np.zeros(3))
                shown_point_cloud = True

            self._update_obs(output)
            with self._act_lock:
                if len(self._next_action) > 0:
                    # Send it
                    self.send_socket.send_pyobj(self._next_action)
                    # Empty it out for the next one
                    self._next_action = dict()

            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            if verbose:
                print("Control mode:", self._control_mode)
                print(
                    f"time taken = {dt} avg = {sum_time/steps} keys={[k for k in output.keys()]}"
                )
            t0 = timeit.default_timer()

            if self._control_mode == "navigation":
                self.move_to_manip_posture()
            elif self._control_mode == "manipulation":
                self.move_to_nav_posture()
                self.navigate_to([0, 0, 0], relative=False, blocking=True)


def main(local: bool = True, robot_ip: str = "192.168.1.15"):
    client = HomeRobotZmqClient(
        robot_ip=robot_ip,
        recv_port=4401,
        send_port=4402,
        use_remote_computer=(not local),
    )
    client.blocking_spin(verbose=True, visualize=False)


if __name__ == "__main__":
    main()
