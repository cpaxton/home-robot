# (c) 2024 chris paxton under MIT license

import threading
import time
import timeit
from threading import Lock
from typing import List, Optional

import click
import cv2
import numpy as np
import rclpy
import zmq

from home_robot.core.interfaces import ContinuousNavigationAction, Observations
from home_robot.core.robot import RobotClient
from home_robot.motion.robot import RobotModel
from home_robot.motion.stretch import HelloStretchKinematics
from home_robot.utils.image import Camera
from home_robot.utils.point_cloud import show_point_cloud

# import faulthandler
# faulthandler.enable()


class HomeRobotZmqClient(RobotClient):

    _iter = 0  # Tracks number of actions set, never reset this

    def __init__(
        self,
        robot_ip: str = "192.168.1.15",
        recv_port: int = 4401,
        send_port: int = 4402,
        use_remote_computer: bool = True,
        urdf_path: str = "",
        ik_type: str = "pinocchio",
        visualize_ik: bool = False,
        grasp_frame: Optional[str] = None,
        ee_link_name: Optional[str] = None,
        manip_mode_controlled_joints: Optional[List[str]] = None,
    ):
        """
        Create a client to communicate with the robot over ZMQ.

        Args:
            robot_ip: The IP address of the robot
            recv_port: The port to receive observations on
            send_port: The port to send actions to on the robot
            use_remote_computer: Whether to use a remote computer to connect to the robot
            urdf_path: The path to the URDF file for the robot
            ik_type: The type of IK solver to use
            visualize_ik: Whether to visualize the IK solution
            grasp_frame: The frame to use for grasping
            ee_link_name: The name of the end effector link
            manip_mode_controlled_joints: The joints to control in manipulation mode
        """
        self.recv_port = recv_port
        self.send_port = send_port
        self.reset()

        # Robot model
        self._robot_model = HelloStretchKinematics(
            urdf_path=urdf_path,
            ik_type=ik_type,
            visualize=visualize_ik,
            grasp_frame=grasp_frame,
            ee_link_name=ee_link_name,
            manip_mode_controlled_joints=manip_mode_controlled_joints,
        )

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

        print(f"Connecting to {self.address} to receive observations...")
        self.recv_socket.connect(self.address)
        print("...connected.")

        self._obs_lock = Lock()
        self._act_lock = Lock()

    def get_base_pose(self) -> np.ndarray:
        """Get the current pose of the base"""
        with self._obs_lock:
            if self._obs is None:
                return None
            gps = self._obs["gps"]
            compass = self._obs["compass"]
        return np.concatenate([gps, compass], axis=-1)

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
        self.send_action()
        with self._obs_lock:
            # Clear observation
            self._obs = None

    def reset(self):
        """Reset everything in the robot's internal state"""
        self._control_mode = None
        self._next_action = dict()
        self._obs = None
        self._thread = None
        self._finish = False
        self._last_step = -1

    def switch_to_navigation_mode(self):
        with self._act_lock:
            self._next_action["control_mode"] = "navigation"
        self.send_action()
        self._wait_for_mode("navigation")

    def switch_to_manipulation_mode(self):
        with self._act_lock:
            self._next_action["control_mode"] = "manipulation"
        self.send_action()
        self._wait_for_mode("manipulation")

    def move_to_nav_posture(self):
        with self._act_lock:
            self._next_action["posture"] = "navigation"
        self.send_action()
        self._wait_for_mode("navigation")

    def move_to_manip_posture(self):
        with self._act_lock:
            self._next_action["posture"] = "manipulation"
        self.send_action()
        self._wait_for_mode("manipulation")

    def _wait_for_mode(self, mode, verbose: bool = False):
        t0 = timeit.default_timer()
        while True:
            with self._obs_lock:
                if verbose:
                    print(f"Waiting for mode {mode} current mode {self._control_mode}")
                if self._control_mode == mode:
                    break
            time.sleep(0.1)
            t1 = timeit.default_timer()
            if t1 - t0 > 5.0:
                raise RuntimeError(f"Timeout waiting for mode {mode}")

    def _wait_for_action(self, block_id: int, verbose: bool = False):
        t0 = timeit.default_timer()
        while True:
            with self._obs_lock:
                if verbose:
                    print(
                        f"Waiting for step {block_id} last acknowledged step {self._last_step}"
                    )
                if self._last_step >= block_id:
                    break
            time.sleep(0.1)
            t1 = timeit.default_timer()
            if t1 - t0 > 15.0:
                raise RuntimeError(
                    f"Timeout waiting for block with step id = {block_id}"
                )

    def in_manipulation_mode(self) -> bool:
        """is the robot ready to grasp"""
        return self._control_mode == "manipulation"

    def in_navigation_mode(self) -> bool:
        """Is the robot to move around"""
        return self._control_mode == "navigation"

    def last_motion_failed(self) -> bool:
        """Override this if you want to check to see if a particular motion failed, e.g. it was not reachable and we don't know why."""
        return False

    def get_robot_model(self) -> RobotModel:
        """return a model of the robot for planning"""
        return self._robot_model

    def _update_obs(self, obs):
        """Update observation internally with lock"""
        with self._obs_lock:
            self._obs = obs
            self._control_mode = obs["control_mode"]
            self._last_step = obs["step"]

    def get_observation(self):
        """Get the current observation"""
        with self._obs_lock:
            if self._obs is None:
                return None
            observation = Observations(
                gps=self._obs["gps"],
                compass=self._obs["compass"],
                rgb=self._obs["rgb"],
                depth=self._obs["depth"],
                xyz=self._obs["xyz"],
            )
            observation.joint = self._obs.get("joint", None)
            observation.ee_pose = self._obs.get("ee_pose", None)
            observation.camera_K = self._obs.get("camera_K", None)
            observation.camera_pose = self._obs.get("camera_pose", None)
        return observation

    def send_action(self):
        print("sending", self._next_action)
        blocking = False
        block_id = None
        with self._act_lock:
            blocking = self._next_action.get("nav_blocking", False)
            block_id = self._iter
            # Send it
            self._next_action["step"] = block_id
            self._iter += 1
            self.send_socket.send_pyobj(self._next_action)
            # Empty it out for the next one
            self._next_action = dict()
            time.sleep(0.2)

        if blocking:
            # Wait for the command to
            self._wait_for_action(block_id)
        print("done sending")

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

        while not self._finish:

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
            # with self._act_lock:
            #    if len(self._next_action) > 0:

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

    def start(self) -> bool:
        """Start running blocking thread in a separate thread"""
        self._thread = threading.Thread(target=self.blocking_spin)
        self._finish = False
        self._thread.start()
        return True

    def __del__(self):
        self._finish = True
        self.recv_socket.close()
        self.send_socket.close()
        self.context.term()
        if self._thread is not None:
            self._thread.join()


@click.command()
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
@click.option("--recv_port", default=4401, help="Port to receive observations on")
@click.option("--send_port", default=4402, help="Port to send actions to on the robot")
@click.option("--robot_ip", default="192.168.1.15")
def main(
    local: bool = True,
    recv_port: int = 4401,
    send_port: int = 4402,
    robot_ip: str = "192.168.1.15",
):
    client = HomeRobotZmqClient(
        robot_ip=robot_ip,
        recv_port=recv_port,
        send_port=send_port,
        use_remote_computer=(not local),
    )
    client.blocking_spin(verbose=True, visualize=False)


if __name__ == "__main__":
    main()
