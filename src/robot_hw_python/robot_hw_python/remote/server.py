#!/usr/bin/env python
# (c) 2024 chris paxton for Hello Robot, under MIT license

import threading
import time
import timeit
from typing import Optional

import click
import cv2
import numpy as np
import rclpy
import zmq

import home_robot.utils.compression as compression
from home_robot.core.interfaces import ContinuousNavigationAction
from robot_hw_python.remote import StretchClient


class ZmqServer:

    # How often should we print out info about our performance
    report_steps = 100
    fast_report_steps = 10000
    debug_compression: bool = False

    def _make_pub_socket(self, send_port, use_remote_computer: bool = True):
        socket = self.context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.RCVHWM, 1)

        if use_remote_computer:
            send_address = "tcp://*:" + str(send_port)
        else:
            desktop_ip = "127.0.0.1"
            send_address = f"tcp://{desktop_ip}:" + str(send_port)

        print(f"Publishing on {send_address}...")
        socket.bind(send_address)
        return socket

    def __init__(
        self,
        send_port: int = 4401,
        recv_port: int = 4402,
        send_state_port: int = 4403,
        send_servo_port: int = 4404,
        use_remote_computer: bool = True,
        verbose: bool = False,
        image_scaling: float = 0.25,
        ee_image_scaling: float = 1.0,  # 0.6,
    ):
        self.verbose = verbose
        self.client = StretchClient(d405=True)
        self.context = zmq.Context()
        self.image_scaling = image_scaling
        self.ee_image_scaling = ee_image_scaling

        # Set up the publisher socket using ZMQ
        self.send_socket = self._make_pub_socket(send_port, use_remote_computer)

        # Publisher for state-only messages (FAST spin rate)
        self.send_state_socket = self._make_pub_socket(
            send_state_port, use_remote_computer
        )

        # Publisher for visual servoing images (lower size, faster publishing rate)
        self.send_servo_socket = self._make_pub_socket(
            send_servo_port, use_remote_computer
        )

        # Set up the receiver/subscriber using ZMQ
        self.recv_socket = self.context.socket(zmq.SUB)
        self.recv_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.recv_socket.setsockopt(zmq.SNDHWM, 1)
        self.recv_socket.setsockopt(zmq.RCVHWM, 1)
        self.recv_socket.setsockopt(zmq.CONFLATE, 1)
        self._last_step = -1

        # Make connections
        if use_remote_computer:
            recv_address = "tcp://*:" + str(recv_port)
        else:
            desktop_ip = "127.0.0.1"
            recv_address = f"tcp://{desktop_ip}:" + str(recv_port)

        print(f"Listening on {recv_address}...")
        self.recv_socket.bind(recv_address)
        self.recv_address = recv_address
        print("Done!")

        # for the threads
        self.control_mode = "none"
        self._done = False

    def get_control_mode(self):
        if self.client.in_manipulation_mode():
            control_mode = "manipulation"
        elif self.client.in_navigation_mode():
            control_mode = "navigation"
        else:
            control_mode = "none"
        return control_mode

    def spin_send(self):

        # Create a stretch client to get information
        sum_time: float = 0
        steps: int = 0
        t0 = timeit.default_timer()
        while rclpy.ok() and not self._done:
            # get information
            # Still about 0.01 seconds to get observations
            obs = self.client.get_observation(compute_xyz=False)
            rgb, depth = obs.rgb, obs.depth
            width, height = rgb.shape[:2]

            # Convert depth into int format
            depth = (depth * 1000).astype(np.uint16)

            # Make both into jpegs
            _, rgb = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
            _, depth = cv2.imencode(
                ".jp2", depth, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 800]
            )

            # Get the other fields from an observation
            # rgb = compression.to_webp(rgb)
            data = {
                "rgb": rgb,
                "depth": depth,
                "camera_K": obs.camera_K.cpu().numpy(),
                "camera_pose": obs.camera_pose,
                "joint": obs.joint,
                "gps": obs.gps,
                "compass": obs.compass,
                "rgb_width": width,
                "rgb_height": height,
                "control_mode": self.get_control_mode(),
                "last_motion_failed": self.client.last_motion_failed(),
                "recv_address": self.recv_address,
                "step": self._last_step,
                "at_goal": self.client.at_goal(),
            }
            self.send_socket.send_pyobj(data)

            # Finish with some speed info
            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            t0 = t1
            if self.verbose or steps % self.report_steps == 0:
                print(f"[SEND FULL STATE] time taken = {dt} avg = {sum_time/steps}")

            time.sleep(1e-4)
            t0 = timeit.default_timer()

    def spin_send_state(self):
        """Send a faster version of the state for tracking joint states and robot base"""
        # Create a stretch client to get information
        sum_time: float = 0
        steps: int = 0
        t0 = timeit.default_timer()
        while rclpy.ok() and not self._done:
            q, dq, eff = self.client.get_joint_state()
            message = {
                "base_pose": self.client.get_base_pose(),
                "joint_positions": q,
                "joint_velocities": dq,
                "joint_efforts": eff,
                "control_mode": self.get_control_mode(),
            }
            self.send_state_socket.send_pyobj(message)

            # Finish with some speed info
            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            t0 = t1
            if self.verbose or steps % self.fast_report_steps == 0:
                print(f"[SEND FAST STATE] time taken = {dt} avg = {sum_time/steps}")

            time.sleep(1e-4)
            t0 = timeit.default_timer()

    def spin_recv(self):
        sum_time: float = 0
        steps = 0
        t0 = timeit.default_timer()
        while rclpy.ok() and not self._done:
            try:
                action = self.recv_socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                if self.verbose:
                    print(" - no action received")
                action = None
            if self.verbose:
                print(f" - {self.control_mode=}")
                print(f" - prev action step: {self._last_step}")
            if action is not None:
                if True or self.verbose:
                    print(f" - Action received: {action}")
                self._last_step = action.get("step", -1)
                print(
                    f"Action #{self._last_step} received:",
                    [str(key) for key in action.keys()],
                )
                if self.verbose:
                    print(f" - last action step: {self._last_step}")
                if "posture" in action:
                    if action["posture"] == "manipulation":
                        self.client.move_to_manip_posture()
                        self.client.switch_to_manipulation_mode()
                    elif action["posture"] == "navigation":
                        self.client.move_to_nav_posture()
                        self.client.switch_to_navigation_mode()
                    else:
                        print(
                            " - posture",
                            action["posture"],
                            "not recognized or supported.",
                        )
                if "control_mode" in action:
                    if action["control_mode"] == "manipulation":
                        self.client.switch_to_manipulation_mode()
                        self.control_mode = "manipulation"
                    elif action["control_mode"] == "navigation":
                        self.client.switch_to_navigation_mode()
                        self.control_mode = "navigation"
                    else:
                        print(
                            " - control mode",
                            action["control_mode"],
                            "not recognized or supported.",
                        )
                if "xyt" in action:
                    if self.verbose:
                        print(
                            "Is robot in navigation mode?",
                            self.client.in_navigation_mode(),
                        )
                        print(
                            f"{action['xyt']} {action['nav_relative']} {action['nav_blocking']}"
                        )
                    self.client.navigate_to(
                        action["xyt"],
                        relative=action["nav_relative"],
                    )
                if "joint" in action:
                    # This allows for executing motor commands on the robot relatively quickly
                    if self.verbose:
                        print(f"Moving arm to config={action['joint']}")
                    self.client.arm_to(action["joint"])
                if "gripper" in action:
                    if self.verbose or True:
                        print(f"Moving gripper to {action['gripper']}")
                    self.client.manip.move_gripper(action["gripper"])

            # Finish with some speed info
            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            t0 = t1
            if self.verbose or steps % self.fast_report_steps == 0:
                print(f"[RECV] time taken = {dt} avg = {sum_time/steps}")

            time.sleep(1e-4)
            t0 = timeit.default_timer()

    def _rescale_color_and_depth(self, color_image, depth_image, scaling: float = 0.5):
        color_image = cv2.resize(
            color_image,
            (0, 0),
            fx=scaling,
            fy=scaling,
            interpolation=cv2.INTER_AREA,
        )
        depth_image = cv2.resize(
            depth_image,
            (0, 0),
            fx=scaling,
            fy=scaling,
            interpolation=cv2.INTER_NEAREST,
        )
        return color_image, depth_image

    def spin_send_servo(self):
        """Send the images here as well"""
        sum_time: float = 0
        steps: int = 0
        t0 = timeit.default_timer()

        # depth_camera_info, color_camera_info = self.ee_cam.get_camera_infos()
        # head_depth_camera_info, head_color_camera_info = self.head_cam.get_camera_infos()
        # depth_scale = self.ee_cam.get_depth_scale()
        # head_depth_scale = self.head_cam.get_depth_scale()

        while not self._done:
            # Read images from the end effector and head cameras
            obs = self.client.get_observation(compute_xyz=False)
            head_color_image, head_depth_image = self._rescale_color_and_depth(
                obs.rgb, obs.depth, self.image_scaling
            )
            ee_depth_image = self.client.ee_dpt_cam.get()
            ee_color_image = self.client.ee_rgb_cam.get()
            ee_color_image, ee_depth_image = self._rescale_color_and_depth(
                ee_color_image, ee_depth_image, self.ee_image_scaling
            )
            # depth_image, color_image = self._get_images(from_head=False, verbose=verbose)

            if self.debug_compression:
                ct0 = timeit.default_timer()
            compressed_ee_depth_image = compression.zip_depth(ee_depth_image)
            compressed_head_depth_image = compression.zip_depth(head_depth_image)
            if self.debug_compression:
                ct1 = timeit.default_timer()
            compressed_ee_color_image = compression.to_webp(ee_color_image)
            compressed_head_color_image = compression.to_webp(head_color_image)
            # compressed_head_color_image = compressed_ee_color_image
            if self.debug_compression:
                # TODO: remove debug code
                # print(f"{ee_color_image.shape=}")
                # print(f"{ee_depth_image.shape=}")
                # print(f"{head_color_image.shape=}")
                # print(f"{head_depth_image.shape=}")
                ct2 = timeit.default_timer()
                print(
                    ct1 - ct0,
                    f"{len(compressed_head_depth_image)=}",
                    ct2 - ct1,
                    f"{len(compressed_head_color_image)=}",
                )

            d405_output = {
                "ee_cam/color_camera_K": self.client.ee_rgb_cam.get_K(),
                "ee_cam/depth_camera_K": self.client.ee_dpt_cam.get_K(),
                "ee_cam/color_image": compressed_ee_color_image,
                "ee_cam/color_image/shape": ee_color_image.shape,
                "ee_cam/depth_image": compressed_ee_depth_image,
                "ee_cam/depth_image/shape": ee_depth_image.shape,
                "ee_cam/image_scaling": self.ee_image_scaling,
                "head_cam/color_camera_K": self.client.rgb_cam.get_K(),
                "head_cam/depth_camera_K": self.client.dpt_cam.get_K(),
                "head_cam/color_image": compressed_head_color_image,
                "head_cam/color_image/shape": head_color_image.shape,
                "head_cam/depth_image": compressed_head_depth_image,
                "head_cam/depth_image/shape": head_depth_image.shape,
                "head_cam/image_scaling": self.image_scaling,
                "robot/config": obs.joint,
                # "robot/ee_position": ee_pos,
                # "robot/ee_rotation": ee_rot,
            }

            self.send_servo_socket.send_pyobj(d405_output)

            # Finish with some speed info
            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            t0 = t1
            # if self.verbose or steps % self.fast_report_steps == 1:
            if self.verbose or steps % 100 == 1:
                print(
                    f"[SEND SERVO STATE] time taken = {dt} avg = {sum_time/steps} rate={1/(sum_time/steps)}"
                )

            time.sleep(1e-5)
            t0 = timeit.default_timer()

    def start(self):
        """Starts both threads spinning separately for efficiency."""
        self._send_thread = threading.Thread(target=self.spin_send)
        self._recv_thread = threading.Thread(target=self.spin_recv)
        self._send_state_thread = threading.Thread(target=self.spin_send_state)
        self._send_servo_thread = threading.Thread(target=self.spin_send_servo)
        self._done = False
        self._send_thread.start()
        self._recv_thread.start()
        self._send_state_thread.start()
        self._send_servo_thread.start()

    def __del__(self):
        self._done = True
        # Wait for the threads to finish
        time.sleep(0.15)

        # Close threads
        self._send_thread.join()
        self._recv_thread.join()
        self._send_state_thread.join()
        self._send_servo_thread.join()

        # Close sockets
        self.recv_socket.close()
        self.send_socket.close()
        self.send_state_socket.close()
        self.context.term()


@click.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
@click.option("--send_port", default=4401, help="Port to send observations to")
@click.option("--recv_port", default=4402, help="Port to receive actions from")
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
def main(
    send_port: int = 4401,
    recv_port: int = 4402,
    local: bool = False,
):
    rclpy.init()
    server = ZmqServer(
        send_port=send_port,
        recv_port=recv_port,
        use_remote_computer=(not local),
    )
    server.start()


if __name__ == "__main__":
    main()
