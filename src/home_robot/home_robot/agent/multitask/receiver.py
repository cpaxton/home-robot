# (c) 2024 chris paxton under MIT license

import time
import timeit

import click
import cv2
import numpy as np
import rclpy
import zmq

from home_robot.utils.image import Camera
from home_robot.utils.point_cloud import show_point_cloud


class HomeRobotZmqClient:
    def __init__(
        self,
        robot_ip: str = "192.168.1.15",
        recv_port: int = 4401,
        send_port: int = 4402,
        use_remote_computer: bool = True,
    ):
        self.recv_port = recv_port
        self.send_port = send_port

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

    def blocking_spin(self, verbose: bool = False, visualize: bool = False):
        """this is just for testing"""
        sum_time = 0
        steps = 0
        t0 = timeit.default_timer()
        camera = None
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
            if visualize:
                show_point_cloud(output["xyz"], output["rgb"] / 255.0, orig=np.zeros(3))

            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            if verbose:
                print(
                    f"time taken = {dt} avg = {sum_time/steps} keys={[k for k in output.keys()]}"
                )
            t0 = timeit.default_timer()


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
