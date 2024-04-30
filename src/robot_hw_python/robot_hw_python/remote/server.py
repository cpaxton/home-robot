# (c) 2024 chris paxton for Hello Robot, under MIT license

import time
import timeit

import cv2
import numpy as np
import rclpy
import zmq

from robot_hw_python.remote import StretchClient


def main(port=4401, use_remote_computer: bool = True):
    rclpy.init()
    client = StretchClient()

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.RCVHWM, 1)
    if use_remote_computer:
        address = "tcp://*:" + str(port)
    else:
        address = "tcp://127.0.0.1:" + str(port)
    socket.bind(address)

    # Create a stretch client to get information
    sum_time: float = 0
    steps = 0
    t0 = timeit.default_timer()
    while rclpy.ok():
        # get information
        obs = client.get_observation()
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
        }

        socket.send_pyobj(data)
        t1 = timeit.default_timer()
        dt = t1 - t0
        sum_time += dt
        steps += 1
        t0 = t1
        print(f"time taken = {dt} avg = {sum_time/steps}")

        time.sleep(0.1)
        t0 = timeit.default_timer()


if __name__ == "__main__":
    main()
