# (c) 2024 chris paxton under MIT license

import rclpy
import zmq

from robot_hw_python.remote import StretchClient


def main():
    rclpy.init()
    client = StretchClient()

    # Create a stretch client to get information
    print(client)


if __name__ == "__main__":
    main()
