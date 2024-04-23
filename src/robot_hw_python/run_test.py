import rclpy

rclpy.init()

from robot_hw_python.remote import StretchClient

# replace the path for hab_Stretch
r = StretchClient(
    urdf_path="/home/hello-robot/yaswanth/robot-controller/src/home_robot_hw/assets/hab_stretch/urdf"
)

# Navigation
r.nav.navigate_to([0.5, 0, 0])

# Manipulation
# r.switch_to_manipulation_mode()
# state = r.manip.get_joint_positions()
# state[1] = 0.8 # Lift
# r.manip.goto_joint_positions(state)


# Head roatetions
# r.switch_to_manipulation_mode()
# r.head.set_pan_tilt(0.1)
