#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class SafetyNode(Node):
    """
    The class that handles emergency braking.
    """

    def __init__(self):
        super().__init__('safety_node')
        """
        One publisher should publish to the /drive topic with a AckermannDriveStamped drive message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /ego_racecar/odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed = 0.

        # iTTC (seconds) below which we consider a collision imminent
        self.declare_parameter('ttc_threshold', 1.0)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10)

    def odom_callback(self, odom_msg):
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        ranges = np.array(scan_msg.ranges)
        angles = scan_msg.angle_min + \
            np.arange(len(ranges)) * scan_msg.angle_increment

        # Only trust real, in-range returns; ignore inf/nan (no return)
        # and anything outside the sensor's rated range.
        valid = np.isfinite(ranges) & \
            (ranges >= scan_msg.range_min) & \
            (ranges <= scan_msg.range_max)

        # Range rate r_dot = -v_x * cos(theta): projecting the vehicle's
        # forward speed onto each beam gives how fast that beam's range
        # is closing. {-r_dot}_+ = max(v_x * cos(theta), 0) so beams to
        # the side/rear (which aren't closing due to forward motion)
        # get an iTTC of infinity instead of a meaningless value.
        closing_rate = self.speed * np.cos(angles)
        denom = np.maximum(closing_rate, 0.0)

        ittc = np.full_like(ranges, np.inf)
        can_compute = valid & (denom > 1e-3)
        ittc[can_compute] = ranges[can_compute] / denom[can_compute]

        min_ittc = np.min(ittc) if ittc.size else np.inf

        ttc_threshold = self.get_parameter(
            'ttc_threshold').get_parameter_value().double_value

        if min_ittc < ttc_threshold:
            self.brake()

    def brake(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.speed = 0.0
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
