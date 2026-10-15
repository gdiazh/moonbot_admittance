# Copyright 2026 Gustavo Diaz
# SPDX-License-Identifier: Apache-2.0

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped

import numpy as np
import threading
import time

import roboticstoolbox as rtb

from moonbot_launcher.urdf.base import Joint, Link, wrap_in_robot
from typing import List
from moonbot_launcher.assemblies.alone import alone, raw_urdf
from moonbot_launcher.modules.hero import (
    MhArmV2,
    MhArmV3,
    MhArmV3NoGripperJoints,
)
import tempfile
import os
from spatialmath import SE3
import csv


def skew(p):
    return np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])


class GravityCalibrationNode(Node):

    def __init__(self):
        super().__init__('gravity_calibration_node')

        # -----------------------------
        # Robot model
        # -----------------------------
        # robot
        robot_modules = [MhArmV3NoGripperJoints(2),]
        urdf = wrap_in_robot(raw_urdf(robot_modules, base_link_name="leg302_in_base", xyz=np.zeros(3)))
        # 3. Save to a temporary file and load it into the Robotics Toolbox
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.urdf') as temp_file:
            temp_file.write(urdf)
            temp_filepath = temp_file.name
        try:
            # Load the robot model
            self.robot = rtb.ERobot.URDF(temp_filepath)
            print(self.robot)
        finally:
            # Clean up the temporary file
            os.remove(temp_filepath)

        # -----------------------------
        # Subscribers
        # -----------------------------
        self.joint_msg = None
        self.wrench_msg = None

        self.create_subscription(JointState, '/leg302/joint_read', self.joint_callback, 10)

        self.create_subscription(WrenchStamped, '/leptrino', self.wrench_callback, 10)

        self.desired_js_order = [
                'leg302_joint1', 'leg302_joint2', 'leg302_joint3', 
                'leg302_joint4', 'leg302_joint5', 'leg302_joint6', 'leg302_joint7'
            ]
        self.joint_index_map = None

        # -----------------------------
        # Storage
        # -----------------------------
        self.samples = []

        # Thread for user input
        self.thread = threading.Thread(target=self.user_loop)
        self.thread.start()

        self.csv_file = "gravity_calibration_data.csv"

        n_q = self.robot.n  # number of joints

        header = ["sample_id"]

        # Joint names
        header += self.desired_js_order

        # Force
        header += ["Fx", "Fy", "Fz"]

        # Torque
        header += ["Tx", "Ty", "Tz"]

        # Create file with header if it doesn't exist
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        self.load_existing_data()

    # -----------------------------
    # Callbacks
    # -----------------------------
    def joint_callback(self, msg):
        self.joint_msg = msg

        # Build mapping once
        if self.joint_index_map is None:
            self.joint_index_map = {}

            for i, name in enumerate(msg.name):
                self.joint_index_map[name] = i

            # Check all required joints exist
            for jn in self.desired_js_order:
                if jn not in self.joint_index_map:
                    self.get_logger().error(f"Joint {jn} not found in JointState!")

    def wrench_callback(self, msg):
        self.wrench_msg = msg

    # -----------------------------
    # Data collection
    # -----------------------------

    def save_sample_to_csv(self, sample_id, q, F, tau):

        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)

            row = [sample_id]
            row.extend(q.tolist())
            row.extend(F.tolist())
            row.extend(tau.tolist())

            writer.writerow(row)

    def load_existing_data(self):

        if not os.path.exists(self.csv_file):
            return

        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header

            for row in reader:
                row = list(map(float, row))

                sample_id = int(row[0])
                n_q = self.robot.n

                q = np.array(row[1:1+n_q])
                F = np.array(row[1+n_q:1+n_q+3])
                tau = np.array(row[1+n_q+3:1+n_q+6])

                self.samples.append((q, F, tau))

        self.get_logger().info(f"Loaded {len(self.samples)} existing samples from CSV.")

    def get_ordered_q(self):

        if self.joint_msg is None:
            return None

        name_to_pos = dict(zip(self.joint_msg.name, self.joint_msg.position))

        q = []

        for jn in self.desired_js_order:
            if jn not in name_to_pos:
                return None
            q.append(name_to_pos[jn])

        return np.array(q)

    def collect_sample(self, duration=60.0):

        self.get_logger().info(f"Collecting data for {duration} seconds...")

        F_list = []
        tau_list = []
        q_list = []

        start = time.time()

        while time.time() - start < duration:
            if self.joint_msg is None or self.wrench_msg is None:
                continue

            q = self.get_ordered_q()
            if q is None:
                continue

            F = np.array([
                self.wrench_msg.wrench.force.x,
                self.wrench_msg.wrench.force.y,
                self.wrench_msg.wrench.force.z
            ])

            tau = np.array([
                self.wrench_msg.wrench.torque.x,
                self.wrench_msg.wrench.torque.y,
                self.wrench_msg.wrench.torque.z
            ])

            q_list.append(q)
            F_list.append(F)
            tau_list.append(tau)

            time.sleep(0.01)  # ~100 Hz

        q_avg = np.mean(q_list, axis=0)
        F_avg = np.mean(F_list, axis=0)
        tau_avg = np.mean(tau_list, axis=0)

        self.get_logger().info(f"Collected sample: F={F_avg}, tau={tau_avg}")

        return q_avg, F_avg, tau_avg

    # -----------------------------
    # Identification
    # -----------------------------
    def compute_mass_and_com(self):

        if len(self.samples) < 2:
            self.get_logger().error("Not enough samples!")
            return

        # g = np.array([0, 0, -9.81])                                       # robot on pallet
        g = np.array([-9.81*np.cos(np.pi/4), 0, -9.81*np.sin(np.pi/4)])     # robot on dragon (wheel module)

        # Reference pose (first one)
        q0, F0, _ = self.samples[0]
        R0 = self.robot.fkine(q0, end="leg302_tfs_cm_link").R
        g0 = R0.T @ g
        print("g0: ", g0)

        # -----------------------------
        # Mass estimation
        # -----------------------------
        m_estimates = []

        for (q, F, _) in self.samples[1:]:
            R = self.robot.fkine(q, end="leg302_tfs_cm_link").R
            gi = R.T @ g

            denom = np.linalg.norm(gi - g0)
            if denom > 1e-6:
                m_estimates.append(np.linalg.norm(F) / denom)

        m = np.mean(m_estimates)

        # -----------------------------
        # COM estimation
        # -----------------------------
        A = []
        b = []

        for (_, F, tau) in self.samples:
            A.append(skew(F).T)
            b.append(tau)

        A = np.vstack(A)
        b = np.hstack(b)

        r_com, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        self.get_logger().info(f"\nEstimated mass: {m:.4f} kg")
        self.get_logger().info(f"Estimated COM (sensor frame) [m]: {r_com}")

    # -----------------------------
    # Main loop
    # -----------------------------
    def user_loop(self):

        while rclpy.ok():

            print(f"\n Current number of samples: {len(self.samples)}")

            cmd = input("Type 'yes' when ready for next pose, or 'stop': ")

            if cmd.lower() == "yes":

                sample_id = len(self.samples)

                q_avg, F_avg, tau_avg = self.collect_sample(duration=5.0)

                self.samples.append((q_avg, F_avg, tau_avg))

                # Save immediately
                self.save_sample_to_csv(sample_id, q_avg, F_avg, tau_avg)

                print(f" Sample {sample_id} saved.")

            elif cmd.lower() == "stop":

                self.get_logger().info("Computing results...")
                self.compute_mass_and_com()
                break

            else:
                print("Unknown command. Use 'yes' or 'stop'.")


def main(args=None):
    rclpy.init(args=args)
    node = GravityCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()