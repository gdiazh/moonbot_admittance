# Copyright 2026 Gustavo Diaz
# SPDX-License-Identifier: Apache-2.0

import rclpy
import numpy as np
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from geometry_msgs.msg import WrenchStamped

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


class LowPassFilter:
    def __init__(self, alpha, size):
        self.alpha = alpha
        self.y = np.zeros(size)
        self.initialized = False

    def filter(self, x):
        if not self.initialized:
            self.y = x.copy()
            self.initialized = True
            return self.y

        self.y = self.alpha * x + (1 - self.alpha) * self.y
        return self.y


class AdmittanceController(Node):

    def __init__(self):
        super().__init__('adn_controller')
        self.joint_state_pub_ = self.create_publisher(JointState, "/leg302/joint_set", 100)
        self.fee_pub_ = self.create_publisher(WrenchStamped, "/F_ee", 30)

        self.jset_sub = self.create_subscription(JointState, '/leg302/joint_set_teleop', self.update_jset, 10)
        self.js_sub = self.create_subscription(JointState, '/leg302/joint_read', self.update_js, 10)
        self.wrench_sub = self.create_subscription(WrenchStamped, '/leptrino', self.update_wrench, 10)
        self.wrench = np.zeros(6)

        self.js = JointState()
        self.js_positions = [0.0] * 8
        self.jset_positions = [0.0] * 8
        self.gripper_position = 0

        self.admittance_calc_period = 1.0/100.0  # seconds
        self.admittance_calc_timer = self.create_timer(self.admittance_calc_period, self.admittance_loop)
        # self.admittance_calc_timer = self.create_timer(self.admittance_calc_period, self.position_control_loop)

        # robot
        robot_modules = [MhArmV3NoGripperJoints(2),]
        # urdf = wrap_in_robot(raw_urdf(robot_modules, base_link_name="base_link_dragon_V3", xyz=np.array([0.113, 0, 0.113])))
        urdf = wrap_in_robot(raw_urdf(robot_modules, base_link_name="leg302_in_base", xyz=np.zeros(3)))
        # 3. Save to a temporary file and load into the Robotics Toolbox
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

        self.desired_js_order = [
                'leg302_joint1', 'leg302_joint2', 'leg302_joint3', 
                'leg302_joint4', 'leg302_joint5', 'leg302_joint6', 'leg302_joint7', 'leg302_grip2'
            ]

        # -----------------------------
        # Admittance parameters
        # -----------------------------
        self.n = 7  # Number of arm DOF

        # Virtual inertia
        # Working_values_1 = [1.2, 10, 20] # Values for demo manually moving eef with hand
        Working_values_1 = [2, 180, 100] # Values for for brick insertion
        self.Mq = Working_values_1[0]*np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # Damping
        self.Dq = Working_values_1[1]*np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # Stiffness (low → compliant)
        self.Kq = Working_values_1[2]*np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        self.Minv = np.linalg.inv(self.Mq)

        # Insertion axis
        self.insertion_axis = np.array([np.cos(67.5*np.pi/180), -np.sin(67.5*np.pi/180), 0])  # Z axis in world frame translated into the T/F frame

        # -----------------------------
        # State variables
        # -----------------------------
        self.q_offset = np.zeros(self.n)
        self.qdot = np.zeros(self.n)

        # Low pass filter
        dt = 0.01   # 100 Hz loop
        fc = 2.0    # cutoff frequency

        alpha = dt / (dt + 1/(2*np.pi*fc))

        self.wrench_filter = LowPassFilter(alpha, 6)

    def update_js(self, msg):
        ordered_positions = [0.0] * 8 # 7 arm DOF + gripper
        try:
            for i, joint_name in enumerate(self.desired_js_order):
                # Find where the desired joint is in the incoming message
                msg_index = msg.name.index(joint_name)
                ordered_positions[i] = msg.position[msg_index]
                
            self.js_positions = ordered_positions
            
        except ValueError as e:
            self.get_logger().debug(f'Missing expected joint in message: {e}')

    def compute_gravity_wrench(self, R_be, m, r_com_ee):
        # g = np.array([0, 0, -9.81])                                       # base frame for static/pallet mount
        g = np.array([-9.81*np.cos(np.pi/4), 0, -9.81*np.sin(np.pi/4)])     # base frame after arm is mounted on the dragon

        # Force in base
        Fg_base = m * g

        # Transform to sensor/EE frame
        Fg_ee = R_be.T @ Fg_base

        # Torque from COM
        tau_ee = np.cross(r_com_ee, Fg_ee)

        return np.hstack((Fg_ee, tau_ee))

    def update_jset(self, msg):
        set_ordered_positions = [0.0] * 8
        try:
            for i, joint_name in enumerate(self.desired_js_order):
                # Find where the desired joint is in the incoming message
                msg_index = msg.name.index(joint_name)
                set_ordered_positions[i] = msg.position[msg_index]
                
            # self.get_logger().info(f'Set Ordered positions: {set_ordered_positions}')
            self.jset_positions = set_ordered_positions
            
        except ValueError as e:
            self.get_logger().debug(f'Missing expected joint in Set message: {e}')

    def update_wrench(self, msg):
        self.wrench = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

    def skew(self, p):
        return np.array([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]
        ])

    def wrench_transform(self, R, p):
        X = np.zeros((6,6))
        X[0:3, 0:3] = R
        X[3:6, 0:3] = self.skew(p) @ R
        X[3:6, 3:6] = R
        return X

    def apply_deadband(self, F, force_thresh=4.0, torque_thresh=0.1):
        F_filtered = F.copy()

        # Linear forces
        for i in range(3):
            if abs(F_filtered[i]) < force_thresh:
                F_filtered[i] = 0.0

        # Torques
        for i in range(3, 6):
            if abs(F_filtered[i]) < torque_thresh:
                F_filtered[i] = 0.0

        return F_filtered


    def is_force_valid(self, F, max_force=60.0, max_torque=15.0):
        # Check linear force magnitude
        if np.linalg.norm(F[:3]) > max_force:
            return False

        # Check torque magnitude
        if np.linalg.norm(F[3:]) > max_torque:
            return False

        return True

    def admittance_step(self, q_current, q_leader, F_ee, dt):
         # 1. Jacobian (base frame)
         #leg302_gripper2_base:+x->up, +y->left, +z->back // leg302_gripper2 or leg302_dragon_straight +x->forward, +y->left, +z->up
        J = self.robot.jacob0(q_current, end="leg302_tfs_cm_link")

        # 2. Forward kinematics, EEF Pose
        T = self.robot.fkine(q_current, end="leg302_tfs_cm_link") #leg302_gripper2 or leg302_dragon_straight
        R = T.R     # 3x3 rotation
        p = T.t     # position vector

        # 3. Transform wrench to base frame
        X = self.wrench_transform(R, p)
        F_base = X @ F_ee

        # 4. Convert to joint torques
        tau_ext = J.T @ F_base

        # Scale torque
        tau_ext *= 0.8

        # 5. Admittance dynamics
        # qddot = M^{-1} (tau - D*qdot - K*q_offset)
        qddot = self.Minv @ (tau_ext - self.Dq @ self.qdot - self.Kq @ self.q_offset)

        # Integration
        self.qdot += qddot * dt
        self.q_offset += self.qdot * dt

        # 6. Clamping
        self.q_offset = np.clip(self.q_offset, -3.0, 3.0)
        self.qdot = np.clip(self.qdot, -3.0, 3.0)

        # 7. Output compliant trajectory
        beta = 0.5
        q_target = q_leader + beta * self.q_offset

        return q_target

    def admittance_loop(self):
        if not (np.any(np.array(self.jset_positions))) or not (np.any(self.wrench)):
            print("Not initialized yet...")
            return
        dt = self.admittance_calc_period
        q_current = np.array(self.js_positions[:-1])
        q_leader = np.array(self.jset_positions[:-1])

        # Force sensor (EEF frame)
        F_ee = self.wrench

        # 2. Get current pose
        T = self.robot.fkine(q_current, end="leg302_tfs_cm_link") #leg302_gripper2 or leg302_dragon_straight
        R_be = T.R
        p = T.t

        # 3. Compute gravity
        F_grav = self.compute_gravity_wrench(R_be, 2.8, [-0.00480738, 0.0029812, 0.06152538]) #

        # 4. Subtract gravity and bias
        F_ee = F_ee - F_grav - np.array([3, 0, -4, 0, 0, 0]) # 

        # 5. Apply low pass filter
        F_ee = self.wrench_filter.filter(F_ee)

        # 6. For compliance only in the insertion axis
        F = F_ee[:3]
        F_parallel = np.dot(F, self.insertion_axis) * self.insertion_axis
        if np.linalg.norm(F_parallel) < 5.0:
            F_parallel = 0
        F_perp = F - F_parallel
        alpha = 0.4  # small response along insertion
        F_ee[:3] = F_perp + alpha * F_parallel

        # 7. Publish wrench
        fee_msg = WrenchStamped()
        fee_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        fee_msg.wrench.force.x = F_ee[0]
        fee_msg.wrench.force.y = F_ee[1]
        fee_msg.wrench.force.z = F_ee[2]
        fee_msg.wrench.torque.x = F_ee[3]
        fee_msg.wrench.torque.y = F_ee[4]
        fee_msg.wrench.torque.z = F_ee[5]
        self.fee_pub_.publish(fee_msg)

        # 8. Apply Deadband
        F_ee = self.apply_deadband(F_ee)

        # 9. Gripper limit check
        if self.jset_positions[-1]>0.0224:
            self.jset_positions[-1] = 0.0224
        
        # 10. Wrench safety check
        if not self.is_force_valid(F_ee):
            print("!!!!! Force sensor out of bounds, ignoring input force!!!!!")
            q_out = self.jset_positions
        else:
            q_target = self.admittance_step(q_current, q_leader, F_ee, dt)
            q_out = np.append(q_target, self.jset_positions[-1])
        
        joint_state_set = JointState()
        joint_state_set.header.stamp = rclpy.clock.Clock().now().to_msg()
        joint_state_set.name = ['leg302_joint1', 'leg302_joint2', 'leg302_joint3', 'leg302_joint4', 'leg302_joint5', 'leg302_joint6', 'leg302_joint7', 'leg302_grip2']
        joint_state_set.position = q_out

        self.joint_state_pub_.publish(joint_state_set)

    def position_control_loop(self):
        if not (np.any(np.array(self.jset_positions))):
            print("Not initialized yet...")
            return
        if self.jset_positions[-1]>0.0224:
            self.jset_positions[-1] = 0.0224
        q_out = self.jset_positions
        joint_state_set = JointState()
        joint_state_set.header.stamp = rclpy.clock.Clock().now().to_msg()
        joint_state_set.name = ['leg302_joint1', 'leg302_joint2', 'leg302_joint3', 'leg302_joint4', 'leg302_joint5', 'leg302_joint6', 'leg302_joint7', 'leg302_grip2']
        joint_state_set.position = q_out

        self.joint_state_pub_.publish(joint_state_set)


def main(args=None):
    rclpy.init(args=args)

    adm_controller_nh = AdmittanceController()

    rclpy.spin(adm_controller_nh)

    # Destroy the node
    adm_controller_nh.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()