---
sidebar_position: 9
---

# Week 8: Locomotion Algorithms

## Learning Objectives

By the end of this week, you will be able to:
- Understand different types of robot locomotion
- Implement basic walking algorithms for legged robots
- Design gait patterns for stable locomotion
- Calculate inverse kinematics for leg control
- Implement balance control for dynamic locomotion

## 8.1 Introduction to Robot Locomotion

Robot locomotion is the capability of a robot to move through its environment. Different locomotion types are suited for different applications:

- **Wheeled**: Efficient on flat surfaces, high speed
- **Tracked**: Good for rough terrain, high traction
- **Legged**: Versatile terrain navigation, obstacle climbing
- **Aerial**: 3D navigation, access to hard-to-reach areas
- **Swimming**: Underwater applications

For humanoid robots, legged locomotion presents unique challenges and opportunities.

### 8.1.1 Locomotion Challenges
```
┌─────────────────────────────────────────────────────────────┐
│              Locomotion Challenges                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Balance    │  │  Terrain    │  │  Dynamic   │        │
│  │  Control    │  │  Adaption   │  │  Stability │        │
│  │             │  │             │  │             │        │
│  │ • COM       │  │ • Foot      │  │ • Gait      │        │
│  │   Control   │  │   Placement │  │   Timing    │        │
│  │ • Fall      │  │ • Obstacle  │  │ • Impact    │        │
│  │   Prevention│  │   Negotiation│ │   Control   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 8.2 Types of Locomotion

### 8.2.1 Static vs Dynamic Locomotion
- **Static Locomotion**: Robot is always in stable equilibrium
- **Dynamic Locomotion**: Robot uses momentum and dynamics for movement

### 8.2.2 Gait Classification
- **Walking**: Alternating support phases
- **Running**: Both feet leave ground simultaneously
- **Trotting**: Diagonal leg pairs move together
- **Pacing**: Lateral leg pairs move together

## 8.3 Walking Algorithms

### 8.3.1 Inverted Pendulum Model
The simplest model for walking is the inverted pendulum:

```python
# inverted_pendulum.py
import numpy as np
import matplotlib.pyplot as plt


class InvertedPendulumController:
    def __init__(self, robot_height=0.8, gravity=9.81):
        self.robot_height = robot_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / robot_height)

        # State: [x, vx, y, vy] (CoM position and velocity)
        self.state = np.zeros(4)
        self.support_foot = np.array([0.0, 0.0])  # Position of support foot

    def step(self, dt):
        """Update pendulum state for one time step."""
        x, vx, y, vy = self.state

        # Inverted pendulum dynamics
        # d²x/dt² = ω² * (x - x_support)
        x_ddot = self.omega**2 * (x - self.support_foot[0])
        y_ddot = self.omega**2 * (y - self.support_foot[1])

        # Update state
        self.state[0] += vx * dt + 0.5 * x_ddot * dt**2
        self.state[1] += x_ddot * dt
        self.state[2] += vy * dt + 0.5 * y_ddot * dt**2
        self.state[3] += y_ddot * dt

    def set_support_foot(self, foot_pos):
        """Set the position of the support foot."""
        self.support_foot = foot_pos

    def is_stable(self):
        """Check if the pendulum is within stable bounds."""
        margin = 0.1  # Stability margin
        return (abs(self.state[0] - self.support_foot[0]) < margin and
                abs(self.state[2] - self.support_foot[1]) < margin)


class WalkingController:
    def __init__(self, step_length=0.3, step_height=0.05, step_time=0.8):
        self.step_length = step_length
        self.step_height = step_height
        self.step_time = step_time
        self.time_in_step = 0

        # Robot configuration
        self.left_foot = np.array([0.0, -0.1])
        self.right_foot = np.array([0.0, 0.1])
        self.com = np.array([0.0, 0.0])  # Center of mass

        # Walking state
        self.current_support = 'right'  # Start with right foot support
        self.swing_foot_trajectory = None

    def step(self, dt):
        """Execute one control step."""
        self.time_in_step += dt

        # Update swing foot trajectory
        if self.time_in_step < self.step_time:
            self.update_swing_foot(dt)
        else:
            # Step complete, switch support foot
            self.switch_support_foot()
            self.time_in_step = 0

        # Update CoM to maintain balance
        self.update_balance()

    def update_swing_foot(self, dt):
        """Update the trajectory of the swing foot."""
        progress = self.time_in_step / self.step_time

        if self.current_support == 'right':
            target = self.right_foot + np.array([self.step_length, 0])
            start = self.left_foot
        else:
            target = self.left_foot + np.array([self.step_length, 0])
            start = self.right_foot

        # Cubic interpolation for smooth trajectory
        t = progress
        swing_x = start[0] + (target[0] - start[0]) * t
        swing_y = start[1] + (target[1] - start[1]) * t

        # Add step height for foot clearance
        height_factor = np.sin(np.pi * t) * self.step_height
        if self.current_support == 'right':
            self.left_foot[0] = swing_x
            self.left_foot[1] = swing_y + height_factor
        else:
            self.right_foot[0] = swing_x
            self.right_foot[1] = swing_y + height_factor

    def switch_support_foot(self):
        """Switch the support foot at the end of a step."""
        if self.current_support == 'right':
            self.right_foot = self.left_foot + np.array([self.step_length, 0])
            self.current_support = 'left'
        else:
            self.left_foot = self.right_foot + np.array([self.step_length, 0])
            self.current_support = 'right'

    def update_balance(self):
        """Update CoM position to maintain balance."""
        # Simple balance control: keep CoM between feet
        support_x = self.right_foot[0] if self.current_support == 'right' else self.left_foot[0]

        # Proportional control to keep CoM over support polygon
        kp = 1.0
        error = support_x - self.com[0]
        self.com[0] += kp * error * 0.01  # Small adjustment
```

### 8.3.2 Zero Moment Point (ZMP) Control
```python
# zmp_controller.py
class ZMPController:
    def __init__(self, robot_mass=50.0, gravity=9.81):
        self.mass = robot_mass
        self.gravity = gravity
        self.com = np.array([0.0, 0.0, 0.8])  # CoM position
        self.com_vel = np.zeros(3)
        self.com_acc = np.zeros(3)

    def calculate_zmp(self):
        """Calculate Zero Moment Point."""
        x_com, y_com, z_com = self.com
        x_ddot, y_ddot, z_ddot = self.com_acc

        # ZMP = [x_com, y_com] - [z_com/g] * [x_ddot, y_ddot]
        zmp_x = x_com - (z_com / self.gravity) * x_ddot
        zmp_y = y_com - (z_com / self.gravity) * y_ddot

        return np.array([zmp_x, zmp_y])

    def is_stable(self, support_polygon):
        """Check if ZMP is within support polygon."""
        zmp = self.calculate_zmp()
        # Check if ZMP is inside the convex hull of support points
        return self.point_in_polygon(zmp, support_polygon)

    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using ray casting."""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
```

## 8.4 Gait Generation

### 8.4.1 Basic Gait Patterns
```python
# gait_patterns.py
class GaitGenerator:
    def __init__(self, robot_params):
        self.params = robot_params
        self.phase = 0.0
        self.gait_type = 'walk'

    def generate_walk_gait(self, time, amplitude=0.1, frequency=1.0):
        """Generate walking gait pattern."""
        phase = time * frequency * 2 * np.pi

        # Hip and knee trajectories for walking
        left_hip = amplitude * np.sin(phase)
        right_hip = amplitude * np.sin(phase + np.pi)  # Opposite phase
        left_knee = amplitude * 0.5 * np.sin(phase + np.pi/2)
        right_knee = amplitude * 0.5 * np.sin(phase + 3*np.pi/2)

        return {
            'left_hip': left_hip,
            'right_hip': right_hip,
            'left_knee': left_knee,
            'right_knee': right_knee
        }

    def generate_trot_gait(self, time, amplitude=0.1, frequency=1.0):
        """Generate trotting gait pattern."""
        phase = time * frequency * 2 * np.pi

        # Diagonal leg pairs move together
        # Left front and right back move together
        # Right front and left back move together
        left_hip = amplitude * np.sin(phase)
        right_hip = amplitude * np.sin(phase + np.pi)
        left_knee = amplitude * 0.5 * np.sin(phase + np.pi/2)
        right_knee = amplitude * 0.5 * np.sin(phase + 3*np.pi/2)

        return {
            'left_hip': left_hip,
            'right_hip': right_hip,
            'left_knee': left_knee,
            'right_knee': right_knee
        }
```

### 8.4.2 Foot Trajectory Planning
```python
# foot_trajectory.py
class FootTrajectoryGenerator:
    def __init__(self, step_length=0.3, step_height=0.05, step_time=0.8):
        self.step_length = step_length
        self.step_height = step_height
        self.step_time = step_time

    def generate_foot_trajectory(self, start_pos, end_pos, time_progress):
        """Generate smooth foot trajectory."""
        if time_progress > 1.0:
            time_progress = 1.0
        elif time_progress < 0.0:
            time_progress = 0.0

        # Position interpolation
        x = start_pos[0] + (end_pos[0] - start_pos[0]) * time_progress
        y = start_pos[1] + (end_pos[1] - start_pos[1]) * time_progress

        # Height profile (sinusoidal for smooth takeoff/landing)
        height = np.sin(np.pi * time_progress) * self.step_height

        return np.array([x, y, height])
```

## 8.5 Inverse Kinematics for Leg Control

### 8.5.1 Two-Link Leg Inverse Kinematics
```python
# inverse_kinematics.py
import math


class LegIKSolver:
    def __init__(self, thigh_length=0.3, shank_length=0.3):
        self.thigh_length = thigh_length
        self.shank_length = shank_length

    def solve_2d(self, target_x, target_y):
        """Solve inverse kinematics for 2D planar leg."""
        # Calculate distance to target
        dist_sq = target_x**2 + target_y**2
        dist = math.sqrt(dist_sq)

        # Check if target is reachable
        max_reach = self.thigh_length + self.shank_length
        min_reach = abs(self.thigh_length - self.shank_length)

        if dist > max_reach or dist < min_reach:
            raise ValueError("Target position is not reachable")

        # Calculate knee angle using law of cosines
        cos_knee = (self.thigh_length**2 + self.shank_length**2 - dist_sq) / \
                   (2 * self.thigh_length * self.shank_length)
        knee_angle = math.pi - math.acos(cos_knee)

        # Calculate hip angle
        cos_hip = (self.thigh_length**2 + dist_sq - self.shank_length**2) / \
                  (2 * self.thigh_length * dist)
        hip_angle = math.atan2(target_y, target_x) - math.acos(cos_hip)

        return hip_angle, knee_angle

    def solve_3d(self, target_pos, leg_offset=0.1):
        """Solve inverse kinematics for 3D leg with abduction."""
        target_x, target_y, target_z = target_pos

        # Calculate abduction angle
        abduction = math.atan2(target_y, target_z)

        # Project to 2D plane for hip/knee calculation
        yz_dist = math.sqrt(target_y**2 + target_z**2)
        projected_x = target_x
        projected_y = yz_dist - leg_offset  # Account for leg offset

        # Solve 2D IK
        hip_angle, knee_angle = self.solve_2d(projected_x, projected_y)

        return abduction, hip_angle, knee_angle


class LegController:
    def __init__(self, leg_id, ik_solver):
        self.leg_id = leg_id
        self.ik_solver = ik_solver
        self.current_angles = [0.0, 0.0, 0.0]  # abd, hip, knee

    def move_to_position(self, target_pos):
        """Move leg to target position using IK."""
        try:
            angles = self.ik_solver.solve_3d(target_pos)
            self.current_angles = angles
            return True
        except ValueError as e:
            print(f"Leg {self.leg_id}: {e}")
            return False

    def get_joint_commands(self):
        """Get joint angle commands for the leg."""
        return {
            f'leg_{self.leg_id}_abd': self.current_angles[0],
            f'leg_{self.leg_id}_hip': self.current_angles[1],
            f'leg_{self.leg_id}_knee': self.current_angles[2]
        }
```

## 8.6 Balance Control

### 8.6.1 Center of Mass Control
```python
# balance_control.py
class BalanceController:
    def __init__(self, robot_mass=50.0, com_height=0.8):
        self.mass = robot_mass
        self.com_height = com_height
        self.com_pos = np.array([0.0, 0.0, com_height])
        self.com_vel = np.zeros(3)
        self.com_acc = np.zeros(3)

        # PID controllers for balance
        self.x_controller = PIDController(kp=100.0, ki=10.0, kd=5.0)
        self.y_controller = PIDController(kp=100.0, ki=10.0, kd=5.0)

    def update_balance(self, target_com_pos, dt):
        """Update balance control to reach target CoM position."""
        # Calculate errors
        x_error = target_com_pos[0] - self.com_pos[0]
        y_error = target_com_pos[1] - self.com_pos[1]

        # Generate control forces
        x_force = self.x_controller.update(x_error, dt)
        y_force = self.y_controller.update(y_error, dt)

        # Apply forces to CoM
        self.com_acc[0] = x_force / self.mass
        self.com_acc[1] = y_force / self.mass

        # Update position and velocity
        self.com_vel[0] += self.com_acc[0] * dt
        self.com_vel[1] += self.com_acc[1] * dt
        self.com_pos[0] += self.com_vel[0] * dt
        self.com_pos[1] += self.com_vel[1] * dt

        return self.com_pos.copy()


class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        """Update PID controller."""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error

        return (self.kp * error +
                self.ki * self.integral +
                self.kd * derivative)
```

### 8.6.2 Capture Point Control
```python
# capture_point.py
class CapturePointController:
    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.tau = np.sqrt(com_height / gravity)  # Time constant

    def calculate_capture_point(self, com_pos, com_vel):
        """Calculate the capture point."""
        # Capture point = CoM position + CoM velocity * tau
        capture_point_x = com_pos[0] + com_vel[0] * self.tau
        capture_point_y = com_pos[1] + com_vel[1] * self.tau

        return np.array([capture_point_x, capture_point_y])

    def is_falling(self, com_pos, com_vel, support_polygon):
        """Check if robot is falling based on capture point."""
        cp = self.calculate_capture_point(com_pos, com_vel)

        # If capture point is outside support polygon, robot is falling
        return not self.point_in_polygon(cp, support_polygon)

    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon."""
        # Implementation from previous example
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
```

## 8.7 Walking Pattern Generators

### 8.7.1 Central Pattern Generator (CPG)
```python
# cpg_walking.py
class CentralPatternGenerator:
    def __init__(self, frequency=1.0):
        self.frequency = frequency
        self.phase = 0.0
        self.omega = 2 * np.pi * frequency

    def step(self, dt):
        """Update CPG phase."""
        self.phase += self.omega * dt
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi

    def get_leg_signal(self, leg_phase_offset):
        """Get activation signal for a specific leg."""
        phase = self.phase + leg_phase_offset
        return np.sin(phase)

    def setup_quadruped_gait(self):
        """Setup CPG for quadruped walking."""
        # Leg phase offsets for walking gait
        # LF, RF, LH, RH (left front, right front, left hind, right hind)
        self.leg_phases = {
            'LF': 0.0,           # Left Front
            'RF': np.pi,         # Right Front (opposite to LF)
            'LH': np.pi,         # Left Hind (same phase as RF)
            'RH': 0.0            # Right Hind (same phase as LF)
        }

    def get_leg_commands(self):
        """Get all leg commands based on current phase."""
        commands = {}
        for leg, phase_offset in self.leg_phases.items():
            commands[leg] = self.get_leg_signal(phase_offset)
        return commands
```

## 8.8 Implementation Example: Simple Walking Controller

```python
# walking_controller_example.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import numpy as np


class SimpleWalkingController(Node):
    def __init__(self):
        super().__init__('walking_controller')

        # Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.com_pub = self.create_publisher(Float32MultiArray, '/com_state', 10)

        # Subscribers
        self.cmd_sub = self.create_subscription(
            Twist, '/walking_cmd', self.cmd_callback, 10)

        # Walking parameters
        self.step_length = 0.3
        self.step_height = 0.05
        self.step_time = 0.8
        self.walk_speed = 0.0
        self.turn_rate = 0.0

        # Robot state
        self.time_in_step = 0.0
        self.swing_leg = 'left'
        self.support_leg = 'right'
        self.left_foot_pos = np.array([0.0, -0.1, 0.0])
        self.right_foot_pos = np.array([0.0, 0.1, 0.0])
        self.com_pos = np.array([0.0, 0.0, 0.8])

        # Timer for walking control
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100 Hz

        # IK solver
        self.ik_solver = LegIKSolver()

    def cmd_callback(self, msg):
        """Handle walking commands."""
        self.walk_speed = msg.linear.x
        self.turn_rate = msg.angular.z

    def control_loop(self):
        """Main walking control loop."""
        dt = 0.01  # 100 Hz

        # Update walking phase
        self.time_in_step += dt

        if self.time_in_step >= self.step_time:
            # Complete step, switch support
            if self.support_leg == 'left':
                self.support_leg = 'right'
                self.swing_leg = 'left'
                self.right_foot_pos[0] = self.left_foot_pos[0] + self.step_length
            else:
                self.support_leg = 'left'
                self.swing_leg = 'right'
                self.left_foot_pos[0] = self.right_foot_pos[0] + self.step_length

            self.time_in_step = 0.0

        # Calculate swing foot trajectory
        progress = self.time_in_step / self.step_time
        if self.swing_leg == 'left':
            target_x = self.right_foot_pos[0] + self.step_length
            self.left_foot_pos[0] = self.right_foot_pos[0] + progress * self.step_length
            self.left_foot_pos[1] = -0.1  # Swing foot side offset
            self.left_foot_pos[2] = np.sin(np.pi * progress) * self.step_height
        else:
            target_x = self.left_foot_pos[0] + self.step_length
            self.right_foot_pos[0] = self.left_foot_pos[0] + progress * self.step_length
            self.right_foot_pos[1] = 0.1  # Swing foot side offset
            self.right_foot_pos[2] = np.sin(np.pi * progress) * self.step_height

        # Update CoM to maintain balance
        self.update_balance()

        # Calculate joint angles using IK
        left_angles = self.calculate_leg_angles(self.left_foot_pos)
        right_angles = self.calculate_leg_angles(self.right_foot_pos)

        # Publish joint commands
        self.publish_joint_commands(left_angles, right_angles)

    def update_balance(self):
        """Update CoM position for balance."""
        # Simple balance: keep CoM between feet
        avg_foot_x = (self.left_foot_pos[0] + self.right_foot_pos[0]) / 2
        avg_foot_y = (self.left_foot_pos[1] + self.right_foot_pos[1]) / 2

        # Move CoM toward average foot position
        kp = 0.1
        self.com_pos[0] += kp * (avg_foot_x - self.com_pos[0])
        self.com_pos[1] += kp * (avg_foot_y - self.com_pos[1])

    def calculate_leg_angles(self, foot_pos):
        """Calculate joint angles for a leg using IK."""
        try:
            # Convert foot position to hip-relative coordinates
            hip_offset = 0.1  # Side offset of hip joint
            hip_to_foot = np.array([
                foot_pos[0] - self.com_pos[0],  # x
                foot_pos[1] - self.com_pos[1] - hip_offset,  # y
                foot_pos[2] - (self.com_pos[2] - 0.8)  # z (assuming robot height 0.8m)
            ])

            # Solve IK
            angles = self.ik_solver.solve_3d(hip_to_foot, leg_offset=hip_offset)
            return angles
        except ValueError:
            # If position is unreachable, return current angles
            return [0.0, 0.0, 0.0]

    def publish_joint_commands(self, left_angles, right_angles):
        """Publish joint commands."""
        msg = JointState()
        msg.name = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle'
        ]
        msg.position = [
            left_angles[1], left_angles[2], 0.0,  # Hip, knee, ankle for left
            right_angles[1], right_angles[2], 0.0  # Hip, knee, ankle for right
        ]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_pub.publish(msg)

        # Publish CoM state
        com_msg = Float32MultiArray()
        com_msg.data = self.com_pos.tolist()
        self.com_pub.publish(com_msg)


def main(args=None):
    rclpy.init(args=args)
    walking_controller = SimpleWalkingController()
    rclpy.spin(walking_controller)
    walking_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## 8.9 Advanced Locomotion Concepts

### 8.9.1 Model Predictive Control (MPC) for Walking
MPC can optimize walking trajectories by predicting future states and minimizing a cost function.

### 8.9.2 Adaptive Gait Control
Systems that adjust gait parameters based on terrain or robot state.

## 8.10 Best Practices

1. **Stability First**: Always prioritize balance over speed
2. **Smooth Transitions**: Ensure gait transitions are smooth
3. **Real-time Performance**: Optimize algorithms for real-time execution
4. **Robustness**: Handle sensor failures and unexpected disturbances
5. **Energy Efficiency**: Optimize for minimal energy consumption

## Practical Exercise

### Exercise 8.1: Implement a Walking Controller
**Objective**: Create a complete walking controller for a simulated humanoid robot

1. Implement inverse kinematics for a 6-DOF leg
2. Design a stable walking gait pattern
3. Implement balance control using ZMP or Capture Point
4. Create a complete walking controller node
5. Test the controller in simulation
6. Analyze stability margins and walking speed

**Deliverable**: Complete walking controller with stable gait, balance control, and performance analysis.

## Summary

Week 8 introduced locomotion algorithms for legged robots, including walking patterns, inverse kinematics, and balance control. You learned to generate stable gaits and maintain balance during dynamic locomotion. These algorithms are fundamental for humanoid robot mobility.

[Next: Week 9 - Manipulation & Grasping →](./week9-manipulation-grasping.md) | [Previous: Week 7 - Perception & Control Integration ←](./week7-perception-control-integration.md)