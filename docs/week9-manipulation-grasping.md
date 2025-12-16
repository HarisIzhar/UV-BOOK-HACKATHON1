---
sidebar_position: 10
---

# Week 9: Manipulation & Grasping

## Learning Objectives

By the end of this week, you will be able to:
- Understand robotic manipulation kinematics and dynamics
- Implement inverse kinematics for multi-DOF manipulator arms
- Design grasp planning algorithms for object manipulation
- Execute grasping and manipulation tasks using ROS
- Integrate perception with manipulation for object interaction

## 9.1 Introduction to Robotic Manipulation

Robotic manipulation involves the controlled movement and interaction of robot end-effectors with objects in the environment. This field encompasses:
- **Kinematics**: Geometric relationships of robot joints and links
- **Dynamics**: Forces and torques required for movement
- **Grasping**: Securely holding and manipulating objects
- **Planning**: Sequences of movements to achieve tasks

### 9.1.1 Manipulation Challenges
```
┌─────────────────────────────────────────────────────────────┐
│              Manipulation Challenges                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Grasp      │  │  Motion     │  │  Force      │        │
│  │  Planning   │  │  Planning   │  │  Control    │        │
│  │             │  │             │  │             │        │
│  │ • Object    │  │ • Collision │  │ • Contact   │        │
│  │   Geometry  │  │   Avoidance │  │   Stability │        │
│  │ • Friction  │  │ • Kinematic │  │ • Impedance │        │
│  │   Modeling  │  │   Planning  │  │   Control   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 9.2 Manipulator Kinematics

### 9.2.1 Forward Kinematics
Forward kinematics calculates the end-effector position from joint angles:

```python
# forward_kinematics.py
import numpy as np


def dh_transform(a, alpha, d, theta):
    """Denavit-Hartenberg transformation matrix."""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])


class ManipulatorKinematics:
    def __init__(self, dh_params):
        """
        Initialize manipulator with DH parameters.
        dh_params: list of tuples (a, alpha, d, theta) for each joint
        """
        self.dh_params = dh_params
        self.num_joints = len(dh_params)

    def forward_kinematics(self, joint_angles):
        """Calculate end-effector pose from joint angles."""
        if len(joint_angles) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint angles, got {len(joint_angles)}")

        # Start with identity matrix
        T = np.eye(4)

        for i, angle in enumerate(joint_angles):
            a, alpha, d, _ = self.dh_params[i]
            # Update theta with joint angle
            T_joint = dh_transform(a, alpha, d, angle)
            T = np.dot(T, T_joint)

        # Extract position and orientation
        position = T[:3, 3]
        orientation = T[:3, :3]

        return position, orientation


# Example: 3-DOF planar manipulator
dh_params_3dof = [
    (0.1, 0, 0, 0),      # Joint 1: a=0.1, alpha=0, d=0, theta is variable
    (0.1, 0, 0, 0),      # Joint 2: a=0.1, alpha=0, d=0, theta is variable
    (0.1, 0, 0, 0)       # Joint 3: a=0.1, alpha=0, d=0, theta is variable
]

manipulator = ManipulatorKinematics(dh_params_3dof)
joint_angles = [np.pi/4, np.pi/6, -np.pi/3]
pos, orient = manipulator.forward_kinematics(joint_angles)
print(f"End-effector position: {pos}")
print(f"End-effector orientation:\n{orient}")
```

### 9.2.2 Jacobian Matrix
The Jacobian relates joint velocities to end-effector velocities:

```python
# jacobian.py
class ManipulatorJacobian:
    def __init__(self, manipulator):
        self.manipulator = manipulator

    def calculate_jacobian(self, joint_angles):
        """Calculate the geometric Jacobian matrix."""
        num_joints = len(joint_angles)
        jacobian = np.zeros((6, num_joints))  # [linear; angular]

        # Get all transformation matrices
        T_total = np.eye(4)
        T_list = [T_total.copy()]

        for i, angle in enumerate(joint_angles):
            a, alpha, d, _ = self.manipulator.dh_params[i]
            T_joint = dh_transform(a, alpha, d, angle)
            T_total = np.dot(T_total, T_joint)
            T_list.append(T_total.copy())

        # End-effector position
        end_pos = T_list[-1][:3, 3]

        # Calculate Jacobian columns
        for i in range(num_joints):
            # Z-axis of joint i in base frame
            z_i = T_list[i][:3, 2]
            # Position of joint i in base frame
            p_i = T_list[i][:3, 3]

            # Linear velocity component
            jacobian[:3, i] = np.cross(z_i, end_pos - p_i)
            # Angular velocity component
            jacobian[3:, i] = z_i

        return jacobian

    def inverse_jacobian(self, joint_angles, method='svd'):
        """Calculate inverse of Jacobian."""
        J = self.calculate_jacobian(joint_angles)

        if method == 'svd':
            # Use SVD to handle singularities
            U, s, Vt = np.linalg.svd(J)
            # Create pseudo-inverse
            s_inv = np.where(s > 1e-6, 1./s, 0)
            J_inv = np.dot(Vt.T, np.dot(np.diag(s_inv), U.T))
        else:
            # Use Moore-Penrose pseudo-inverse
            J_inv = np.linalg.pinv(J)

        return J_inv
```

## 9.3 Inverse Kinematics

### 9.3.1 Analytical Inverse Kinematics
For simple manipulators, analytical solutions may exist:

```python
# analytical_ik.py
class AnalyticalIK:
    def __init__(self, link_lengths):
        self.l1, self.l2 = link_lengths

    def solve_2dof_planar(self, x, y):
        """Solve inverse kinematics for 2-DOF planar manipulator."""
        # Distance from origin to target
        r = np.sqrt(x**2 + y**2)

        # Check if target is reachable
        if r > self.l1 + self.l2:
            raise ValueError("Target is outside workspace")
        if r < abs(self.l1 - self.l2):
            raise ValueError("Target is inside workspace but unreachable")

        # Calculate joint angles
        cos_theta2 = (x**2 + y**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        theta2 = np.arccos(cos_theta2)

        k1 = self.l1 + self.l2 * np.cos(theta2)
        k2 = self.l2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        return theta1, theta2

    def solve_3dof_spherical(self, pos, orientation=None):
        """Solve for 3-DOF spherical wrist manipulator."""
        x, y, z = pos

        # First joint (waist) - orient towards projection on ground plane
        theta1 = np.arctan2(y, x)

        # Distance from base to target (projected to arm plane)
        r = np.sqrt(x**2 + y**2)
        d = z  # Height

        # This is a simplified example - full solution requires more joints
        return [theta1, 0.0, 0.0]
```

### 9.3.2 Numerical Inverse Kinematics
For complex manipulators, numerical methods are often used:

```python
# numerical_ik.py
from scipy.optimize import minimize


class NumericalIK:
    def __init__(self, manipulator, max_iterations=100, tolerance=1e-6):
        self.manipulator = manipulator
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(self, target_pose, initial_guess=None):
        """Solve inverse kinematics using numerical optimization."""
        if initial_guess is None:
            initial_guess = np.zeros(self.manipulator.num_joints)

        def objective_function(joint_angles):
            """Minimize distance to target pose."""
            current_pos, current_orient = self.manipulator.forward_kinematics(joint_angles)
            target_pos, target_orient = target_pose

            # Position error
            pos_error = np.linalg.norm(current_pos - target_pos)

            # Orientation error (simplified - use quaternion distance in practice)
            orient_error = np.linalg.norm(current_orient - target_orient)

            return pos_error + 0.1 * orient_error  # Weight position more heavily

        # Optimize joint angles
        result = minimize(
            objective_function,
            initial_guess,
            method='BFGS',
            options={'maxiter': self.max_iterations}
        )

        if result.success and objective_function(result.x) < self.tolerance:
            return result.x
        else:
            raise ValueError("IK solution not found within tolerance")


class JacobianIK:
    def __init__(self, manipulator, step_size=0.01, max_iterations=1000):
        self.manipulator = manipulator
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.jacobian_calc = ManipulatorJacobian(manipulator)

    def solve(self, target_pos, target_orient=None, initial_joints=None):
        """Solve IK using Jacobian transpose method."""
        if initial_joints is None:
            current_joints = np.zeros(self.manipulator.num_joints)
        else:
            current_joints = initial_joints.copy()

        for i in range(self.max_iterations):
            # Get current end-effector pose
            current_pos, current_orient = self.manipulator.forward_kinematics(current_joints)

            # Calculate error
            pos_error = target_pos - current_pos
            if np.linalg.norm(pos_error) < 0.001:  # 1mm tolerance
                break

            # Calculate Jacobian
            J = self.jacobian_calc.calculate_jacobian(current_joints)

            # Use pseudo-inverse to find joint adjustments
            J_inv = np.linalg.pinv(J[:3, :])  # Only position part
            joint_delta = np.dot(J_inv, pos_error)

            # Update joint angles
            current_joints += self.step_size * joint_delta

        return current_joints
```

## 9.4 Grasp Planning

### 9.4.1 Grasp Types and Strategies
```python
# grasp_types.py
class GraspTypes:
    """Different types of grasps for robotic manipulation."""

    # Power grasps - for heavy objects
    POWER_GRASP = "power"
    # Precision grasps - for delicate objects
    PRECISION_GRASP = "precision"
    # Pinch grasps - for small objects
    PINCH_GRASP = "pinch"

    @staticmethod
    def get_grasp_for_object(object_properties):
        """Select appropriate grasp type based on object properties."""
        weight = object_properties.get('weight', 1.0)
        size = object_properties.get('size', 1.0)
        fragility = object_properties.get('fragility', 0.5)  # 0-1 scale

        if weight > 2.0:  # Heavy object
            return GraspTypes.POWER_GRASP
        elif fragility > 0.7:  # Fragile object
            return GraspTypes.PRECISION_GRASP
        elif size < 0.05:  # Small object
            return GraspTypes.PINCH_GRASP
        else:
            return GraspTypes.POWER_GRASP


class GraspPlanner:
    def __init__(self, robot_hand):
        self.robot_hand = robot_hand
        self.contact_points = []
        self.grasp_configurations = []

    def plan_grasp_points(self, object_mesh):
        """Plan grasp points based on object geometry."""
        # This is a simplified approach
        # In practice, use more sophisticated algorithms

        # Find stable grasp points on object surface
        grasp_points = self.find_stable_grasps(object_mesh)
        return grasp_points

    def find_stable_grasps(self, object_mesh):
        """Find stable grasp points using geometric analysis."""
        # Simplified approach - in practice, use:
        # - Convex hull analysis
        # - Force closure analysis
        # - Friction cone analysis

        # Sample points on object surface
        surface_points = self.sample_surface(object_mesh)

        stable_grasps = []
        for point in surface_points:
            if self.is_stable_grasp(point, object_mesh):
                stable_grasps.append(point)

        return stable_grasps

    def sample_surface(self, object_mesh):
        """Sample points on object surface."""
        # Simplified sampling
        # In practice, use mesh sampling algorithms
        return [np.random.random(3) for _ in range(20)]

    def is_stable_grasp(self, point, object_mesh):
        """Check if grasp at point would be stable."""
        # Simplified stability check
        # In practice, use force closure analysis
        return True
```

### 9.4.2 Force Closure Analysis
```python
# force_closure.py
class ForceClosureAnalyzer:
    def __init__(self, friction_coefficient=0.8):
        self.mu = friction_coefficient

    def analyze_2d_grasp(self, contact_points, contact_normals):
        """Analyze 2D grasp for force closure."""
        # For 2D: need at least 3 contacts for force closure
        if len(contact_points) < 3:
            return False

        # Check if contact points can generate any wrench
        # This is a simplified check - full analysis is more complex
        return self.check_wrench_space(contact_points, contact_normals)

    def check_wrench_space(self, contact_points, contact_normals):
        """Check if contacts can resist any external wrench."""
        # Simplified wrench space analysis
        # In practice, use convex hull methods
        return True  # Placeholder

    def calculate_grasp_quality(self, contact_points, contact_normals, object_com):
        """Calculate grasp quality metric."""
        # Distance from center of mass
        distances = [np.linalg.norm(cp - object_com) for cp in contact_points]

        # Average distance (stability indicator)
        avg_distance = np.mean(distances)

        # Number of contacts (more contacts = better)
        num_contacts = len(contact_points)

        # Quality score
        quality = num_contacts * avg_distance
        return quality
```

## 9.5 Grasp Execution

### 9.5.1 Grasp Control Strategies
```python
# grasp_control.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Pose, WrenchStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np


class GraspController(Node):
    def __init__(self):
        super().__init__('grasp_controller')

        # Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.gripper_pub = self.create_publisher(JointState, '/gripper_commands', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.wrench_sub = self.create_subscription(
            WrenchStamped, '/wrench', self.wrench_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10)

        # Internal state
        self.current_joints = {}
        self.wrench_data = None
        self.bridge = CvBridge()

        # Grasp parameters
        self.gripper_force_limit = 50.0  # Newtons
        self.approach_velocity = 0.05  # m/s
        self.grasp_velocity = 0.01  # m/s

    def joint_state_callback(self, msg):
        """Update current joint states."""
        for i, name in enumerate(msg.name):
            self.current_joints[name] = msg.position[i]

    def wrench_callback(self, msg):
        """Update force/torque sensor data."""
        self.wrench_data = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])

    def image_callback(self, msg):
        """Process camera image for object detection."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # Object detection and pose estimation would happen here

    def approach_object(self, target_pose):
        """Move end-effector to approach position near object."""
        # Plan trajectory to approach position
        # This should be 5-10cm from the object
        approach_pos = self.calculate_approach_position(target_pose)

        # Execute trajectory
        success = self.move_to_pose(approach_pos)
        return success

    def calculate_approach_position(self, object_pose):
        """Calculate approach position based on object pose."""
        # Move 10cm away from object in the approach direction
        approach_distance = 0.1  # 10cm

        # For now, approach from the front (simplified)
        approach_pos = np.array([
            object_pose.position.x - approach_distance,
            object_pose.position.y,
            object_pose.position.z
        ])

        return approach_pos

    def execute_grasp(self, grasp_pose, grasp_type='power'):
        """Execute the grasp at the specified pose."""
        # 1. Move to pre-grasp position
        if not self.approach_object(grasp_pose):
            return False

        # 2. Move to grasp position
        if not self.move_to_pose(grasp_pose):
            return False

        # 3. Close gripper
        if not self.close_gripper(grasp_type):
            # If grasp fails, retreat
            self.retreat_from_object()
            return False

        # 4. Lift object slightly
        self.lift_object()

        return True

    def close_gripper(self, grasp_type):
        """Close gripper with appropriate force for grasp type."""
        cmd = JointState()
        cmd.name = ['gripper_joint']

        if grasp_type == 'power':
            # Close firmly but not too tight
            cmd.position = [0.02]  # Adjust based on gripper design
            cmd.effort = [self.gripper_force_limit]
        elif grasp_type == 'precision':
            # Close gently
            cmd.position = [0.01]
            cmd.effort = [self.gripper_force_limit * 0.5]
        else:
            # Default grasp
            cmd.position = [0.015]
            cmd.effort = [self.gripper_force_limit * 0.7]

        cmd.header.stamp = self.get_clock().now().to_msg()
        self.gripper_pub.publish(cmd)

        # Wait for gripper to close
        self.get_logger().info(f"Closing gripper for {grasp_type} grasp")
        return True

    def lift_object(self):
        """Lift the object after successful grasp."""
        # Move up by 5cm to clear the surface
        current_pose = self.get_current_pose()
        lift_pose = current_pose.copy()
        lift_pose[2] += 0.05  # Lift 5cm

        self.move_to_pose(lift_pose)
        self.get_logger().info("Object lifted after grasp")

    def retreat_from_object(self):
        """Retreat from object after failed grasp."""
        current_pose = self.get_current_pose()
        retreat_pose = current_pose.copy()
        retreat_pose[0] -= 0.05  # Move back 5cm

        self.move_to_pose(retreat_pose)
        self.get_logger().info("Retreated after failed grasp")

    def move_to_pose(self, target_pose):
        """Move end-effector to target pose (simplified)."""
        # In practice, this would use MoveIt or similar
        # For now, we'll just log the movement
        self.get_logger().info(f"Moving to pose: {target_pose}")
        return True

    def get_current_pose(self):
        """Get current end-effector pose (simplified)."""
        # In practice, get from forward kinematics
        return np.array([0.0, 0.0, 0.5])  # Placeholder
```

### 9.5.2 Adaptive Grasp Control
```python
# adaptive_grasp.py
class AdaptiveGraspController:
    def __init__(self, grasp_controller):
        self.grasp_controller = grasp_controller
        self.force_threshold = 10.0  # N
        self.slip_threshold = 0.1   # arbitrary units

    def adaptive_grasp(self, object_properties, target_pose):
        """Execute adaptive grasp based on object properties."""
        # Determine appropriate grasp type
        grasp_type = GraspTypes.get_grasp_for_object(object_properties)

        # Execute initial grasp
        success = self.grasp_controller.execute_grasp(target_pose, grasp_type)

        if success:
            # Test grasp stability
            stability = self.test_grasp_stability()
            if stability < 0.8:  # Not stable enough
                self.adjust_grasp_strength()

        return success

    def test_grasp_stability(self):
        """Test if grasp is stable."""
        # Monitor force/torque sensors
        if self.grasp_controller.wrench_data is not None:
            forces = self.grasp_controller.wrench_data[:3]
            # Simple stability metric
            stability = 1.0 - min(np.abs(forces)) / self.force_threshold
            return max(0.0, min(1.0, stability))

        return 0.5  # Unknown stability

    def adjust_grasp_strength(self):
        """Adjust grasp strength based on stability feedback."""
        # Increase grip force if unstable, decrease if too tight
        pass
```

## 9.6 Manipulation Planning

### 9.6.1 Trajectory Planning for Manipulation
```python
# manipulation_planning.py
class ManipulationPlanner:
    def __init__(self, manipulator):
        self.manipulator = manipulator

    def plan_reach_trajectory(self, start_pose, end_pose, num_waypoints=20):
        """Plan trajectory for reaching a target."""
        # Interpolate between start and end poses
        trajectory = []

        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            # Linear interpolation in Cartesian space
            current_pos = (1 - t) * start_pose[:3] + t * end_pose[:3]
            current_orient = self.slerp(start_pose[3:], end_pose[3:], t)

            # Solve IK for each waypoint
            try:
                joint_angles = self.solve_ik_with_avoidance(current_pos, current_orient)
                trajectory.append(joint_angles)
            except ValueError:
                # If IK fails, try with collision avoidance
                joint_angles = self.solve_ik_with_avoidance(current_pos, current_orient)
                trajectory.append(joint_angles)

        return trajectory

    def solve_ik_with_avoidance(self, pos, orient):
        """Solve IK with collision avoidance."""
        # Use numerical IK with collision constraints
        # This is a simplified version
        ik_solver = NumericalIK(self.manipulator)
        target_pose = (pos, orient)
        initial_guess = np.zeros(self.manipulator.num_joints)

        return ik_solver.solve(target_pose, initial_guess)

    def slerp(self, quat1, quat2, t):
        """Spherical linear interpolation for orientations."""
        # Simplified - in practice use proper quaternion SLERP
        return (1 - t) * quat1 + t * quat2
```

### 9.6.2 Collision-Avoidant Path Planning
```python
# collision_avoidance.py
class CollisionAvoidancePlanner:
    def __init__(self, robot_model, environment):
        self.robot = robot_model
        self.environment = environment

    def plan_safe_path(self, start_config, goal_config):
        """Plan collision-free path using RRT or similar algorithm."""
        # Simplified collision checking
        path = [start_config]

        # In practice, use RRT, PRM, or other sampling-based planners
        current_config = start_config.copy()

        # Simple straight-line approach with collision checking
        for _ in range(50):  # Max iterations
            # Move toward goal
            direction = goal_config - current_config
            step = 0.1 * direction / np.linalg.norm(direction)
            new_config = current_config + step

            if self.is_collision_free(new_config):
                path.append(new_config)
                current_config = new_config

                if np.linalg.norm(current_config - goal_config) < 0.01:
                    break
            else:
                # Try random direction to escape local minima
                random_step = 0.05 * np.random.random(len(current_config))
                new_config = current_config + random_step

                if self.is_collision_free(new_config):
                    path.append(new_config)
                    current_config = new_config

        return path

    def is_collision_free(self, config):
        """Check if robot configuration is collision-free."""
        # Calculate robot link positions using forward kinematics
        link_positions = self.calculate_link_positions(config)

        # Check collision with environment
        for pos in link_positions:
            if self.is_in_collision(pos):
                return False
        return True

    def calculate_link_positions(self, config):
        """Calculate positions of all robot links."""
        # Use forward kinematics to get link positions
        positions = []
        T = np.eye(4)

        for i, angle in enumerate(config):
            if i < len(self.robot.dh_params):
                a, alpha, d, _ = self.robot.dh_params[i]
                T_joint = dh_transform(a, alpha, d, angle)
                T = np.dot(T, T_joint)
                positions.append(T[:3, 3])

        return positions

    def is_in_collision(self, position):
        """Check if position is in collision with environment."""
        # Simplified collision detection
        # In practice, use proper collision detection libraries
        return False
```

## 9.7 Perception-Action Integration

### 9.7.1 Visual Servoing for Grasping
```python
# visual_servoing_grasp.py
class VisualServoingGrasp:
    def __init__(self, camera_matrix, image_size):
        self.camera_matrix = camera_matrix
        self.image_width, self.image_height = image_size
        self.pixel_error_threshold = 5  # pixels

    def align_to_object(self, object_pose_2d, target_pixel):
        """Align end-effector with object using visual feedback."""
        # Calculate pixel error
        pixel_error = target_pixel - object_pose_2d

        if np.linalg.norm(pixel_error) < self.pixel_error_threshold:
            return True  # Aligned

        # Convert pixel error to Cartesian velocity
        velocity_cmd = self.pixel_to_cartesian_velocity(pixel_error)

        # Execute velocity command
        self.execute_cartesian_velocity(velocity_cmd)

        return False  # Not yet aligned

    def pixel_to_cartesian_velocity(self, pixel_error):
        """Convert pixel error to Cartesian velocity."""
        # Use interaction matrix (Jacobian in image space)
        # Simplified approach
        K = 0.001  # Gain
        velocity = K * pixel_error
        return velocity

    def execute_cartesian_velocity(self, velocity):
        """Execute Cartesian velocity command."""
        # Convert Cartesian velocity to joint velocities using Jacobian
        # In practice, integrate with robot controller
        pass
```

### 9.7.2 Multi-Modal Sensor Fusion for Grasping
```python
# sensor_fusion_grasp.py
class MultiModalGraspController:
    def __init__(self):
        self.vision_data = None
        self.force_data = None
        self.tactile_data = None
        self.fusion_weights = {
            'vision': 0.6,
            'force': 0.3,
            'tactile': 0.1
        }

    def integrate_sensors(self):
        """Integrate data from multiple sensors for grasping."""
        grasp_quality = 0.0
        total_weight = 0.0

        if self.vision_data:
            vision_quality = self.evaluate_vision_grasp()
            grasp_quality += self.fusion_weights['vision'] * vision_quality
            total_weight += self.fusion_weights['vision']

        if self.force_data:
            force_quality = self.evaluate_force_grasp()
            grasp_quality += self.fusion_weights['force'] * force_quality
            total_weight += self.fusion_weights['force']

        if self.tactile_data:
            tactile_quality = self.evaluate_tactile_grasp()
            grasp_quality += self.fusion_weights['tactile'] * tactile_quality
            total_weight += self.fusion_weights['tactile']

        if total_weight > 0:
            return grasp_quality / total_weight
        else:
            return 0.5  # Default quality if no sensors available

    def evaluate_vision_grasp(self):
        """Evaluate grasp quality based on vision data."""
        # Analyze object shape, size, orientation
        return 0.8  # Placeholder

    def evaluate_force_grasp(self):
        """Evaluate grasp quality based on force data."""
        # Analyze grasp force and stability
        return 0.7  # Placeholder

    def evaluate_tactile_grasp(self):
        """Evaluate grasp quality based on tactile data."""
        # Analyze contact stability
        return 0.9  # Placeholder
```

## 9.8 Implementation Example: Complete Manipulation Node

```python
# manipulation_example.py
class ManipulationNode(Node):
    def __init__(self):
        super().__init__('manipulation_node')

        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.result_pub = self.create_publisher(String, '/manipulation_result', 10)

        self.object_sub = self.create_subscription(
            Pose, '/object_pose', self.object_callback, 10)

        # Components
        self.ik_solver = NumericalIK(None)  # Will be configured with actual manipulator
        self.grasp_planner = GraspPlanner(None)
        self.motion_planner = ManipulationPlanner(None)
        self.grasp_controller = GraspController()

        # State
        self.target_object = None
        self.is_executing = False

    def object_callback(self, msg):
        """Handle detected object pose."""
        if not self.is_executing:
            self.target_object = msg
            self.execute_manipulation_task()

    def execute_manipulation_task(self):
        """Execute complete manipulation task."""
        if self.target_object is None:
            return

        self.is_executing = True
        success = False

        try:
            # 1. Plan approach trajectory
            approach_pose = self.calculate_approach_pose(self.target_object)
            approach_traj = self.motion_planner.plan_reach_trajectory(
                self.get_current_pose(), approach_pose)

            # 2. Execute approach
            for waypoint in approach_traj:
                if not self.move_to_joint_config(waypoint):
                    raise Exception("Failed to reach approach position")

            # 3. Plan grasp
            grasp_pose = self.calculate_grasp_pose(self.target_object)
            grasp_type = GraspTypes.get_grasp_for_object({'size': 0.1, 'weight': 0.5})

            # 4. Execute grasp
            success = self.grasp_controller.execute_grasp(grasp_pose, grasp_type)

            # 5. Handle result
            if success:
                result_msg = String()
                result_msg.data = "Grasp successful"
                self.result_pub.publish(result_msg)
            else:
                result_msg = String()
                result_msg.data = "Grasp failed"
                self.result_pub.publish(result_msg)

        except Exception as e:
            self.get_logger().error(f"Manipulation task failed: {e}")
            success = False
        finally:
            self.is_executing = False

    def calculate_approach_pose(self, object_pose):
        """Calculate approach pose in front of object."""
        # Implementation depends on robot kinematics
        pass

    def calculate_grasp_pose(self, object_pose):
        """Calculate optimal grasp pose for object."""
        # Implementation depends on object properties
        pass

    def move_to_joint_config(self, joint_config):
        """Move robot to specified joint configuration."""
        # Implementation depends on robot controller
        cmd = JointState()
        cmd.position = joint_config
        cmd.header.stamp = self.get_clock().now().to_msg()
        self.cmd_pub.publish(cmd)
        return True

    def get_current_pose(self):
        """Get current end-effector pose."""
        # Implementation depends on robot state
        return np.zeros(6)


def main(args=None):
    rclpy.init(args=args)
    manipulation_node = ManipulationNode()
    rclpy.spin(manipulation_node)
    manipulation_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## 9.9 Advanced Manipulation Concepts

### 9.9.1 Dexterous Manipulation
Advanced manipulation involving multiple contact points and complex object interactions.

### 9.9.2 Learning-Based Grasping
Using machine learning to improve grasp success rates based on experience.

## 9.10 Best Practices

1. **Safety First**: Always implement force limits and collision detection
2. **Gradual Approach**: Approach objects slowly and carefully
3. **Sensor Feedback**: Use multiple sensors for robust grasping
4. **Adaptability**: Adjust grasp parameters based on object properties
5. **Recovery**: Implement recovery strategies for failed grasps

## Practical Exercise

### Exercise 9.1: Robotic Grasping System
**Objective**: Create a complete grasping system that integrates perception and manipulation

1. Implement inverse kinematics for a 6-DOF manipulator arm
2. Design a grasp planning algorithm for various object shapes
3. Create a grasp execution controller with force feedback
4. Integrate perception data for object pose estimation
5. Test the system with simulated objects
6. Analyze grasp success rates and stability

**Deliverable**: Complete grasping system with perception, planning, and execution components.

## Summary

Week 9 covered robotic manipulation and grasping, including kinematics, grasp planning, and execution strategies. You learned to solve inverse kinematics problems, plan stable grasps, and integrate perception with manipulation. These skills are essential for robots that need to interact with objects in their environment.

[Next: Week 10 - Locomotion & Manipulation Integration →](./week10-locomotion-manipulation-integration.md) | [Previous: Week 8 - Locomotion Algorithms ←](./week8-locomotion-algorithms.md)