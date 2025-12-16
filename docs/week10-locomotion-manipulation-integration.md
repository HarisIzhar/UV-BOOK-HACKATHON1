---
sidebar_position: 11
---

# Week 10: Locomotion & Manipulation Integration

## Learning Objectives

By the end of this week, you will be able to:
- Coordinate locomotion and manipulation in humanoid robots
- Implement whole-body control for combined tasks
- Design dynamic balancing during manipulation
- Create coordinated multi-limb behaviors
- Handle dynamic interactions between locomotion and manipulation

## 10.1 Introduction to Locomotion-Manipulation Integration

Humanoid robots must coordinate their locomotion and manipulation capabilities to perform complex tasks. This integration presents unique challenges:

- **Dynamic Balance**: Manipulation forces affect robot stability
- **Multi-Task Coordination**: Simultaneous walking and manipulation
- **Resource Allocation**: Sharing computational and actuator resources
- **Task Prioritization**: Balancing mobility vs. manipulation goals

### 10.1.1 Integration Architecture
```
┌─────────────────────────────────────────────────────────────┐
│        Locomotion-Manipulation Integration                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐              ┌─────────────┐              │
│  │  Locomotion │ ◄──────────► │  Whole-Body │ ◄──────────► │
│  │  Control    │              │  Controller │              │
│  │             │              │             │              │
│  │ • Gait      │              │ • Task      │              │
│  │   Planning  │              │   Priorities│              │
│  │ • Balance   │              │ • Redundancy│              │
│  │   Control   │              │ • Dynamics  │              │
│  └─────────────┘              │ • Optimization│            │
│                               └─────────────┘              │
│  ┌─────────────┐              ┌─────────────┐              │
│  │ Manipulation│ ◄──────────► │ Perception  │              │
│  │  Control    │              │  Fusion     │              │
│  │             │              │             │              │
│  │ • Grasp     │              │ • State     │              │
│  │   Planning  │              │   Estimation│              │
│  │ • Trajectory│              │ • Multi-    │              │
│  │   Control   │              │   Sensor    │              │
│  └─────────────┘              │   Fusion    │              │
│                               └─────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## 10.2 Whole-Body Control Framework

### 10.2.1 Task-Priority Based Control
Whole-body control manages multiple tasks with different priorities:

```python
# whole_body_control.py
import numpy as np
from scipy.optimize import minimize


class WholeBodyController:
    def __init__(self, robot_model):
        self.model = robot_model
        self.tasks = []  # List of tasks with priorities
        self.joint_limits = robot_model.joint_limits

    def add_task(self, task, priority, weight=1.0):
        """Add a task to the control hierarchy."""
        self.tasks.append({
            'task': task,
            'priority': priority,  # 0 = highest priority
            'weight': weight,
            'active': True
        })

    def solve_control(self, current_state):
        """Solve whole-body control using prioritized optimization."""
        # Sort tasks by priority
        sorted_tasks = sorted(self.tasks, key=lambda x: x['priority'])

        # Solve tasks in order of priority
        joint_velocities = np.zeros(self.model.num_joints)

        for task_info in sorted_tasks:
            if not task_info['active']:
                continue

            task = task_info['task']
            priority = task_info['priority']
            weight = task_info['weight']

            # Calculate desired velocity for this task
            task_vel = task.calculate_velocity(current_state)

            # Project onto null space of higher priority tasks
            if priority > 0:
                # Calculate null space projection
                null_space = self.calculate_null_space(joint_velocities)
                task_vel = np.dot(null_space, task_vel)

            # Apply with weight
            joint_velocities += weight * task_vel

        return joint_velocities

    def calculate_null_space(self, primary_solution):
        """Calculate null space projection matrix."""
        # For a solution x₀, null space is I - J⁺J where J⁺ is pseudo-inverse
        # This is a simplified approach
        J = self.model.jacobian  # Robot Jacobian
        J_pinv = np.linalg.pinv(J)
        I = np.eye(self.model.num_joints)
        null_space = I - np.dot(J_pinv, J)
        return null_space


class Task:
    """Base class for control tasks."""
    def __init__(self, name, desired_value, gain=1.0):
        self.name = name
        self.desired_value = desired_value
        self.gain = gain

    def calculate_velocity(self, current_state):
        """Calculate desired velocity to achieve task."""
        current_value = self.get_current_value(current_state)
        error = self.desired_value - current_value
        return self.gain * error

    def get_current_value(self, current_state):
        """Get current value of task variable."""
        raise NotImplementedError


class PositionTask(Task):
    """Task to control end-effector position."""
    def __init__(self, name, desired_position, gain=1.0):
        super().__init__(name, desired_position, gain)

    def get_current_value(self, current_state):
        """Get current end-effector position."""
        # Use forward kinematics
        pos, _ = self.model.forward_kinematics(current_state)
        return pos


class BalanceTask(Task):
    """Task to maintain balance."""
    def __init__(self, name, com_reference, gain=1.0):
        super().__init__(name, com_reference, gain)

    def get_current_value(self, current_state):
        """Get current center of mass position."""
        # Calculate CoM from current joint configuration
        return self.model.calculate_com(current_state)
```

### 10.2.2 Hierarchical Control Structure
```python
# hierarchical_control.py
class HierarchicalController:
    def __init__(self):
        # High-level: Task planning
        self.task_planner = TaskPlanner()

        # Mid-level: Whole-body control
        self.whole_body_controller = WholeBodyController(None)

        # Low-level: Joint control
        self.joint_controller = JointController()

    def execute_task(self, task_description):
        """Execute a high-level task through the hierarchy."""
        # Plan the task
        task_plan = self.task_planner.plan(task_description)

        # Execute with whole-body coordination
        for subtask in task_plan:
            self.execute_subtask(subtask)

    def execute_subtask(self, subtask):
        """Execute a subtask with proper coordination."""
        # Determine required coordination between locomotion and manipulation
        if subtask.requires_locomotion and subtask.requires_manipulation:
            # Coordinate both simultaneously
            self.coordinate_locomanipulation(subtask)
        elif subtask.requires_locomotion:
            # Pure locomotion task
            self.execute_locomotion(subtask)
        elif subtask.requires_manipulation:
            # Pure manipulation task
            self.execute_manipulation(subtask)

    def coordinate_locomanipulation(self, task):
        """Coordinate locomotion and manipulation for complex tasks."""
        # Add both locomotion and manipulation tasks to whole-body controller
        self.whole_body_controller.add_task(
            task.locomotion_task, priority=1, weight=0.8)
        self.whole_body_controller.add_task(
            task.manipulation_task, priority=2, weight=0.9)

        # Add balance task as highest priority
        balance_task = BalanceTask("balance", task.com_reference)
        self.whole_body_controller.add_task(balance_task, priority=0, weight=1.0)

        # Execute coordinated control
        current_state = self.get_robot_state()
        joint_velocities = self.whole_body_controller.solve_control(current_state)
        self.joint_controller.execute_velocities(joint_velocities)
```

## 10.3 Dynamic Balance During Manipulation

### 10.3.1 Center of Mass Management
```python
# balance_management.py
class BalanceManager:
    def __init__(self, robot_mass=50.0, com_height=0.8):
        self.robot_mass = robot_mass
        self.com_height = com_height
        self.gravity = 9.81
        self.tau = np.sqrt(com_height / self.gravity)  # Time constant for inverted pendulum

        # Current state
        self.com_pos = np.array([0.0, 0.0, com_height])
        self.com_vel = np.zeros(3)
        self.com_acc = np.zeros(3)

        # Support polygon (area where CoM must stay to maintain balance)
        self.support_polygon = self.calculate_support_polygon()

    def update_balance(self, manipulation_forces, dt):
        """Update balance considering manipulation forces."""
        # Calculate effect of manipulation forces on CoM
        external_forces = self.calculate_external_forces(manipulation_forces)

        # Update CoM dynamics
        self.com_acc = external_forces / self.robot_mass
        self.com_vel += self.com_acc * dt
        self.com_pos += self.com_vel * dt

        # Check if CoM is still in support polygon
        if not self.is_balanced():
            # Generate corrective action
            corrective_action = self.generate_balance_correction()
            return corrective_action

        return None  # No correction needed

    def calculate_external_forces(self, manipulation_forces):
        """Calculate external forces acting on robot."""
        # Sum of all forces: gravity + manipulation forces + other forces
        gravity_force = np.array([0.0, 0.0, -self.robot_mass * self.gravity])

        total_force = gravity_force.copy()
        for force in manipulation_forces:
            total_force += force

        return total_force

    def is_balanced(self):
        """Check if robot is currently balanced."""
        # Check if CoM projection is within support polygon
        com_xy = self.com_pos[:2]
        return self.point_in_polygon(com_xy, self.support_polygon)

    def generate_balance_correction(self):
        """Generate corrective actions when out of balance."""
        # Calculate capture point
        cp = self.calculate_capture_point()

        # If capture point is outside support, need to step
        if not self.point_in_polygon(cp, self.support_polygon):
            # Plan a step to new support polygon
            step_position = self.plan_balance_step(cp)
            return {'type': 'step', 'position': step_position}
        else:
            # Adjust CoM position
            target_com = self.calculate_balance_target()
            return {'type': 'adjust', 'target': target_com}

    def calculate_capture_point(self):
        """Calculate capture point for balance control."""
        # Capture point = CoM position + CoM velocity * sqrt(height/gravity)
        cp_x = self.com_pos[0] + self.com_vel[0] * self.tau
        cp_y = self.com_pos[1] + self.com_vel[1] * self.tau
        return np.array([cp_x, cp_y])

    def plan_balance_step(self, capture_point):
        """Plan a step to maintain balance."""
        # Move support polygon toward capture point
        # This is a simplified approach
        return capture_point

    def calculate_balance_target(self):
        """Calculate target CoM position for balance."""
        # Move CoM toward center of support polygon
        center_of_support = np.mean(self.support_polygon, axis=0)
        return center_of_support

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

    def calculate_support_polygon(self):
        """Calculate current support polygon."""
        # For a biped with feet at positions f1 and f2
        # This is a simplified representation
        foot_separation = 0.2  # meters
        return np.array([
            [-0.1, -foot_separation/2],  # Left foot
            [-0.1, foot_separation/2],   # Right foot
            [0.1, foot_separation/2],    # Right foot front
            [0.1, -foot_separation/2]    # Left foot front
        ])
```

### 10.3.2 Predictive Balance Control
```python
# predictive_balance.py
class PredictiveBalanceController:
    def __init__(self, horizon=10, dt=0.1):
        self.horizon = horizon  # Prediction horizon (steps)
        self.dt = dt           # Time step
        self.com_trajectory = []
        self.force_trajectory = []

    def predict_balance(self, current_state, planned_forces):
        """Predict balance state over time horizon."""
        # Simulate CoM trajectory with planned forces
        com_states = [current_state['com_pos'].copy()]
        com_vels = [current_state['com_vel'].copy()]

        for i in range(self.horizon):
            # Get planned force for this time step
            if i < len(planned_forces):
                force = planned_forces[i]
            else:
                force = np.zeros(3)  # No force after planning horizon

            # Calculate acceleration
            acc = force / current_state['mass']

            # Update velocity and position
            new_vel = com_vels[-1] + acc * self.dt
            new_pos = com_states[-1] + new_vel * self.dt

            com_vels.append(new_vel)
            com_states.append(new_pos)

        return com_states, com_vels

    def optimize_balance_forces(self, current_state, manipulation_task):
        """Optimize manipulation forces to maintain balance."""
        def cost_function(forces_flat):
            # Reshape flat forces array
            forces = forces_flat.reshape(-1, 3)

            # Predict balance trajectory
            com_states, com_vels = self.predict_balance(current_state, forces)

            # Calculate cost: balance error + task completion
            balance_cost = 0
            task_cost = 0

            for i, com_pos in enumerate(com_states):
                # Balance cost: distance from center of support
                support_center = current_state['support_center']
                balance_error = np.linalg.norm(com_pos[:2] - support_center)
                balance_cost += balance_error**2

                # Task cost: deviation from desired manipulation
                if i < len(manipulation_task.desired_forces):
                    desired_force = manipulation_task.desired_forces[i]
                    task_error = np.linalg.norm(forces[i] - desired_force)
                    task_cost += task_error**2

            return balance_cost * 0.1 + task_cost  # Weighted combination

        # Optimize forces to minimize cost
        initial_forces = np.array(manipulation_task.desired_forces)
        initial_flat = initial_forces.flatten()

        result = minimize(
            cost_function,
            initial_flat,
            method='BFGS'
        )

        if result.success:
            optimized_forces = result.x.reshape(-1, 3)
            return optimized_forces
        else:
            # Return original forces if optimization fails
            return initial_forces
```

## 10.4 Coordinated Multi-Limb Control

### 10.4.1 Limb Coordination Strategies
```python
# limb_coordination.py
class LimbCoordinator:
    def __init__(self, robot_model):
        self.model = robot_model
        self.left_arm = LimbController('left_arm', robot_model.left_arm_config)
        self.right_arm = LimbController('right_arm', robot_model.right_arm_config)
        self.left_leg = LimbController('left_leg', robot_model.left_leg_config)
        self.right_leg = LimbController('right_leg', robot_model.right_leg_config)

        # Coordination rules
        self.coordination_rules = {
            'bimanual': self.coordinate_bimanual,
            'locomanipulation': self.coordinate_locomanipulation,
            'dynamic_balance': self.coordinate_dynamic_balance
        }

    def coordinate_bimanual(self, task):
        """Coordinate both arms for bimanual tasks."""
        # Calculate coordinated motion for both arms
        left_traj, right_traj = self.plan_bimanual_trajectory(task)

        # Execute simultaneously with synchronization
        self.left_arm.execute_trajectory(left_traj)
        self.right_arm.execute_trajectory(right_traj)

    def coordinate_locomanipulation(self, task):
        """Coordinate legs and arms for simultaneous tasks."""
        # Plan manipulation trajectory
        arm_traj = self.plan_arm_trajectory(task.manipulation_goal)

        # Plan locomotion to support manipulation
        leg_traj = self.plan_leg_trajectory_for_manipulation(
            task.manipulation_goal, arm_traj)

        # Execute with timing coordination
        self.execute_coordinated_motion(arm_traj, leg_traj)

    def coordinate_dynamic_balance(self, task):
        """Coordinate all limbs for dynamic balance during manipulation."""
        # High-priority: Balance maintenance
        balance_task = self.generate_balance_task(task)

        # Medium-priority: Manipulation task
        manipulation_task = self.generate_manipulation_task(task)

        # Use whole-body controller to coordinate all limbs
        self.execute_whole_body_task(balance_task, manipulation_task)

    def plan_bimanual_trajectory(self, task):
        """Plan coordinated trajectory for both arms."""
        # Example: Object transfer between hands
        left_waypoints = []
        right_waypoints = []

        # Plan approach, grasp, transfer, release sequence
        for phase in task.phases:
            left_wp, right_wp = self.plan_phase(phase)
            left_waypoints.extend(left_wp)
            right_waypoints.extend(right_wp)

        return left_waypoints, right_waypoints

    def plan_leg_trajectory_for_manipulation(self, manipulation_goal, arm_trajectory):
        """Plan leg motion to support manipulation task."""
        # Adjust stance to support manipulation forces
        # This could involve stepping or adjusting foot placement
        leg_trajectory = []

        for arm_state in arm_trajectory:
            # Calculate manipulation forces at this state
            manipulation_force = self.estimate_manipulation_force(arm_state)

            # Calculate required stance adjustment
            required_stance = self.calculate_stance_for_force(manipulation_force)

            leg_trajectory.append(required_stance)

        return leg_trajectory

    def execute_coordinated_motion(self, arm_trajectory, leg_trajectory):
        """Execute coordinated motion with proper timing."""
        # Synchronize arm and leg execution
        max_len = max(len(arm_trajectory), len(leg_trajectory))

        for i in range(max_len):
            arm_state = arm_trajectory[min(i, len(arm_trajectory)-1)] if i < len(arm_trajectory) else arm_trajectory[-1]
            leg_state = leg_trajectory[min(i, len(leg_trajectory)-1)] if i < len(leg_trajectory) else leg_trajectory[-1]

            # Execute both simultaneously
            self.left_arm.execute_single_state(arm_state['left'])
            self.right_arm.execute_single_state(arm_state['right'])
            self.left_leg.execute_single_state(leg_state['left'])
            self.right_leg.execute_single_state(leg_state['right'])
```

### 10.4.2 Redundancy Resolution
```python
# redundancy_resolution.py
class RedundancyResolver:
    def __init__(self, robot_model):
        self.model = robot_model
        self.null_space_projector = self.calculate_null_space_projector()

    def resolve_redundancy(self, primary_task, secondary_goals):
        """Resolve redundancy by optimizing secondary goals."""
        # Primary task: high-priority task (e.g., end-effector position)
        # Secondary goals: optimization objectives (e.g., joint centering, obstacle avoidance)

        # Calculate null space of primary task
        J_primary = self.calculate_task_jacobian(primary_task)
        J_pinv = np.linalg.pinv(J_primary)
        I = np.eye(self.model.num_joints)
        null_space = I - np.dot(J_pinv, J_primary)

        # Calculate secondary task velocities in null space
        secondary_vel = np.zeros(self.model.num_joints)
        for goal in secondary_goals:
            goal_vel = self.calculate_secondary_velocity(goal)
            # Project into null space of primary task
            goal_vel_null = np.dot(null_space, goal_vel)
            secondary_vel += goal_vel_null

        # Combine primary and secondary velocities
        primary_vel = self.calculate_primary_velocity(primary_task)
        final_vel = primary_vel + secondary_vel

        return final_vel

    def calculate_task_jacobian(self, task):
        """Calculate Jacobian for a specific task."""
        if task.type == 'end_effector_position':
            return self.model.end_effector_jacobian
        elif task.type == 'com_position':
            return self.model.com_jacobian
        else:
            raise ValueError(f"Unknown task type: {task.type}")

    def calculate_secondary_velocity(self, goal):
        """Calculate velocity for secondary optimization goal."""
        if goal.type == 'joint_centering':
            # Move joints toward center of range
            current_joints = self.model.get_current_joints()
            center_joints = goal.target_joints
            error = center_joints - current_joints
            return goal.gain * error
        elif goal.type == 'obstacle_avoidance':
            # Avoid joint limits or obstacles
            return self.calculate_avoidance_velocity(goal)
        else:
            return np.zeros(self.model.num_joints)

    def calculate_primary_velocity(self, task):
        """Calculate velocity for primary task."""
        current_value = self.get_task_value(task)
        error = task.target - current_value
        return task.gain * error

    def get_task_value(self, task):
        """Get current value of task variable."""
        if task.type == 'end_effector_position':
            pos, _ = self.model.forward_kinematics(self.model.get_current_joints())
            return pos
        elif task.type == 'com_position':
            return self.model.calculate_com(self.model.get_current_joints())
        else:
            raise ValueError(f"Unknown task type: {task.type}")
```

## 10.5 Integration Algorithms

### 10.5.1 Model Predictive Control (MPC) for Locomanipulation
```python
# mpc_locomanipulation.py
class LocomanipulationMPC:
    def __init__(self, horizon=20, dt=0.05):
        self.horizon = horizon
        self.dt = dt
        self.state_dim = 12  # Example: CoM pos/vel, joint pos/vel
        self.control_dim = 6  # Example: CoM forces, joint torques

    def setup_optimization(self):
        """Set up MPC optimization problem."""
        # State: [CoM_pos, CoM_vel, joint_pos, joint_vel]
        # Control: [CoM_force, joint_torque]

        # Dynamics model: x_{k+1} = f(x_k, u_k)
        # Cost function: l(x_k, u_k) = ||x_k - x_ref||_Q² + ||u_k||_R²
        # Constraints: joint limits, force limits, balance constraints

        pass

    def solve(self, current_state, reference_trajectory):
        """Solve MPC problem for locomanipulation."""
        # Define optimization variables
        # Minimize: sum of stage costs + terminal cost
        # Subject to: dynamics constraints, state constraints, control constraints

        # This would typically use a specialized optimization solver
        # like OSQP, IPOPT, or similar

        # Return optimal control sequence
        return np.zeros((self.horizon, self.control_dim))

    def update_reference(self, manipulation_task, locomotion_task):
        """Update reference trajectory based on tasks."""
        # Combine manipulation and locomotion references
        combined_reference = self.combine_references(
            manipulation_task, locomotion_task)
        return combined_reference

    def combine_references(self, manipulation_task, locomotion_task):
        """Combine manipulation and locomotion references."""
        # Weighted combination based on task priorities
        manip_weight = manipulation_task.priority
        loco_weight = locomotion_task.priority

        combined_ref = (manip_weight * manipulation_task.reference +
                       loco_weight * locomotion_task.reference) / (manip_weight + loco_weight)

        return combined_ref
```

### 10.5.2 Reactive Control for Real-time Integration
```python
# reactive_control.py
class ReactiveLocomanipulationController:
    def __init__(self):
        self.manipulation_controller = ManipulationController()
        self.locomotion_controller = LocomotionController()
        self.balance_controller = BalanceController()
        self.coordination_manager = CoordinationManager()

    def step(self, sensor_data, task_descriptions):
        """Execute one control step with integration."""
        # 1. Update state estimates
        state = self.update_state(sensor_data)

        # 2. Evaluate current situation
        balance_status = self.balance_controller.evaluate_balance(state)
        manipulation_status = self.manipulation_controller.evaluate_progress(state)
        locomotion_status = self.locomotion_controller.evaluate_progress(state)

        # 3. Determine coordination strategy
        coordination_strategy = self.coordination_manager.select_strategy(
            balance_status, manipulation_status, locomotion_status)

        # 4. Generate control commands based on strategy
        manipulation_cmd = self.manipulation_controller.generate_command(
            state, task_descriptions.manipulation)

        locomotion_cmd = self.locomotion_controller.generate_command(
            state, task_descriptions.locomotion)

        # 5. Apply coordination constraints
        if coordination_strategy == 'balance_priority':
            # Prioritize balance over manipulation
            manipulation_cmd = self.apply_balance_constraints(
                manipulation_cmd, state)
        elif coordination_strategy == 'coordinated':
            # Coordinate both simultaneously
            manipulation_cmd, locomotion_cmd = self.coordinate_commands(
                manipulation_cmd, locomotion_cmd, state)

        # 6. Execute commands
        self.execute_commands(manipulation_cmd, locomotion_cmd)

    def apply_balance_constraints(self, manipulation_cmd, state):
        """Modify manipulation command to maintain balance."""
        # Check if manipulation forces would compromise balance
        predicted_forces = self.estimate_manipulation_forces(manipulation_cmd)

        if self.balance_controller.would_lose_balance(predicted_forces, state):
            # Scale down manipulation forces
            safe_forces = self.balance_controller.calculate_safe_forces(
                predicted_forces, state)
            return self.scale_command_by_forces(manipulation_cmd, safe_forces)

        return manipulation_cmd

    def coordinate_commands(self, manip_cmd, loco_cmd, state):
        """Coordinate manipulation and locomotion commands."""
        # This could involve:
        # - Timing coordination
        # - Force distribution
        # - Sequential execution
        # - Parallel execution with constraints

        # Example: Execute locomotion during manipulation hold
        if self.manipulation_controller.is_holding_object(state):
            # Reduce locomotion speed/effort to maintain grasp
            loco_cmd.velocity *= 0.5

        return manip_cmd, loco_cmd

    def execute_commands(self, manipulation_cmd, locomotion_cmd):
        """Execute the coordinated commands."""
        # Send commands to appropriate subsystems
        self.manipulation_controller.execute(manipulation_cmd)
        self.locomotion_controller.execute(locomotion_cmd)
```

## 10.6 Implementation Example: Coordinated Task Execution

```python
# coordinated_task_example.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import String
import numpy as np


class CoordinatedTaskNode(Node):
    def __init__(self):
        super().__init__('coordinated_task_node')

        # Publishers
        self.arm_cmd_pub = self.create_publisher(JointState, '/arm_commands', 10)
        self.base_cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/task_status', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)

        # Controllers
        self.whole_body_controller = WholeBodyController(None)
        self.balance_manager = BalanceManager()
        self.limb_coordinator = LimbCoordinator(None)

        # Task state
        self.current_joints = {}
        self.imu_data = None
        self.task_queue = []
        self.is_executing = False

        # Control timer
        self.control_timer = self.create_timer(0.02, self.control_loop)  # 50 Hz

    def joint_state_callback(self, msg):
        """Update joint state."""
        for i, name in enumerate(msg.name):
            self.current_joints[name] = msg.position[i]

    def imu_callback(self, msg):
        """Update IMU data."""
        self.imu_data = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

    def add_task(self, task_description):
        """Add task to execution queue."""
        self.task_queue.append(task_description)

    def control_loop(self):
        """Main control loop for coordinated tasks."""
        if not self.task_queue or self.is_executing:
            return

        # Get current robot state
        current_state = self.get_robot_state()

        # Check balance status
        balance_ok = self.check_balance(current_state)

        if not balance_ok:
            # Emergency balance correction
            self.correct_balance(current_state)
            return

        # Get next task
        task = self.task_queue[0]

        # Determine if task requires coordination
        if task.requires_both_locomotion_and_manipulation:
            # Execute coordinated task
            success = self.execute_coordinated_task(task, current_state)
        elif task.requires_manipulation:
            # Pure manipulation task
            success = self.execute_manipulation_task(task, current_state)
        elif task.requires_locomotion:
            # Pure locomotion task
            success = self.execute_locomotion_task(task, current_state)
        else:
            success = True

        if success:
            # Task completed, remove from queue
            self.task_queue.pop(0)

            # Publish completion status
            status_msg = String()
            status_msg.data = f"Task completed: {task.name}"
            self.status_pub.publish(status_msg)

    def get_robot_state(self):
        """Get current robot state for control."""
        state = {
            'joints': self.current_joints.copy(),
            'imu': self.imu_data.copy() if self.imu_data is not None else np.zeros(6),
            'com': self.estimate_com_position(),
            'support_polygon': self.calculate_support_polygon()
        }
        return state

    def check_balance(self, state):
        """Check if robot is currently balanced."""
        # Use IMU data and kinematic model
        com_pos = state['com'][:2]  # X, Y position
        support_polygon = state['support_polygon']

        return self.point_in_polygon(com_pos, support_polygon)

    def correct_balance(self, state):
        """Execute emergency balance correction."""
        # Stop all motion
        self.stop_motion()

        # Calculate balance correction
        com_pos = state['com'][:2]
        support_center = np.mean(state['support_polygon'], axis=0)

        # Move CoM toward support center
        correction_vector = support_center - com_pos
        self.apply_balance_correction(correction_vector)

    def execute_coordinated_task(self, task, state):
        """Execute task requiring both locomotion and manipulation."""
        self.is_executing = True

        try:
            # Plan coordinated motion
            manipulation_plan = self.plan_manipulation(task.manipulation_goal)
            locomotion_plan = self.plan_locomotion(task.locomotion_goal)

            # Execute with coordination
            success = self.limb_coordinator.execute_coordinated_motion(
                manipulation_plan, locomotion_plan)

            return success

        except Exception as e:
            self.get_logger().error(f"Coordinated task failed: {e}")
            return False
        finally:
            self.is_executing = False

    def execute_manipulation_task(self, task, state):
        """Execute pure manipulation task."""
        self.is_executing = True

        try:
            # Plan manipulation motion
            manip_plan = self.plan_manipulation(task.manipulation_goal)

            # Execute with balance monitoring
            for waypoint in manip_plan:
                # Check balance before each movement
                if not self.check_balance(state):
                    self.correct_balance(state)
                    return False

                # Execute manipulation
                self.execute_manipulation_waypoint(waypoint)

            return True

        except Exception as e:
            self.get_logger().error(f"Manipulation task failed: {e}")
            return False
        finally:
            self.is_executing = False

    def execute_locomotion_task(self, task, state):
        """Execute pure locomotion task."""
        self.is_executing = True

        try:
            # Plan locomotion
            loco_plan = self.plan_locomotion(task.locomotion_goal)

            # Execute with manipulation monitoring
            for step in loco_plan:
                # Check if manipulation task needs attention
                if self.manipulation_needs_attention(state):
                    # Pause locomotion, maintain manipulation
                    self.maintain_manipulation(state)

                # Execute locomotion step
                self.execute_locomotion_step(step)

            return True

        except Exception as e:
            self.get_logger().error(f"Locomotion task failed: {e}")
            return False
        finally:
            self.is_executing = False

    def stop_motion(self):
        """Stop all robot motion."""
        # Send zero velocity commands
        zero_twist = Twist()
        self.base_cmd_pub.publish(zero_twist)

        # Send zero joint commands
        zero_joints = JointState()
        zero_joints.position = [0.0] * 6  # Example: 6 joints
        self.arm_cmd_pub.publish(zero_joints)

    def estimate_com_position(self):
        """Estimate center of mass position."""
        # Simplified estimation
        # In practice, use forward dynamics model
        return np.array([0.0, 0.0, 0.8])

    def calculate_support_polygon(self):
        """Calculate current support polygon."""
        # Simplified for biped robot
        return np.array([
            [-0.1, -0.1],
            [-0.1, 0.1],
            [0.1, 0.1],
            [0.1, -0.1]
        ])

    def point_in_polygon(self, point, polygon):
        """Check if point is in polygon."""
        # Implementation from previous examples
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

    def plan_manipulation(self, goal):
        """Plan manipulation trajectory."""
        # Simplified planning
        # In practice, use motion planning algorithms
        return [goal]  # Single waypoint for example

    def plan_locomotion(self, goal):
        """Plan locomotion trajectory."""
        # Simplified planning
        return [goal]  # Single step for example

    def execute_manipulation_waypoint(self, waypoint):
        """Execute single manipulation waypoint."""
        cmd = JointState()
        cmd.position = waypoint  # Joint positions
        cmd.header.stamp = self.get_clock().now().to_msg()
        self.arm_cmd_pub.publish(cmd)

    def execute_locomotion_step(self, step):
        """Execute single locomotion step."""
        cmd = Twist()
        cmd.linear.x = step[0]  # Forward velocity
        cmd.angular.z = step[1]  # Angular velocity
        self.base_cmd_pub.publish(cmd)

    def manipulation_needs_attention(self, state):
        """Check if manipulation task needs attention."""
        # Check if object is slipping, etc.
        return False

    def maintain_manipulation(self, state):
        """Maintain current manipulation state."""
        # Keep grip force, maintain position, etc.
        pass

    def apply_balance_correction(self, correction_vector):
        """Apply balance correction."""
        # Move feet, adjust CoM, etc.
        pass


def main(args=None):
    rclpy.init(args=args)
    coordinated_task_node = CoordinatedTaskNode()

    # Example: Add a task that requires both locomotion and manipulation
    class TaskDescription:
        def __init__(self, name, requires_both=False, requires_manip=False, requires_loco=False):
            self.name = name
            self.requires_both_locomotion_and_manipulation = requires_both
            self.requires_manipulation = requires_manip
            self.requires_locomotion = requires_loco
            self.manipulation_goal = [0.5, 0.2, 0.1]  # Example joint positions
            self.locomotion_goal = [0.1, 0.05]  # Example velocities

    # Add a sample task
    sample_task = TaskDescription("move_and_grasp", requires_both=True)
    coordinated_task_node.add_task(sample_task)

    rclpy.spin(coordinated_task_node)
    coordinated_task_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## 10.7 Advanced Integration Techniques

### 10.7.1 Learning-Based Coordination
Using machine learning to improve coordination strategies based on experience.

### 10.7.2 Adaptive Control
Systems that adapt coordination parameters based on task success and environmental conditions.

## 10.8 Real-World Considerations

### 10.8.1 Computational Complexity
- Whole-body control can be computationally expensive
- Consider real-time constraints
- Use simplified models when possible

### 10.8.2 Sensor Fusion
- Integrate data from multiple sensors
- Handle sensor delays and uncertainties
- Maintain consistency across sensor modalities

### 10.8.3 Safety Considerations
- Implement emergency stop mechanisms
- Monitor for dangerous situations
- Maintain graceful degradation

## 10.9 Best Practices

1. **Hierarchical Control**: Use multiple control levels for different time scales
2. **Modularity**: Keep locomotion and manipulation controllers separate but coordinated
3. **Real-time Performance**: Ensure control loops meet timing requirements
4. **Robustness**: Handle failures gracefully and maintain safety
5. **Testing**: Validate coordination in simulation before real robot deployment

## Practical Exercise

### Exercise 10.1: Coordinated Locomotion-Manipulation Task
**Objective**: Create a system that coordinates walking and manipulation for a pick-and-place task

1. Implement a whole-body controller that manages both arms and legs
2. Design balance maintenance during manipulation
3. Create a coordinated pick-and-place task
4. Implement emergency balance recovery
5. Test the system with varying task parameters
6. Analyze coordination performance and balance metrics

**Deliverable**: Complete coordinated control system with pick-and-place capability and balance maintenance.

## Summary

Week 10 covered the integration of locomotion and manipulation in humanoid robots. You learned about whole-body control frameworks, dynamic balance management, and coordinated multi-limb control strategies. This integration is essential for humanoid robots to perform complex real-world tasks that require both mobility and dexterity.

[Next: Week 11 - Multimodal Perception →](./week11-multimodal-perception.md) | [Previous: Week 9 - Manipulation & Grasping ←](./week9-manipulation-grasping.md)