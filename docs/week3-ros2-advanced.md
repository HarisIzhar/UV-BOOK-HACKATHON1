---
sidebar_position: 4
---

# Week 3: Advanced ROS 2 Patterns

## Learning Objectives

By the end of this week, you will be able to:
- Implement ROS 2 actions for long-running tasks with feedback
- Create custom message and service definitions
- Use parameters for system configuration
- Implement advanced debugging and profiling techniques
- Design robust communication patterns for multi-robot systems

## 3.1 ROS 2 Actions for Long-Running Tasks

Actions are ideal for long-running tasks that require feedback and the ability to be canceled:

```python
# action/Fibonacci.action
#goal definition
int32 order
---
#result definition
int32[] sequence
---
#feedback definition
int32[] sequence
```

### Action Server Implementation:

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node

from my_robot_interfaces.action import Fibonacci


class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            callback_group=rclpy.callback_groups.ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Returning result: {result.sequence}')

        return result
```

## 3.2 Custom Message and Service Definitions

### Creating Custom Messages:

```
# msg/RobotState.msg
float64 x
float64 y
float64 theta
float64 linear_velocity
float64 angular_velocity
time timestamp
```

### Creating Custom Services:

```
# srv/MoveToGoal.srv
# Request
float64 x
float64 y
float64 theta
---
# Response
bool success
string message
```

### Package Configuration for Custom Interfaces:

```xml
<!-- package.xml -->
<depend>rosidl_default_generators</depend>

<member_of_group>rosidl_interface_packages</member_of_group>
```

```cmake
# CMakeLists.txt
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotState.msg"
  "srv/MoveToGoal.srv"
  "action/Fibonacci.action"
)
```

## 3.3 Parameter Management

Parameters provide a way to configure nodes at runtime:

```python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'turtlebot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_distance', 0.5)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value

        # Parameter callback for dynamic reconfiguration
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.value > 2.0:
                return SetParametersResult(successful=False, reason='Max velocity too high')
        return SetParametersResult(successful=True)
```

## 3.4 Advanced Communication Patterns

### Client-Server Pattern with Services:

```python
class FibonacciClient(Node):
    def __init__(self):
        super().__init__('fibonacci_client')
        self.cli = self.create_client(Fibonacci, 'fibonacci')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self.future = self.cli.call_async(goal_msg)
        self.future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        result = future.result()
        self.get_logger().info(f'Result: {result.sequence}')
```

### Multi-Robot Communication Patterns:

```python
# Using namespaces for multi-robot systems
class MultiRobotNode(Node):
    def __init__(self, robot_name):
        super().__init__(f'{robot_name}_node', namespace=f'/{robot_name}')

        # Topics will be namespaced: /robot1/sensor_data, /robot2/sensor_data
        self.sensor_sub = self.create_subscription(
            SensorData,
            'sensor_data',
            self.sensor_callback,
            10
        )
```

## 3.5 Performance Optimization and Profiling

### Performance Tools:
- **ros2 doctor**: System health check
- **ros2 run tracetools trace**: Performance tracing
- **ros2 topic hz**: Measure topic frequency
- **ros2 lifecycle**: Lifecycle node management

### Memory and CPU Optimization:

```python
# Efficient message handling
def sensor_callback(self, msg):
    # Process message efficiently
    if self.should_process(msg):
        # Only process relevant messages
        self.process_message(msg)
```

## 3.6 Security in ROS 2

ROS 2 includes security features:
- **Authentication**: Verify node identity
- **Authorization**: Control access to topics/services
- **Encryption**: Secure message transmission

### Setting up Security:
```bash
export ROS_SECURITY_ENABLE=true
export ROS_SECURITY_STRATEGY=Enforce
export ROS_SECURITY_KEYSTORE=/path/to/keystore
```

## Practical Exercise

### Exercise 3.1: Multi-Robot Navigation System
**Objective**: Create a multi-robot system with coordinated navigation

1. Create a custom message for robot state information
2. Implement an action server for navigation goals
3. Create a parameter server for fleet configuration
4. Design a communication pattern for robot coordination
5. Test the system with at least 2 simulated robots

**Deliverable**: Complete multi-robot navigation system with custom messages, actions, parameters, and coordination logic.

## Summary

Week 3 covered advanced ROS 2 patterns including actions, custom interfaces, parameters, and multi-robot communication. You learned to implement complex communication patterns and optimize system performance. These skills are essential for building sophisticated robotic systems.

[Next: Week 4 - Robot Modeling & Physics (URDF/SDF) →](./week4-simulation-modeling.md) | [Previous: Week 2 - ROS 2 Architecture ←](./week2-ros2-architecture.md)