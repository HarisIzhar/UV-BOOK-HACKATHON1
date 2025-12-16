---
sidebar_position: 3
---

# Week 2: ROS 2 Architecture & Communication

## Learning Objectives

By the end of this week, you will be able to:
- Understand ROS 2 architecture and its distributed computing model
- Create ROS 2 packages with nodes, topics, services, and actions
- Implement publisher-subscriber communication patterns
- Use launch files to orchestrate complex systems
- Debug distributed ROS 2 systems

## 2.1 ROS 2 Architecture Overview

ROS 2 (Robot Operating System 2) is the next-generation middleware for robotics applications, designed for:
- **Real-time systems**: Deterministic behavior for time-critical applications
- **Distributed computing**: Multi-robot systems and cloud robotics
- **Security**: Built-in security features for safe deployment
- **Industry standards**: Compliance with DDS (Data Distribution Service) standard

### Key Architectural Components:
- **Nodes**: Processes that perform computation
- **Topics**: Channels for message passing (publish/subscribe)
- **Services**: Request/reply communication patterns
- **Actions**: Goal-oriented communication with feedback
- **Parameters**: Configuration values shared across nodes
- **Launch files**: System orchestration and process management

## 2.2 Creating ROS 2 Packages

A ROS 2 package is the basic building block containing:
- **Source code** (C++ or Python)
- **Configuration files** (package.xml, CMakeLists.txt/setup.py)
- **Launch files** (for system orchestration)
- **Test files** (for validation)

### Package Structure:
```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml            # Package metadata
├── src/                   # Source code
│   ├── publisher_node.cpp
│   └── subscriber_node.cpp
├── launch/                # Launch files
│   └── robot_system.launch.py
├── config/                # Configuration files
└── test/                  # Unit tests
```

## 2.3 Nodes and Communication Patterns

### Nodes
Nodes are the fundamental computational units in ROS 2:

```python
# Python example
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World'
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics (Publish/Subscribe)
- **Asynchronous** communication pattern
- **Many-to-many**: Multiple publishers and subscribers
- **Real-time** capable with Quality of Service (QoS) settings

### Services (Request/Reply)
- **Synchronous** communication pattern
- **One-to-one**: Single client-server interaction
- **Blocking**: Client waits for response

### Actions (Goal-Based)
- **Asynchronous** with feedback and status
- **Goal-oriented**: Long-running tasks with intermediate feedback
- **Cancellable**: Goals can be preempted

## 2.4 Quality of Service (QoS) Settings

QoS profiles define communication behavior:

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

# Reliable communication with queue size of 10
qos_profile = QoSProfile(
    depth=10,
    durability=QoSDurabilityPolicy.VOLATILE,
    reliability=QoSReliabilityPolicy.RELIABLE
)
```

### QoS Parameters:
- **Reliability**: Best effort vs. reliable delivery
- **Durability**: Volatile vs. transient local
- **History**: Keep all vs. keep last N messages
- **Depth**: Queue size for history

## 2.5 Launch Files for System Orchestration

Launch files allow you to start multiple nodes with a single command:

```python
# launch/robot_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='publisher_node',
            name='publisher',
            parameters=[
                {'param1': 'value1'},
                {'param2': 42}
            ]
        ),
        Node(
            package='my_robot_package',
            executable='subscriber_node',
            name='subscriber'
        )
    ])
```

## 2.6 Debugging Distributed Systems

### Tools for ROS 2 Debugging:
- **ros2 topic**: Inspect topic data and connections
- **ros2 service**: Call services and inspect interfaces
- **ros2 action**: Monitor and send action goals
- **ros2 node**: List and info about running nodes
- **rqt**: GUI tools for visualization and debugging
- **ros2 bag**: Record and replay data for offline analysis

### Common Debugging Patterns:
- **Echo topics**: `ros2 topic echo /topic_name`
- **Call services**: `ros2 service call /service_name interface_type request_data`
- **Check node graph**: `ros2 run rqt_graph rqt_graph`

## Practical Exercise

### Exercise 2.1: Simple Publisher-Subscriber System
**Objective**: Create a simple ROS 2 system with one publisher and one subscriber

1. Create a new ROS 2 package named `week2_tutorial`
2. Implement a publisher node that publishes a custom message every 1 second
3. Implement a subscriber node that receives and logs the message
4. Create a launch file to start both nodes
5. Test the system using ROS 2 command-line tools

**Deliverable**: Complete ROS 2 package with publisher, subscriber, and launch file that demonstrates basic communication.

## Summary

Week 2 introduced ROS 2 architecture and communication patterns. You learned to create packages, implement different communication patterns (topics, services, actions), configure QoS settings, and use launch files for system orchestration. These concepts form the foundation for all communication in your robotic systems.

[Next: Week 3 - Advanced ROS 2 Patterns →](./week3-ros2-advanced.md) | [Previous: Week 1 - Introduction to Physical AI ←](./week1-intro-physical-ai.md)