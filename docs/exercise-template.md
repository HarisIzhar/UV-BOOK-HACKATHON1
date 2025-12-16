---
sidebar_position: 103
---

# Practical Exercise Template

## Overview

This template provides a standardized format for creating practical exercises throughout the Physical AI & Humanoid Robotics book. Each exercise should follow this structure to ensure consistency and clear learning outcomes.

## Exercise Template Structure

### Exercise [WEEK].[EXERCISE NUMBER]: [EXERCISE TITLE]

**Objective**: [Clear, specific objective that students should achieve]

**Prerequisites**:
- [Prerequisite 1 - e.g., "Completion of Week 2 content"]
- [Prerequisite 2 - e.g., "ROS 2 environment setup"]
- [Prerequisite 3 - e.g., "Basic Python programming knowledge"]

**Learning Outcomes**: By completing this exercise, you will be able to:
- [Outcome 1 - specific, measurable skill]
- [Outcome 2 - specific, measurable skill]
- [Outcome 3 - specific, measurable skill]

**Estimated Time**: [X] hours/minutes

### Background Information

[Provide necessary context for the exercise. Include any relevant theory, concepts, or information students need to understand before beginning the exercise.]

### Exercise Steps

#### Step 1: [STEP TITLE]
**Description**: [Detailed instructions for this step]

**Expected Result**: [What students should see or achieve at the end of this step]

**Hints/Tips**:
- [Helpful hint 1]
- [Helpful hint 2]

#### Step 2: [STEP TITLE]
**Description**: [Detailed instructions for this step]

**Expected Result**: [What students should see or achieve at the end of this step]

**Verification**: [How students can verify they completed this step correctly]

[Continue with additional steps as needed]

### Troubleshooting

**Common Issues**:
- **Issue 1**: [Description of common problem] → **Solution**: [How to fix it]
- **Issue 2**: [Description of common problem] → **Solution**: [How to fix it]

**Debugging Tips**:
- [Tip 1 for debugging]
- [Tip 2 for debugging]

### Deliverables

Students must submit/provide evidence of:
1. [Deliverable 1 - e.g., "Working Python script that performs the required function"]
2. [Deliverable 2 - e.g., "Screenshot of successful execution"]
3. [Deliverable 3 - e.g., "Brief written explanation of key concepts learned"]

### Assessment Criteria

**Grading Rubric**:
- **Excellent (A)**: [Criteria for excellent performance]
- **Proficient (B)**: [Criteria for proficient performance]
- **Developing (C)**: [Criteria for developing performance]
- **Beginning (D)**: [Criteria for beginning performance]

**Self-Assessment Questions**:
- [Question 1 - "Can you explain the key concept to someone else?"]
- [Question 2 - "Can you modify the solution for a different scenario?"]
- [Question 3 - "Can you identify potential improvements to your implementation?"]

### Extensions (Optional)

For advanced students who complete the exercise early:
- [Extension activity 1]
- [Extension activity 2]

### Resources

**Additional Reading**:
- [Resource 1 with link if applicable]
- [Resource 2 with link if applicable]

**Helpful Commands**:
```bash
# [Description of command]
command goes here
```

---

## Example Exercise Using the Template

### Exercise 2.1: Simple Publisher-Subscriber System

**Objective**: Create a ROS 2 system with one publisher and one subscriber that communicate over a custom topic.

**Prerequisites**:
- Completion of Week 2 content
- ROS 2 Humble environment setup
- Basic Python programming knowledge

**Learning Outcomes**: By completing this exercise, you will be able to:
- Create a ROS 2 publisher node that sends messages
- Create a ROS 2 subscriber node that receives messages
- Use launch files to orchestrate multiple nodes
- Verify communication between nodes using ROS 2 tools

**Estimated Time**: 2 hours

### Background Information

In ROS 2, nodes communicate through topics using a publish-subscribe pattern. Publishers send messages to topics, and subscribers receive messages from topics. This pattern enables decoupled, asynchronous communication between different parts of a robotic system.

### Exercise Steps

#### Step 1: Create a New ROS 2 Package
**Description**: Create a new ROS 2 package named `exercise_2_1` with proper dependencies.

**Expected Result**: A new ROS 2 package with correct structure and dependencies.

**Commands**:
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python exercise_2_1
```

#### Step 2: Implement the Publisher Node
**Description**: Create a publisher node that sends "Hello World" messages every second.

**Expected Result**: A working publisher node that sends messages to `/chatter` topic.

**Code Template**:
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 1.0  # seconds
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
```

#### Step 3: Implement the Subscriber Node
**Description**: Create a subscriber node that receives and logs messages from the publisher.

**Expected Result**: A working subscriber node that receives and logs messages.

**Verification**: Run both nodes and verify messages are received correctly.

#### Step 4: Create a Launch File
**Description**: Create a launch file to start both nodes simultaneously.

**Expected Result**: A launch file that starts both publisher and subscriber nodes.

**Verification**: Use the launch file to start the system and verify communication.

### Deliverables

Students must submit/provide evidence of:
1. Complete ROS 2 package with publisher, subscriber, and launch file
2. Screenshot showing successful communication between nodes
3. Brief written explanation of the publish-subscribe pattern and its advantages

This template ensures consistency across all practical exercises in the book while providing clear guidance and expectations for students.