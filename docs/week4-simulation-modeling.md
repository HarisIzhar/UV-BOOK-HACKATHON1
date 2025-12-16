---
sidebar_position: 5
---

# Week 4: Robot Modeling & Physics (URDF/SDF)

## Learning Objectives

By the end of this week, you will be able to:
- Create robot models using URDF (Unified Robot Description Format)
- Design robot kinematic chains with proper joint definitions
- Configure physical properties for accurate simulation
- Integrate sensors into robot models
- Validate robot models using simulation tools

## 4.1 Introduction to Robot Modeling

Robot modeling is the process of creating digital representations of physical robots for simulation, visualization, and control development. In robotics, two primary formats are used:

- **URDF (Unified Robot Description Format)**: ROS standard for robot modeling
- **SDF (Simulation Description Format)**: Gazebo's native format

Both formats describe a robot's physical and kinematic properties, enabling accurate simulation and control.

## 4.2 URDF Fundamentals

URDF is an XML-based format that describes robot structure using a tree of connected links and joints.

### 4.2.1 Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </visual>
  </link>
</robot>
```

### 4.2.2 Links and Joints

**Links** represent rigid bodies with physical properties:
- **Visual**: How the link appears in simulation
- **Collision**: How the link interacts with other objects
- **Inertial**: Mass, center of mass, and inertial tensor

**Joints** define how links connect:
- **Fixed**: No movement between links
- **Revolute**: Single-axis rotation with limits
- **Continuous**: Unlimited single-axis rotation
- **Prismatic**: Single-axis translation with limits
- **Floating**: 6DOF movement
- **Planar**: Movement in a plane

## 4.3 Creating a Simple Robot Model

Let's create a simple differential drive robot:

```xml
<?xml version="1.0"?>
<robot name="diff_drive_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Constants -->
  <xacro:property name="PI" value="3.1415926535897931"/>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Wheels -->
  <xacro:macro name="wheel" params="suffix parent x_reflect y_reflect">
    <link name="${suffix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="${PI/2} 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.04"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="${PI/2} 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.04"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.2"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${suffix}_wheel_hinge" type="continuous">
      <parent link="${parent}"/>
      <child link="${suffix}_wheel"/>
      <origin xyz="${x_reflect*0.15} ${y_reflect*0.2} 0" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <!-- Instantiate wheels -->
  <xacro:wheel suffix="left" parent="base_link" x_reflect="1" y_reflect="1"/>
  <xacro:wheel suffix="right" parent="base_link" x_reflect="1" y_reflect="-1"/>
</robot>
```

## 4.4 Physical Properties and Inertial Parameters

Accurate physical properties are crucial for realistic simulation:

### 4.4.1 Mass and Inertia
- **Mass**: Total weight of the link
- **Center of Mass**: Point where the mass is concentrated
- **Inertia Tensor**: Resistance to rotational motion

### 4.4.2 Material Properties
- **Friction**: Static and dynamic friction coefficients
- **Bounce**: Restitution coefficient for collision response
- **Damping**: Energy loss during motion

## 4.5 SDF (Simulation Description Format)

SDF is Gazebo's native format that supports more advanced features:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_robot">
    <link name="chassis">
      <pose>0 0 0.1 0 0 0</pose>
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.0395833</ixx>
          <iyy>0.0395833</iyy>
          <izz>0.075</izz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.3 0.15</size>
          </box>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.3 0.15</size>
          </box>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0 1 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
```

## 4.6 Sensor Integration

Sensors can be attached to robot models for perception:

### 4.6.1 Camera Sensor
```xml
<sensor name="camera" type="camera">
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### 4.6.2 IMU Sensor
```xml
<sensor name="imu" type="imu">
  <always_on>1</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </x>
    </angular_velocity>
  </imu>
</sensor>
```

## 4.7 Model Validation and Testing

### 4.7.1 URDF Validation
```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# View robot model
urdf_to_graphiz /path/to/robot.urdf
```

### 4.7.2 Visualization
```bash
# Launch robot in RViz
roslaunch urdf_tutorial display.launch model:="/path/to/robot.urdf"
```

## 4.8 Best Practices

1. **Start Simple**: Begin with basic shapes and add complexity gradually
2. **Use Xacro**: Leverage Xacro macros for parameterization and reusability
3. **Validate Regularly**: Check models frequently during development
4. **Match Real Robot**: Ensure simulation parameters match physical robot
5. **Consider Performance**: Balance detail with simulation speed

## Practical Exercise

### Exercise 4.1: Create a Custom Robot Model
**Objective**: Design a robot model with at least 4 links and 3 joints

1. Create a URDF file for a simple manipulator robot with:
   - Base link
   - 2 rotating joints
   - 1 prismatic joint
   - End effector link

2. Include proper inertial parameters for each link
3. Add visual and collision properties
4. Validate the model using URDF tools
5. Visualize the model in RViz

**Deliverable**: Complete URDF file with proper kinematic chain and physical properties.

## Summary

Week 4 introduced robot modeling using URDF and SDF formats. You learned to create kinematic chains, configure physical properties, and integrate sensors into robot models. These skills are essential for simulation-based robotics development.

[Next: Week 5 - Simulation Environments →](./week5-simulation-environments.md) | [Previous: Week 3 - Advanced ROS 2 Patterns ←](./week3-ros2-advanced.md)