---
sidebar_position: 6
---

# Week 5: Simulation Environments

## Learning Objectives

By the end of this week, you will be able to:
- Set up and configure Gazebo simulation environments
- Design complex simulation worlds with obstacles and objects
- Integrate robots into simulation environments
- Configure physics properties and sensor models
- Implement sensor simulation and data processing

## 5.1 Introduction to Simulation in Robotics

Simulation is a critical component of robotics development, providing a safe, cost-effective environment for testing algorithms before deployment on physical robots. Key benefits include:

- **Safety**: Test dangerous scenarios without risk
- **Cost**: Reduce hardware wear and development costs
- **Repeatability**: Exact same conditions for consistent testing
- **Speed**: Accelerate development cycles
- **Scalability**: Test multi-robot scenarios

## 5.2 Gazebo Simulation Environment

Gazebo is a 3D simulation environment that provides:
- High-fidelity physics simulation
- Realistic sensor simulation
- Flexible environment creation
- Integration with ROS/ROS2

### 5.2.1 Gazebo Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Gazebo Simulator                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Physics    │  │  Sensor     │  │  Graphics   │        │
│  │  Engine     │  │  Models     │  │  Engine     │        │
│  │  (ODE)      │  │  (Ray,      │  │  (OGRE)     │        │
│  └─────────────┘  │  Camera,     │  └─────────────┘        │
│                   │  IMU, etc.)  │                        │
│                   └─────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 5.2.2 Launching Gazebo
```bash
# Launch empty world
gazebo

# Launch with specific world
gazebo /path/to/world.world

# Launch with ROS integration
roslaunch gazebo_ros empty_world.launch
```

## 5.3 World Definition and Environment Creation

### 5.3.1 World File Structure
Gazebo worlds are defined in SDF format:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- Physics engine -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Environment lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.6 0.4 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.0 0.0 0.0 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Include robot -->
    <include>
      <uri>model://my_robot</uri>
      <pose>0 0 0.5 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### 5.3.2 Creating Obstacles and Objects

```xml
<!-- Static box obstacle -->
<model name="box_obstacle">
  <pose>2 2 0.5 0 0 0</pose>
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <box>
          <size>1 1 1</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>1 1 1</size>
        </box>
      </geometry>
      <material>
        <ambient>0.5 0.5 0.5 1</ambient>
        <diffuse>0.5 0.5 0.5 1</diffuse>
      </material>
    </visual>
  </link>
</model>

<!-- Moving object -->
<model name="moving_object">
  <pose>0 0 1 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <sphere>
          <radius>0.2</radius>
        </sphere>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <sphere>
          <radius>0.2</radius>
        </sphere>
      </geometry>
      <material>
        <ambient>1 0 0 1</ambient>
        <diffuse>1 0 0 1</diffuse>
      </material>
    </visual>
  </link>
</model>
```

## 5.4 Physics Configuration

### 5.4.1 Physics Engine Parameters
- **Max Step Size**: Time increment for physics updates
- **Real Time Factor**: Simulation speed relative to real time
- **Update Rate**: Frequency of physics calculations

### 5.4.2 Material Properties
```xml
<material name="friction_material">
  <pbr>
    <metal>
      <albedo_map>file://materials/textures/rough_metal.png</albedo_map>
      <roughness_map>file://materials/textures/roughness.png</roughness_map>
      <metalness_map>file://materials/textures/metalness.png</metalness_map>
    </metal>
  </pbr>
</material>
```

## 5.5 Sensor Simulation

### 5.5.1 Camera Sensor Configuration
```xml
<sensor name="camera" type="camera">
  <pose>0.1 0 0.1 0 0 0</pose>
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
    <save enabled="false">
      <path>/tmp/camera_images</path>
    </save>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### 5.5.2 LIDAR Sensor Configuration
```xml
<sensor name="lidar" type="ray">
  <pose>0.1 0 0.2 0 0 0</pose>
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>10.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

## 5.6 ROS Integration with Gazebo

### 5.6.1 Gazebo Plugins
Gazebo plugins enable ROS integration:

```xml
<gazebo>
  <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>cmd_vel:=cmd_vel</remapping>
      <remapping>odom:=odom</remapping>
    </ros>
    <left_joint>left_wheel_hinge</left_joint>
    <right_joint>right_wheel_hinge</right_joint>
    <wheel_separation>0.3</wheel_separation>
    <wheel_diameter>0.1</wheel_diameter>
    <max_wheel_torque>20</max_wheel_torque>
    <max_wheel_acceleration>1.0</max_wheel_acceleration>
    <command_topic>cmd_vel</command_topic>
    <odometry_topic>odom</odometry_topic>
    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_link</robot_base_frame>
  </plugin>
</gazebo>
```

### 5.6.2 Launch File Integration
```xml
<!-- launch/simulation.launch -->
<launch>
  <!-- Start Gazebo with world -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(find my_robot_pkg)/worlds/my_world.world"/>
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
    <arg name="headless" value="false"/>
    <arg name="debug" value="false"/>
  </include>

  <!-- Spawn robot in Gazebo -->
  <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model"
        args="-file $(find my_robot_pkg)/urdf/robot.urdf -urdf -model my_robot"
        respawn="false" output="screen"/>
</launch>
```

## 5.7 Unity Simulation Environment

Unity provides an alternative simulation environment with:
- High-quality graphics rendering
- Physics simulation with PhysX
- VR/AR support
- Real-time ray tracing capabilities

### 5.7.1 Unity-ROS Bridge
The Unity-ROS bridge enables communication between Unity and ROS systems.

### 5.7.2 Environment Assets
Unity allows creation of complex environments using:
- 3D models and assets
- Procedural generation
- Physics materials and properties
- Lighting and atmospheric effects

## 5.8 Simulation Testing Strategies

### 5.8.1 Unit Testing in Simulation
- Test individual robot components
- Validate sensor models
- Verify control algorithms

### 5.8.2 Integration Testing
- Multi-robot scenarios
- Complex environment interactions
- System-level functionality

### 5.8.3 Performance Testing
- Simulation speed vs. real-time factor
- Sensor data throughput
- Physics stability

## 5.9 Best Practices

1. **Start Simple**: Begin with basic environments and add complexity
2. **Validate Physics**: Ensure realistic physical interactions
3. **Match Sensors**: Simulated sensors should match real hardware
4. **Monitor Performance**: Maintain real-time simulation when possible
5. **Document Environments**: Keep track of world parameters for reproducibility

## Practical Exercise

### Exercise 5.1: Create a Navigation Testing Environment
**Objective**: Design a Gazebo world for testing navigation algorithms

1. Create a world file with:
   - Start and goal locations
   - Multiple obstacles of different shapes
   - Dynamic elements (moving objects)
   - Varied terrain types

2. Configure appropriate physics parameters
3. Add a camera sensor for overhead view
4. Test the environment with a simple robot model
5. Document the environment parameters

**Deliverable**: Complete world file with obstacles, sensors, and documentation.

## Summary

Week 5 covered simulation environments including Gazebo and Unity. You learned to create complex simulation worlds, configure physics and sensors, and integrate with ROS systems. Simulation environments are essential for safe and cost-effective robotics development.

[Next: Week 6 - Isaac ROS Fundamentals →](./week6-isaac-fundamentals.md) | [Previous: Week 4 - Robot Modeling ←](./week4-simulation-modeling.md)