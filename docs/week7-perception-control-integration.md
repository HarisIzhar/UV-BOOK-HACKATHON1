---
sidebar_position: 8
---

# Week 7: Perception & Control Integration

## Learning Objectives

By the end of this week, you will be able to:
- Integrate perception outputs with control systems
- Design feedback control loops using sensor data
- Implement state estimation from multiple sensors
- Create sensor fusion algorithms for robust perception
- Develop closed-loop control systems with perception feedback

## 7.1 Introduction to Perception-Control Integration

Perception-control integration is the process of connecting sensor-based environmental understanding with robot control systems. This integration enables robots to:
- Navigate safely through unknown environments
- Manipulate objects based on visual feedback
- Adapt behavior based on sensor observations
- Achieve complex autonomous behaviors

### 7.1.1 Integration Architecture
```
┌─────────────────────────────────────────────────────────────┐
│              Perception-Action Loop                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Sensors   │───▶│  Perception │───▶│   Control   │     │
│  │             │    │   System    │    │   System    │     │
│  │ • Cameras   │    │ • Object    │    │ • Planning  │     │
│  │ • LiDAR     │    │   Detection │    │ • Execution │     │
│  │ • IMU       │    │ • SLAM      │    │ • Feedback  │     │
│  │ • GPS       │    │ • State Est │    │ • Adaptation│     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                   │                   │          │
│         └───────────────────┼───────────────────┘          │
│                             │                              │
│                        ┌────▼────┐                         │
│                        │  Robot  │                         │
│                        │  Action │                         │
│                        │  Space  │                         │
│                        └─────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## 7.2 State Estimation Fundamentals

### 7.2.1 Robot State Representation
The robot's state typically includes:
- **Position**: (x, y, z) coordinates in world frame
- **Orientation**: (roll, pitch, yaw) or quaternion
- **Velocity**: Linear and angular velocities
- **Uncertainty**: Covariance matrix representing confidence

### 7.2.2 Kalman Filter for State Estimation
```python
# kalman_filter.py
import numpy as np
from scipy.linalg import block_diag


class KalmanFilter:
    def __init__(self, dim_x, dim_z):
        """Initialize Kalman Filter.

        Args:
            dim_x: State dimension
            dim_z: Measurement dimension
        """
        self.dim_x = dim_x
        self.dim_z = dim_z

        # State vector [x, y, vx, vy]
        self.x = np.zeros((dim_x, 1))

        # State covariance matrix
        self.P = np.eye(dim_x) * 1000

        # Process noise covariance
        self.Q = np.eye(dim_x) * 0.1

        # Measurement noise covariance
        self.R = np.eye(dim_z) * 1.0

        # State transition matrix
        self.F = np.eye(dim_x)

        # Measurement function
        self.H = np.zeros((dim_z, dim_x))

    def predict(self):
        """Prediction step of Kalman filter."""
        # x = F * x
        self.x = np.dot(self.F, self.x)

        # P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """Update step of Kalman filter.

        Args:
            z: Measurement vector
        """
        # Innovation: y = z - H * x
        y = z - np.dot(self.H, self.x)

        # Innovation covariance: S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman gain: K = P * H^T * S^(-1)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update state: x = x + K * y
        self.x = self.x + np.dot(K, y)

        # Update covariance: P = (I - K * H) * P
        I_KH = np.eye(self.dim_x) - np.dot(K, self.H)
        self.P = np.dot(I_KH, self.P)


class RobotStateEstimator:
    def __init__(self):
        # 4D state: [x, y, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # Initialize state transition matrix for constant velocity model
        dt = 0.05  # 20 Hz
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Initialize measurement matrix (only position measured)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

    def update_position(self, x, y):
        """Update state with new position measurement."""
        z = np.array([[x], [y]])
        self.kf.update(z)
        self.kf.predict()  # Prepare for next prediction

        return self.kf.x.flatten()
```

### 7.2.3 Extended Kalman Filter (EKF)
For nonlinear systems:
```python
class ExtendedKalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    def predict(self, fx, FJacobian):
        """Nonlinear prediction step."""
        self.x = fx(self.x)
        F = FJacobian(self.x)
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def update(self, z, hx, HJacobian):
        """Nonlinear update step."""
        H = HJacobian(self.x)
        y = z - hx(self.x)

        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        I_KH = np.eye(self.dim_x) - np.dot(K, H)
        self.P = np.dot(I_KH, self.P)
```

## 7.3 Sensor Fusion Techniques

### 7.3.1 Kalman Filter-Based Fusion
```python
# sensor_fusion.py
import numpy as np


class SensorFusionNode:
    def __init__(self):
        # State: [x, y, theta, vx, vy, omega]
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 1000

        # Sensor covariances
        self.camera_cov = np.diag([0.01, 0.01, 0.001])  # x, y, theta
        self.odom_cov = np.diag([0.02, 0.02, 0.01])     # x, y, theta
        self.imu_cov = np.diag([0.001, 0.001, 0.0001])  # vx, vy, omega

    def fuse_camera_odom(self, camera_pose, odom_pose):
        """Fuse camera and odometry measurements."""
        # Measurement vector [x, y, theta]
        z_cam = np.array(camera_pose[:3]).reshape(-1, 1)
        z_odom = np.array(odom_pose[:3]).reshape(-1, 1)

        # Measurement matrices
        H = np.array([
            [1, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0]   # theta
        ])

        # Combined measurement and covariance
        R_combined = np.linalg.inv(
            np.linalg.inv(self.camera_cov) +
            np.linalg.inv(self.odom_cov)
        )

        z_combined = (
            np.dot(np.linalg.inv(self.camera_cov), z_cam) +
            np.dot(np.linalg.inv(self.odom_cov), z_odom)
        )
        z_combined = np.dot(R_combined, z_combined)

        # Kalman update
        S = np.dot(np.dot(H, self.covariance), H.T) + R_combined
        K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))

        innovation = z_combined - np.dot(H, self.state.reshape(-1, 1))
        self.state = self.state + np.dot(K, innovation).flatten()
        self.covariance = np.dot((np.eye(6) - np.dot(K, H)), self.covariance)

        return self.state.copy()
```

### 7.3.2 Particle Filter for Non-Gaussian Distributions
```python
class ParticleFilter:
    def __init__(self, num_particles=1000):
        self.num_particles = num_particles
        self.particles = np.random.uniform(-5, 5, (num_particles, 3))  # x, y, theta
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, control_input, noise_std=0.1):
        """Predict particle motion based on control input."""
        # Apply motion model with noise
        for i in range(self.num_particles):
            # Simple motion model
            self.particles[i, 0] += control_input[0] + np.random.normal(0, noise_std)
            self.particles[i, 1] += control_input[1] + np.random.normal(0, noise_std)
            self.particles[i, 2] += control_input[2] + np.random.normal(0, noise_std * 0.1)

    def update(self, measurement, measurement_noise=0.1):
        """Update particle weights based on measurement."""
        for i in range(self.num_particles):
            # Calculate predicted measurement
            predicted_meas = self.particles[i, :2]  # x, y

            # Calculate likelihood
            diff = measurement - predicted_meas
            likelihood = np.exp(-0.5 * np.sum(diff**2) / (measurement_noise**2))

            self.weights[i] *= likelihood

        # Normalize weights
        self.weights += 1e-300  # Avoid zero weights
        self.weights /= np.sum(self.weights)

    def resample(self):
        """Resample particles based on weights."""
        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        """Get state estimate from particles."""
        return np.average(self.particles, weights=self.weights, axis=0)
```

## 7.4 Control Systems with Perception Feedback

### 7.4.1 Feedback Control Architecture
```python
# feedback_control.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import numpy as np
from cv_bridge import CvBridge


class PerceptionControlNode(Node):
    def __init__(self):
        super().__init__('perception_control_node')

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)

        # State variables
        self.current_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.target_pose = np.array([1.0, 1.0, 0.0])   # x, y, theta
        self.obstacle_distance = float('inf')
        self.bridge = CvBridge()

        # Control timer
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        # PID controllers
        self.linear_pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
        self.angular_pid = PIDController(kp=2.0, ki=0.2, kd=0.1)

    def scan_callback(self, msg):
        """Process laser scan data for obstacle detection."""
        # Find minimum distance in front of robot
        front_scan = msg.ranges[len(msg.ranges)//2-30:len(msg.ranges)//2+30]
        if front_scan:
            self.obstacle_distance = min([r for r in front_scan if r > 0.1], default=float('inf'))

    def image_callback(self, msg):
        """Process camera image for target detection."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # In real implementation, this would detect targets in the image
        # For now, we'll simulate target detection
        target_in_view = self.detect_target(cv_image)

        if target_in_view:
            # Update target based on visual detection
            self.update_target_from_vision(target_in_view)

    def odom_callback(self, msg):
        """Update current pose from odometry."""
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y

        # Convert quaternion to euler
        quat = msg.pose.pose.orientation
        self.current_pose[2] = self.quaternion_to_yaw(quat)

    def detect_target(self, image):
        """Detect target in camera image (simplified)."""
        # In real implementation, this would use computer vision
        # For simulation, return a dummy detection
        return None

    def control_loop(self):
        """Main control loop with perception feedback."""
        # Calculate desired motion based on target and obstacles
        cmd_vel = Twist()

        # Calculate distance to target
        dx = self.target_pose[0] - self.current_pose[0]
        dy = self.target_pose[1] - self.current_pose[1]
        distance_to_target = np.sqrt(dx**2 + dy**2)

        # Calculate angle to target
        target_angle = np.arctan2(dy, dx)
        angle_error = target_angle - self.current_pose[2]
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))  # Normalize

        # Obstacle avoidance
        safe_distance = 0.5  # meters
        if self.obstacle_distance < safe_distance:
            # Stop or turn away from obstacle
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.5 if angle_error > 0 else -0.5
        else:
            # Navigate toward target
            linear_speed = self.linear_pid.update(distance_to_target, 0)
            angular_speed = self.angular_pid.update(angle_error, 0)

            # Limit speeds
            cmd_vel.linear.x = max(0.0, min(linear_speed, 0.5))
            cmd_vel.angular.z = max(-1.0, min(angular_speed, 1.0))

        # Publish command
        self.cmd_pub.publish(cmd_vel)

    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle."""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return np.arctan2(siny_cosp, cosy_cosp)


class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt=0.05):
        """Update PID controller."""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative


def main(args=None):
    rclpy.init(args=args)
    perception_control_node = PerceptionControlNode()
    rclpy.spin(perception_control_node)
    perception_control_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 7.4.2 Visual Servoing
```python
# visual_servoing.py
import numpy as np
from geometry_msgs.msg import Twist


class VisualServoingNode:
    def __init__(self):
        self.target_pixel = np.array([320, 240])  # Center of image
        self.current_pixel = np.array([320, 240])
        self.pixel_threshold = 10  # pixels

        # Camera parameters (example values)
        self.focal_length = 525  # pixels
        self.image_width = 640
        self.image_height = 480

    def calculate_image_jacobian(self, z):
        """Calculate image Jacobian for visual servoing."""
        # For a point in image space with depth z
        L = np.array([
            [-self.focal_length/z, 0, self.current_pixel[0]/z,
             self.current_pixel[0]*self.current_pixel[1]/self.focal_length,
             -(self.focal_length + self.current_pixel[0]**2/self.focal_length),
             self.current_pixel[1]],
            [0, -self.focal_length/z, self.current_pixel[1]/z,
             (self.focal_length + self.current_pixel[1]**2/self.focal_length),
             -self.current_pixel[0]*self.current_pixel[1]/self.focal_length,
             -self.current_pixel[0]]
        ])
        return L

    def compute_control(self, pixel_error, depth_estimate):
        """Compute control commands for visual servoing."""
        # Image-based visual servoing
        lambda_gain = 0.5  # Learning rate

        # Simple proportional control
        pixel_vel = -lambda_gain * pixel_error

        # Convert to Cartesian velocities (simplified)
        cmd = Twist()
        cmd.linear.x = -pixel_vel[1] * 0.01  # Vertical error -> forward/backward
        cmd.angular.z = -pixel_vel[0] * 0.005  # Horizontal error -> rotation

        return cmd
```

## 7.5 Closed-Loop Control Systems

### 7.5.1 Control Loop Stability
```python
# stability_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def analyze_control_stability(plant_tf, controller_tf):
    """Analyze stability of closed-loop control system."""
    # Closed-loop transfer function: G/(1+GC)
    closed_loop = signal.feedback(plant_tf * controller_tf, 1)

    # Poles of closed-loop system
    poles = closed_loop.poles

    # Check stability (poles in left half plane)
    stable = all(pole.real < 0 for pole in poles)

    return stable, poles


def adaptive_control_gain(error_history, max_gain=2.0, min_gain=0.1):
    """Adjust control gains based on error history."""
    if len(error_history) < 10:
        return 1.0

    recent_error = np.mean(np.abs(error_history[-5:]))
    historical_error = np.mean(np.abs(error_history[:-5]))

    if recent_error > historical_error * 1.5:
        # Error increasing, reduce gain to prevent oscillation
        return max(min_gain, 0.8 * max_gain)
    elif recent_error < historical_error * 0.7:
        # Error decreasing well, can increase gain
        return min(max_gain, 1.2 * max_gain)
    else:
        return 1.0  # Maintain current gain
```

### 7.5.2 Multi-Sensor Integration
```python
class MultiSensorController:
    def __init__(self):
        self.camera_data = None
        self.lidar_data = None
        self.imu_data = None
        self.odom_data = None

        # Confidence weights for each sensor
        self.weights = {
            'camera': 0.3,
            'lidar': 0.4,
            'imu': 0.2,
            'odom': 0.1
        }

        # Sensor validity flags
        self.validity = {sensor: False for sensor in self.weights.keys()}

    def update_camera(self, data):
        """Update camera data."""
        self.camera_data = data
        self.validity['camera'] = True

    def update_lidar(self, data):
        """Update LiDAR data."""
        self.lidar_data = data
        self.validity['lidar'] = True

    def update_imu(self, data):
        """Update IMU data."""
        self.imu_data = data
        self.validity['imu'] = True

    def update_odom(self, data):
        """Update odometry data."""
        self.odom_data = data
        self.validity['odom'] = True

    def get_fused_estimate(self):
        """Get fused state estimate from all valid sensors."""
        estimates = []
        weights = []

        if self.validity['camera']:
            cam_est = self.process_camera_data()
            estimates.append(cam_est)
            weights.append(self.weights['camera'])

        if self.validity['lidar']:
            lidar_est = self.process_lidar_data()
            estimates.append(lidar_est)
            weights.append(self.weights['lidar'])

        if self.validity['imu']:
            imu_est = self.process_imu_data()
            estimates.append(imu_est)
            weights.append(self.weights['imu'])

        if self.validity['odom']:
            odom_est = self.process_odom_data()
            estimates.append(odom_est)
            weights.append(self.weights['odom'])

        if not estimates:
            return None

        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return None

        fused_estimate = sum(w * e for w, e in zip(weights, estimates)) / total_weight
        return fused_estimate

    def process_camera_data(self):
        """Process camera data for state estimate."""
        # In real implementation, this would extract pose from visual features
        return np.array([0.0, 0.0, 0.0])  # x, y, theta

    def process_lidar_data(self):
        """Process LiDAR data for state estimate."""
        # In real implementation, this would do scan matching or landmark detection
        return np.array([0.0, 0.0, 0.0])

    def process_imu_data(self):
        """Process IMU data for state estimate."""
        # Integrate IMU readings to get pose
        return np.array([0.0, 0.0, 0.0])

    def process_odom_data(self):
        """Process odometry data for state estimate."""
        # Use wheel encoders or visual odometry
        return np.array([0.0, 0.0, 0.0])
```

## 7.6 Real-Time Considerations

### 7.6.1 Timing Constraints
- **Perception**: Often the bottleneck, may run at lower frequency
- **Control**: Should run at high frequency for stability
- **Communication**: Consider message rates and network delays

### 7.6.2 Computational Efficiency
```python
# efficient_processing.py
import time
import threading
from queue import Queue


class RealTimePerceptionControl:
    def __init__(self):
        self.perception_queue = Queue(maxsize=2)  # Limit queue size
        self.control_queue = Queue(maxsize=10)
        self.latest_state = None

        # Start processing threads
        self.perception_thread = threading.Thread(target=self.perception_worker)
        self.control_thread = threading.Thread(target=self.control_worker)

        self.perception_thread.start()
        self.control_thread.start()

    def perception_worker(self):
        """Run perception at lower frequency."""
        while True:
            start_time = time.time()

            # Process perception data
            state_estimate = self.process_perception()

            # Non-blocking put (discards old if queue full)
            try:
                self.perception_queue.put_nowait(state_estimate)
            except:
                pass  # Queue full, discard old data

            # Maintain consistent timing
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.1 - elapsed)  # 10 Hz perception
            time.sleep(sleep_time)

    def control_worker(self):
        """Run control at high frequency."""
        while True:
            start_time = time.time()

            # Get latest state estimate
            if not self.perception_queue.empty():
                try:
                    self.latest_state = self.perception_queue.get_nowait()
                except:
                    pass  # Queue empty

            # Run control loop
            if self.latest_state is not None:
                control_command = self.compute_control(self.latest_state)
                self.publish_command(control_command)

            # High frequency control (100 Hz)
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.01 - elapsed)
            time.sleep(sleep_time)

    def process_perception(self):
        """Process sensor data for state estimation."""
        # Simulated perception processing
        return np.random.random(6)  # [x, y, z, roll, pitch, yaw]

    def compute_control(self, state):
        """Compute control command."""
        # Simulated control computation
        return np.random.random(2)  # [linear_vel, angular_vel]

    def publish_command(self, command):
        """Publish control command."""
        # In real system, this would publish to robot
        pass
```

## 7.7 Best Practices

1. **Modular Design**: Separate perception and control components
2. **Timing Analysis**: Ensure control loops meet real-time requirements
3. **Robustness**: Handle sensor failures gracefully
4. **Calibration**: Maintain accurate sensor calibration
5. **Testing**: Validate integration under various conditions
6. **Monitoring**: Track sensor and control performance metrics

## Practical Exercise

### Exercise 7.1: Integrated Navigation System
**Objective**: Create a complete navigation system that integrates perception and control

1. Implement a particle filter for localization using sensor data
2. Create an obstacle avoidance controller using LiDAR data
3. Design a path following controller using visual markers
4. Integrate all components in a closed-loop system
5. Test the system in simulation
6. Analyze stability and performance metrics

**Deliverable**: Complete integrated system with sensor fusion, state estimation, and closed-loop control.

## Summary

Week 7 covered perception-control integration, including state estimation, sensor fusion, and closed-loop control systems. You learned to combine sensor data into coherent state estimates and use these estimates for feedback control. This integration is essential for autonomous robot operation.

[Next: Week 8 - Locomotion Algorithms →](./week8-locomotion-algorithms.md) | [Previous: Week 6 - Isaac ROS Fundamentals ←](./week6-isaac-fundamentals.md)