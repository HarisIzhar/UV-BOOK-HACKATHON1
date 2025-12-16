---
sidebar_position: 12
---

# Week 11: Multimodal Perception

## Learning Objectives

By the end of this week, you will be able to:
- Understand multimodal perception in robotics and AI systems
- Integrate visual, auditory, tactile, and other sensor modalities
- Implement sensor fusion algorithms for robust perception
- Design cross-modal learning systems for enhanced understanding
- Create multimodal perception pipelines for robotics applications

## 11.1 Introduction to Multimodal Perception

Multimodal perception refers to the integration of information from multiple sensory modalities to form a comprehensive understanding of the environment. In robotics, this involves combining data from:

- **Visual sensors**: Cameras, LIDAR, depth sensors
- **Auditory sensors**: Microphones for sound detection
- **Tactile sensors**: Force/torque sensors, tactile arrays
- **Proprioceptive sensors**: Joint encoders, IMU, encoders
- **Other modalities**: Temperature, humidity, chemical sensors

### 11.1.1 Benefits of Multimodal Perception
```
┌─────────────────────────────────────────────────────────────┐
│            Benefits of Multimodal Perception                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Robustness │  │  Richness   │  │  Context    │        │
│  │             │  │             │  │             │        │
│  │ • Fallback  │  │ • More      │  │ • Better    │        │
│  │   sensors   │  │   complete  │  │   scene     │        │
│  │ • Reduced   │  │   scene     │  │   awareness │        │
│  │   ambiguity │  │   models    │  │ • Semantic  │        │
│  └─────────────┘  └─────────────┘  │   reasoning │        │
│                                  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 11.2 Sensor Modalities in Robotics

### 11.2.1 Visual Perception
Visual sensors provide rich spatial and appearance information:

```python
# visual_perception.py
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped


class VisualPerception:
    def __init__(self):
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.feature_detector = cv2.SIFT_create()

    def detect_objects(self, image):
        """Detect and classify objects in image."""
        # Convert to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(image, 'bgr8')

        # Object detection using deep learning model
        # This is a simplified example
        detections = self.run_object_detector(cv_image)

        # Convert to 3D coordinates using depth
        objects_3d = []
        for det in detections:
            if det.has_depth:
                pos_3d = self.project_to_3d(det.bbox_center, det.depth)
                objects_3d.append({
                    'class': det.class_name,
                    'position': pos_3d,
                    'confidence': det.confidence
                })

        return objects_3d

    def run_object_detector(self, image):
        """Run object detection model on image."""
        # In practice, use a trained model like YOLO, SSD, or similar
        # For this example, return mock detections
        height, width = image.shape[:2]
        mock_detections = [
            {
                'class_name': 'cup',
                'bbox_center': (width//2, height//2),
                'confidence': 0.85,
                'has_depth': True,
                'depth': 1.2  # meters
            }
        ]
        return mock_detections

    def project_to_3d(self, pixel, depth):
        """Project 2D pixel coordinates to 3D world coordinates."""
        if self.camera_matrix is None:
            # Use default camera parameters
            fx, fy = 525.0, 525.0
            cx, cy = 320.0, 240.0
        else:
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]

        x = (pixel[0] - cx) * depth / fx
        y = (pixel[1] - cy) * depth / fy
        z = depth

        return np.array([x, y, z])

    def extract_features(self, image):
        """Extract visual features for SLAM or tracking."""
        cv_image = self.bridge.imgmsg_to_cv2(image, 'mono8')
        keypoints, descriptors = self.feature_detector.detectAndCompute(cv_image, None)
        return keypoints, descriptors


class DepthPerception:
    def __init__(self):
        self.depth_scale = 0.001  # Conversion from millimeters to meters

    def process_depth_image(self, depth_image_msg):
        """Process depth image to extract 3D information."""
        depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, 'passthrough')

        # Convert to meters if needed
        if depth_image.dtype == np.uint16:
            depth_image = depth_image.astype(np.float32) * self.depth_scale

        # Create point cloud
        point_cloud = self.depth_to_point_cloud(depth_image)
        return point_cloud

    def depth_to_point_cloud(self, depth_image):
        """Convert depth image to point cloud."""
        height, width = depth_image.shape

        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Camera intrinsic parameters (simplified)
        fx, fy = 525.0, 525.0
        cx, cy = 320.0, 240.0

        # Convert to 3D coordinates
        x = (u - cx) * depth_image / fx
        y = (v - cy) * depth_image / fy
        z = depth_image

        # Stack to create point cloud
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

        # Remove invalid points (where depth is 0)
        valid_points = points[points[:, 2] > 0]

        return valid_points
```

### 11.2.2 Auditory Perception
Auditory sensors enable sound-based environmental understanding:

```python
# auditory_perception.py
import numpy as np
import scipy.signal
import librosa
from sensor_msgs.msg import AudioData


class AuditoryPerception:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.window_size = 1024
        self.hop_length = 512
        self.spectrogram_buffer = []

    def process_audio(self, audio_data):
        """Process audio data to extract features."""
        # Convert audio data to numpy array
        audio_array = np.frombuffer(audio_data.data, dtype=np.int16).astype(np.float32)
        audio_array /= 32768.0  # Normalize to [-1, 1]

        # Extract audio features
        features = {
            'mfcc': self.extract_mfcc(audio_array),
            'spectrogram': self.extract_spectrogram(audio_array),
            'energy': self.calculate_energy(audio_array),
            'zero_crossing_rate': self.calculate_zero_crossing_rate(audio_array),
            'fundamental_frequency': self.estimate_fundamental_frequency(audio_array)
        }

        return features

    def extract_mfcc(self, audio_signal):
        """Extract Mel-frequency cepstral coefficients."""
        mfccs = librosa.feature.mfcc(
            y=audio_signal,
            sr=self.sample_rate,
            n_mfcc=13,
            n_fft=self.window_size,
            hop_length=self.hop_length
        )
        return mfccs

    def extract_spectrogram(self, audio_signal):
        """Extract spectrogram."""
        frequencies, times, Sxx = scipy.signal.spectrogram(
            audio_signal,
            fs=self.sample_rate,
            nperseg=self.window_size,
            noverlap=self.window_size - self.hop_length
        )
        return frequencies, times, Sxx

    def calculate_energy(self, audio_signal):
        """Calculate audio energy."""
        return np.mean(audio_signal ** 2)

    def calculate_zero_crossing_rate(self, audio_signal):
        """Calculate zero crossing rate."""
        return np.mean(np.abs(np.diff(np.sign(audio_signal))) / 2)

    def estimate_fundamental_frequency(self, audio_signal):
        """Estimate fundamental frequency using autocorrelation."""
        # Use autocorrelation to find fundamental frequency
        autocorr = np.correlate(audio_signal, audio_signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # Find peaks in autocorrelation
        peaks = scipy.signal.find_peaks(autocorr)[0]

        if len(peaks) > 1:
            # Fundamental frequency is inverse of the first significant peak
            first_peak = peaks[1]  # Skip the DC peak at 0
            fundamental_freq = self.sample_rate / first_peak
            return fundamental_freq
        else:
            return 0.0

    def classify_sound(self, features):
        """Classify sounds based on extracted features."""
        # In practice, use a trained classifier
        # This is a simplified example
        energy = features['energy']

        if energy > 0.01:
            return 'loud_sound'
        else:
            return 'quiet_sound'

    def locate_sound_source(self, audio_data_left, audio_data_right):
        """Locate sound source using stereo audio."""
        # Calculate interaural time difference (ITD)
        left_signal = np.frombuffer(audio_data_left.data, dtype=np.int16).astype(np.float32)
        right_signal = np.frombuffer(audio_data_right.data, dtype=np.int16).astype(np.float32)

        # Cross-correlation to find time delay
        correlation = np.correlate(left_signal, right_signal, mode='full')
        delay_idx = np.argmax(correlation) - len(left_signal) + 1

        # Convert to time delay
        time_delay = delay_idx / self.sample_rate

        # Estimate angle based on time delay
        # Simplified model: head diameter = 0.2m, speed of sound = 343 m/s
        head_radius = 0.1  # meters
        speed_of_sound = 343.0  # m/s

        max_delay = head_radius / speed_of_sound
        angle = np.arcsin(time_delay / max_delay) if abs(time_delay) <= max_delay else 0.0

        return np.degrees(angle)
```

### 11.2.3 Tactile Perception
Tactile sensors provide information about physical interactions:

```python
# tactile_perception.py
import numpy as np
from sensor_msgs.msg import WrenchStamped, JointState


class TactilePerception:
    def __init__(self):
        self.contact_threshold = 1.0  # Newtons
        self.slip_threshold = 0.5     # Arbitrary units
        self.force_history = []
        self.tactile_array = None

    def process_wrench_data(self, wrench_msg):
        """Process force/torque data from end-effector."""
        force = np.array([
            wrench_msg.wrench.force.x,
            wrench_msg.wrench.force.y,
            wrench_msg.wrench.force.z
        ])

        torque = np.array([
            wrench_msg.wrench.torque.x,
            wrench_msg.wrench.torque.y,
            wrench_msg.wrench.torque.z
        ])

        # Detect contact
        total_force = np.linalg.norm(force)
        contact_detected = total_force > self.contact_threshold

        # Detect slip
        slip_detected = self.detect_slip(force, torque)

        return {
            'force': force,
            'torque': torque,
            'contact': contact_detected,
            'slip': slip_detected,
            'total_force': total_force
        }

    def detect_slip(self, force, torque):
        """Detect slip based on force/torque patterns."""
        # Simple slip detection based on force changes
        self.force_history.append(force.copy())

        if len(self.force_history) > 10:
            self.force_history.pop(0)

        if len(self.force_history) >= 2:
            # Calculate force change rate
            force_change = np.linalg.norm(
                self.force_history[-1] - self.force_history[0]
            ) / len(self.force_history)

            return force_change > self.slip_threshold

        return False

    def process_tactile_array(self, tactile_data):
        """Process data from tactile sensor array."""
        # tactile_data is a grid of pressure sensors
        # For this example, assume it's a 2D array
        pressure_map = np.array(tactile_data)

        # Extract tactile features
        contact_points = self.find_contact_points(pressure_map)
        object_shape = self.estimate_object_shape(pressure_map, contact_points)
        friction_properties = self.estimate_friction(pressure_map)

        return {
            'contact_points': contact_points,
            'object_shape': object_shape,
            'friction': friction_properties,
            'pressure_distribution': pressure_map
        }

    def find_contact_points(self, pressure_map):
        """Find points of contact from pressure map."""
        # Find points above contact threshold
        contact_threshold = 0.1  # Adjust based on sensor sensitivity
        contact_mask = pressure_map > contact_threshold

        # Get coordinates of contact points
        contact_coords = np.column_stack(np.where(contact_mask))

        return contact_coords

    def estimate_object_shape(self, pressure_map, contact_points):
        """Estimate object shape from contact points."""
        if len(contact_points) < 3:
            return "unknown"

        # Simple shape classification
        # In practice, use more sophisticated algorithms
        if len(contact_points) < 10:
            return "small_object"
        elif len(contact_points) < 50:
            return "medium_object"
        else:
            return "large_object"

    def estimate_friction(self, pressure_map):
        """Estimate friction properties from pressure distribution."""
        # Calculate friction coefficient estimate
        # This is a simplified approach
        avg_pressure = np.mean(pressure_map[pressure_map > 0])

        # Map pressure to friction coefficient (simplified model)
        friction_coeff = min(1.0, avg_pressure * 0.5)  # Adjust scaling factor

        return friction_coeff

    def detect_object_slip(self, joint_state, wrench_data):
        """Detect if object is slipping during manipulation."""
        # Check for sudden changes in force that indicate slip
        force_magnitude = np.linalg.norm(wrench_data['force'])
        force_direction_change = self.detect_force_direction_change(wrench_data['force'])

        return force_direction_change or force_magnitude > 20.0  # Threshold

    def detect_force_direction_change(self, current_force):
        """Detect changes in force direction that may indicate slip."""
        if not hasattr(self, 'prev_force'):
            self.prev_force = current_force.copy()
            return False

        # Calculate angle between previous and current force
        cos_angle = np.dot(self.prev_force, current_force) / (
            np.linalg.norm(self.prev_force) * np.linalg.norm(current_force)
        )

        # Update previous force
        self.prev_force = current_force.copy()

        # If angle is changing rapidly, it may indicate slip
        return np.arccos(np.clip(cos_angle, -1.0, 1.0)) > np.pi / 4  # 45 degrees
```

## 11.3 Sensor Fusion Techniques

### 11.3.1 Kalman Filter for Sensor Fusion
```python
# sensor_fusion_kalman.py
import numpy as np
from scipy.linalg import block_diag


class MultiModalKalmanFilter:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim)

        # Process noise (system model uncertainty)
        self.Q = np.eye(state_dim) * 0.1

        # Measurement matrices for different modalities
        self.measurement_matrices = {}
        self.measurement_noise = {}

    def add_modality(self, modality_name, measurement_dim, measurement_matrix, noise_covariance):
        """Add a new sensor modality to the fusion system."""
        self.measurement_matrices[modality_name] = measurement_matrix
        self.measurement_noise[modality_name] = noise_covariance

    def predict(self, control_input=None):
        """Prediction step of Kalman filter."""
        # State transition (simplified as identity for this example)
        # In practice, use actual system dynamics model
        F = np.eye(self.state_dim)

        # Predict state
        if control_input is not None:
            # Apply control input if available
            B = np.eye(self.state_dim) * 0.1  # Control matrix
            self.state = F @ self.state + B @ control_input
        else:
            self.state = F @ self.state

        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.Q

    def update(self, modality_name, measurement):
        """Update step for a specific modality."""
        if modality_name not in self.measurement_matrices:
            raise ValueError(f"Modality {modality_name} not registered")

        H = self.measurement_matrices[modality_name]
        R = self.measurement_noise[modality_name]

        # Innovation
        innovation = measurement - H @ self.state

        # Innovation covariance
        S = H @ self.covariance @ H.T + R

        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ innovation

        # Update covariance
        I_KH = np.eye(self.state_dim) - K @ H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ R @ K.T

    def fuse_modalities(self, modality_measurements):
        """Fuse measurements from multiple modalities."""
        # For each modality, update the state estimate
        for modality_name, measurement in modality_measurements.items():
            if modality_name in self.measurement_matrices:
                self.update(modality_name, measurement)

        return self.state.copy(), self.covariance.copy()


class CrossModalFusionNode:
    def __init__(self):
        # Initialize Kalman filter for state estimation
        self.kf = MultiModalKalmanFilter(state_dim=6)  # [x, y, z, vx, vy, vz]

        # Add different modalities
        # Visual modality: measures position [x, y, z]
        H_visual = np.array([
            [1, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0]   # z
        ])
        R_visual = np.eye(3) * 0.01  # Low noise for visual measurements
        self.kf.add_modality('visual', 3, H_visual, R_visual)

        # Proprioceptive modality: measures position and velocity
        H_proprio = np.array([
            [1, 0, 0, 0, 0, 0],  # x position
            [0, 1, 0, 0, 0, 0],  # y position
            [0, 0, 1, 0, 0, 0],  # z position
            [0, 0, 0, 1, 0, 0],  # x velocity
            [0, 0, 0, 0, 1, 0],  # y velocity
            [0, 0, 0, 0, 0, 1]   # z velocity
        ])
        R_proprio = np.eye(6) * 0.1  # Higher noise for proprioceptive
        self.kf.add_modality('proprioceptive', 6, H_proprio, R_proprio)

        # Tactile modality: measures contact [binary contact detection]
        H_tactile = np.array([[0, 0, 0, 0, 0, 0]])  # Simplified
        R_tactile = np.eye(1) * 0.5
        self.kf.add_modality('tactile', 1, H_tactile, R_tactile)

    def process_multimodal_input(self, visual_data, proprio_data, tactile_data):
        """Process multimodal sensor data and fuse estimates."""
        # Prepare measurements for each modality
        measurements = {}

        if visual_data is not None:
            measurements['visual'] = np.array([
                visual_data.position.x,
                visual_data.position.y,
                visual_data.position.z
            ])

        if proprio_data is not None:
            measurements['proprioceptive'] = np.concatenate([
                proprio_data.position,
                proprio_data.velocity
            ])

        if tactile_data is not None:
            measurements['tactile'] = np.array([tactile_data.contact])

        # Fuse all available measurements
        state_estimate, uncertainty = self.kf.fuse_modalities(measurements)

        return state_estimate, uncertainty
```

### 11.3.2 Particle Filter for Non-Linear Fusion
```python
# particle_filter_fusion.py
class ParticleFilterFusion:
    def __init__(self, num_particles=1000, state_dim=6):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.random.uniform(-5, 5, (num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, control_input, noise_std=0.1):
        """Predict particle motion based on control input."""
        for i in range(self.num_particles):
            # Add process noise to each particle
            process_noise = np.random.normal(0, noise_std, self.state_dim)
            self.particles[i] += control_input + process_noise

    def update(self, measurements, measurement_models):
        """Update particle weights based on measurements."""
        for i in range(self.num_particles):
            total_likelihood = 1.0

            for modality, (measurement, model_func, noise_std) in measurement_models.items():
                # Predict what this measurement should be for this particle
                predicted_measurement = model_func(self.particles[i])

                # Calculate likelihood based on actual measurement
                innovation = measurement - predicted_measurement
                likelihood = np.exp(-0.5 * np.sum(innovation**2) / noise_std**2)

                total_likelihood *= likelihood

            # Update particle weight
            self.weights[i] *= total_likelihood

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

    def get_uncertainty(self):
        """Get uncertainty estimate."""
        mean = self.estimate()
        variance = np.average(
            (self.particles - mean)**2,
            weights=self.weights,
            axis=0
        )
        return np.sqrt(variance)
```

## 11.4 Cross-Modal Learning

### 11.4.1 Cross-Modal Embedding
```python
# cross_modal_embedding.py
import torch
import torch.nn as nn
import numpy as np


class CrossModalEmbedding(nn.Module):
    def __init__(self, visual_dim=512, audio_dim=128, tactile_dim=64, embed_dim=256):
        super(CrossModalEmbedding, self).__init__()

        # Encoders for different modalities
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.tactile_encoder = nn.Sequential(
            nn.Linear(tactile_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.embed_dim = embed_dim

    def forward(self, visual_features, audio_features, tactile_features):
        """Encode features from different modalities to a common space."""
        visual_embed = self.visual_encoder(visual_features)
        audio_embed = self.audio_encoder(audio_features)
        tactile_embed = self.tactile_encoder(tactile_features)

        return visual_embed, audio_embed, tactile_embed

    def compute_similarity(self, embed1, embed2):
        """Compute similarity between embeddings."""
        # Cosine similarity
        norm1 = torch.norm(embed1, dim=-1, keepdim=True)
        norm2 = torch.norm(embed2, dim=-1, keepdim=True)
        similarity = torch.sum(embed1 * embed2, dim=-1) / (norm1 * norm2).squeeze()
        return similarity


class CrossModalAssociation:
    def __init__(self):
        self.embedding_model = CrossModalEmbedding()
        self.association_memory = {}  # Store learned associations

    def learn_association(self, visual_feat, audio_feat, tactile_feat, label):
        """Learn association between modalities."""
        with torch.no_grad():
            vis_emb, aud_emb, tac_emb = self.embedding_model(
                torch.tensor(visual_feat).float().unsqueeze(0),
                torch.tensor(audio_feat).float().unsqueeze(0),
                torch.tensor(tactile_feat).float().unsqueeze(0)
            )

        # Store the association
        self.association_memory[label] = {
            'visual': vis_emb.squeeze().numpy(),
            'audio': aud_emb.squeeze().numpy(),
            'tactile': tac_emb.squeeze().numpy()
        }

    def recognize_multimodal_pattern(self, visual_input, audio_input, tactile_input):
        """Recognize pattern across modalities."""
        with torch.no_grad():
            vis_emb, aud_emb, tac_emb = self.embedding_model(
                torch.tensor(visual_input).float().unsqueeze(0),
                torch.tensor(audio_input).float().unsqueeze(0),
                torch.tensor(tactile_input).float().unsqueeze(0)
            )

        # Find best matching association
        best_match = None
        best_score = -float('inf')

        for label, embeddings in self.association_memory.items():
            # Compute combined similarity score
            vis_sim = np.dot(vis_emb.squeeze().numpy(), embeddings['visual'])
            aud_sim = np.dot(aud_emb.squeeze().numpy(), embeddings['audio'])
            tac_sim = np.dot(tac_emb.squeeze().numpy(), embeddings['tactile'])

            combined_score = (vis_sim + aud_sim + tac_sim) / 3.0

            if combined_score > best_score:
                best_score = combined_score
                best_match = label

        return best_match, best_score
```

### 11.4.2 Multimodal Object Recognition
```python
# multimodal_recognition.py
class MultimodalObjectRecognizer:
    def __init__(self):
        self.object_database = {}
        self.visual_classifier = self.train_visual_classifier()
        self.audio_classifier = self.train_audio_classifier()
        self.tactile_classifier = self.train_tactile_classifier()

    def train_visual_classifier(self):
        """Train visual classifier (simplified)."""
        # In practice, use a CNN or similar
        return lambda x: {'object_type': 'unknown', 'confidence': 0.0}

    def train_audio_classifier(self):
        """Train audio classifier (simplified)."""
        # In practice, use audio-specific models
        return lambda x: {'sound_type': 'unknown', 'confidence': 0.0}

    def train_tactile_classifier(self):
        """Train tactile classifier (simplified)."""
        # In practice, use tactile-specific models
        return lambda x: {'texture_type': 'unknown', 'confidence': 0.0}

    def recognize_object(self, visual_data, audio_data, tactile_data):
        """Recognize object using multiple modalities."""
        # Get individual modality classifications
        visual_result = self.visual_classifier(visual_data)
        audio_result = self.audio_classifier(audio_data)
        tactile_result = self.tactile_classifier(tactile_data)

        # Fuse results based on confidence
        fused_result = self.fuse_classifications(
            visual_result, audio_result, tactile_result)

        return fused_result

    def fuse_classifications(self, visual_result, audio_result, tactile_result):
        """Fuse classification results from different modalities."""
        # Weighted combination based on confidence
        vis_conf = visual_result.get('confidence', 0.0)
        aud_conf = audio_result.get('confidence', 0.0)
        tac_conf = tactile_result.get('confidence', 0.0)

        total_conf = vis_conf + aud_conf + tac_conf

        if total_conf == 0:
            return {'object': 'unknown', 'confidence': 0.0}

        # Combine labels based on weighted votes
        labels = [visual_result.get('object_type', 'unknown'),
                 audio_result.get('sound_type', 'unknown'),
                 tactile_result.get('texture_type', 'unknown')]

        # For simplicity, use the label with highest confidence
        confidences = [vis_conf, aud_conf, tac_conf]
        best_idx = np.argmax(confidences)

        return {
            'object': labels[best_idx],
            'confidence': confidences[best_idx] / total_conf
        }

    def learn_new_object(self, object_name, visual_data, audio_data, tactile_data):
        """Learn a new object with its multimodal signature."""
        self.object_database[object_name] = {
            'visual_signature': self.extract_visual_signature(visual_data),
            'audio_signature': self.extract_audio_signature(audio_data),
            'tactile_signature': self.extract_tactile_signature(tactile_data)
        }

    def extract_visual_signature(self, visual_data):
        """Extract visual signature from visual data."""
        # Simplified: use average color and shape features
        return {
            'color_histogram': np.random.random(50),  # Placeholder
            'shape_descriptor': np.random.random(10)   # Placeholder
        }

    def extract_audio_signature(self, audio_data):
        """Extract audio signature from audio data."""
        # Simplified: use MFCC features
        return {
            'mfcc_features': np.random.random(13),  # Placeholder
            'spectral_features': np.random.random(10)  # Placeholder
        }

    def extract_tactile_signature(self, tactile_data):
        """Extract tactile signature from tactile data."""
        # Simplified: use pressure distribution features
        return {
            'pressure_map': np.random.random((8, 8)),  # Placeholder
            'texture_features': np.random.random(5)   # Placeholder
        }
```

## 11.5 Multimodal Perception Pipelines

### 11.5.1 Real-time Multimodal Processing
```python
# multimodal_pipeline.py
import threading
import queue
import time
from collections import deque


class MultimodalPipeline:
    def __init__(self, buffer_size=10):
        # Queues for different modalities
        self.visual_queue = queue.Queue(maxsize=buffer_size)
        self.audio_queue = queue.Queue(maxsize=buffer_size)
        self.tactile_queue = queue.Queue(maxsize=buffer_size)

        # Time synchronization buffer
        self.sync_buffer = {
            'visual': deque(maxlen=buffer_size),
            'audio': deque(maxlen=buffer_size),
            'tactile': deque(maxlen=buffer_size)
        }

        # Processing threads
        self.fusion_thread = threading.Thread(target=self.fusion_worker)
        self.fusion_active = False

        # Fusion result
        self.fusion_result = None

    def start_processing(self):
        """Start multimodal processing."""
        self.fusion_active = True
        self.fusion_thread.start()

    def stop_processing(self):
        """Stop multimodal processing."""
        self.fusion_active = False
        self.fusion_thread.join()

    def add_visual_data(self, data, timestamp):
        """Add visual data to pipeline."""
        try:
            self.visual_queue.put_nowait((data, timestamp))
        except queue.Full:
            # Discard oldest if buffer full
            try:
                self.visual_queue.get_nowait()
                self.visual_queue.put_nowait((data, timestamp))
            except:
                pass

    def add_audio_data(self, data, timestamp):
        """Add audio data to pipeline."""
        try:
            self.audio_queue.put_nowait((data, timestamp))
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait((data, timestamp))
            except:
                pass

    def add_tactile_data(self, data, timestamp):
        """Add tactile data to pipeline."""
        try:
            self.tactile_queue.put_nowait((data, timestamp))
        except queue.Full:
            try:
                self.tactile_queue.get_nowait()
                self.tactile_queue.put_nowait((data, timestamp))
            except:
                pass

    def fusion_worker(self):
        """Background worker for data fusion."""
        while self.fusion_active:
            # Synchronize data from different modalities
            synchronized_data = self.synchronize_modalities()

            if synchronized_data is not None:
                # Perform fusion
                result = self.fuse_data(synchronized_data)
                self.fusion_result = result

            time.sleep(0.01)  # 100 Hz processing

    def synchronize_modalities(self):
        """Synchronize data from different modalities based on timestamps."""
        # Get latest data from each modality
        visual_data = None
        audio_data = None
        tactile_data = None

        # Try to get the most recent visual data
        while not self.visual_queue.empty():
            visual_data = self.visual_queue.get_nowait()

        # Try to get the most recent audio data
        while not self.audio_queue.empty():
            audio_data = self.audio_queue.get_nowait()

        # Try to get the most recent tactile data
        while not self.tactile_queue.empty():
            tactile_data = self.tactile_queue.get_nowait()

        if visual_data and audio_data and tactile_data:
            return {
                'visual': visual_data[0],
                'audio': audio_data[0],
                'tactile': tactile_data[0],
                'timestamp': max(visual_data[1], audio_data[1], tactile_data[1])
            }

        return None

    def fuse_data(self, synchronized_data):
        """Fuse synchronized multimodal data."""
        # In practice, implement sophisticated fusion algorithm
        # This is a simplified example

        visual_result = self.process_visual_data(synchronized_data['visual'])
        audio_result = self.process_audio_data(synchronized_data['audio'])
        tactile_result = self.process_tactile_data(synchronized_data['tactile'])

        # Combine results
        fused_result = {
            'environment_state': self.combine_modalities(
                visual_result, audio_result, tactile_result),
            'timestamp': synchronized_data['timestamp']
        }

        return fused_result

    def process_visual_data(self, data):
        """Process visual data."""
        # Placeholder for visual processing
        return {'objects_detected': 0, 'scene_description': 'unknown'}

    def process_audio_data(self, data):
        """Process audio data."""
        # Placeholder for audio processing
        return {'sounds_detected': 0, 'sound_classification': 'unknown'}

    def process_tactile_data(self, data):
        """Process tactile data."""
        # Placeholder for tactile processing
        return {'contacts_detected': 0, 'surface_properties': 'unknown'}

    def combine_modalities(self, visual_result, audio_result, tactile_result):
        """Combine results from different modalities."""
        # Simple combination
        return {
            'visual': visual_result,
            'audio': audio_result,
            'tactile': tactile_result,
            'fused_interpretation': self.generate_interpretation(
                visual_result, audio_result, tactile_result)
        }

    def generate_interpretation(self, visual, audio, tactile):
        """Generate unified interpretation from multimodal data."""
        # Example interpretation logic
        if visual['objects_detected'] > 0 and tactile['contacts_detected'] > 0:
            return "object_interaction"
        elif audio['sounds_detected'] > 0 and visual['objects_detected'] > 0:
            return "object_with_sound"
        else:
            return "environment_sensing"
```

### 11.5.2 Adaptive Multimodal Perception
```python
# adaptive_multimodal.py
class AdaptiveMultimodalPerception:
    def __init__(self):
        self.modality_weights = {
            'visual': 1.0,
            'audio': 1.0,
            'tactile': 1.0
        }
        self.performance_history = {
            'visual': [],
            'audio': [],
            'tactile': []
        }
        self.fusion_strategy = 'confidence_weighted'

    def adapt_modality_weights(self, task_context, performance_feedback):
        """Adapt modality weights based on task and performance."""
        for modality in self.modality_weights:
            if modality in performance_feedback:
                # Update performance history
                self.performance_history[modality].append(
                    performance_feedback[modality])

                # Keep only recent history (last 10 measurements)
                if len(self.performance_history[modality]) > 10:
                    self.performance_history[modality] = \
                        self.performance_history[modality][-10:]

                # Adjust weight based on recent performance
                recent_performance = np.mean(self.performance_history[modality])

                # Increase weight if performance is good, decrease if poor
                if recent_performance > 0.7:  # Threshold for "good" performance
                    self.modality_weights[modality] *= 1.1
                elif recent_performance < 0.3:  # Threshold for "poor" performance
                    self.modality_weights[modality] *= 0.9

                # Clamp weights to reasonable range
                self.modality_weights[modality] = np.clip(
                    self.modality_weights[modality], 0.1, 3.0)

    def select_fusion_strategy(self, environment_context):
        """Select optimal fusion strategy based on context."""
        if environment_context.get('low_light', False):
            reduce visual weight
            self.modality_weights['visual'] *= 0.5
            self.modality_weights['audio'] *= 1.2
        elif environment_context.get('noisy', False):
            # Reduce audio weight in noisy environments
            self.modality_weights['audio'] *= 0.5
            self.modality_weights['visual'] *= 1.2
        elif environment_context.get('object_interaction', False):
            # Increase tactile weight during manipulation
            self.modality_weights['tactile'] *= 1.5

    def fuse_with_adaptation(self, visual_data, audio_data, tactile_data, context):
        """Fuse modalities with adaptive weights."""
        # Select appropriate strategy based on context
        self.select_fusion_strategy(context)

        # Apply adaptive weights
        weighted_visual = visual_data * self.modality_weights['visual']
        weighted_audio = audio_data * self.modality_weights['audio']
        weighted_tactile = tactile_data * self.modality_weights['tactile']

        # Perform weighted fusion
        total_weight = sum(self.modality_weights.values())
        fused_result = (weighted_visual + weighted_audio + weighted_tactile) / total_weight

        return fused_result
```

## 11.6 Implementation Example: Multimodal Scene Understanding

```python
# multimodal_scene_understanding.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, AudioData, JointState, WrenchStamped
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np


class MultimodalSceneUnderstandingNode(Node):
    def __init__(self):
        super().__init__('multimodal_scene_node')

        # Publishers
        self.scene_description_pub = self.create_publisher(
            String, '/scene_description', 10)
        self.object_pose_pub = self.create_publisher(
            PoseStamped, '/detected_object', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10)
        self.audio_sub = self.create_publisher(  # Note: this should be a subscriber
            AudioData, '/microphone/audio', self.audio_callback, 10)
        self.wrench_sub = self.create_subscription(
            WrenchStamped, '/wrench', self.wrench_callback, 10)

        # Perception components
        self.visual_perceptor = VisualPerception()
        self.auditory_perceptor = AuditoryPerception()
        self.tactile_perceptor = TactilePerception()
        self.fusion_engine = MultiModalKalmanFilter(state_dim=6)

        # Internal state
        self.bridge = CvBridge()
        self.current_scene = {}
        self.scene_history = []

        # Processing timer
        self.processing_timer = self.create_timer(0.1, self.process_scene)

    def image_callback(self, msg):
        """Process visual data."""
        try:
            # Detect objects in image
            objects_3d = self.visual_perceptor.detect_objects(msg)
            self.current_scene['visual_objects'] = objects_3d

            # Extract visual features for tracking
            keypoints, descriptors = self.visual_perceptor.extract_features(msg)
            self.current_scene['visual_features'] = {
                'keypoints': keypoints,
                'descriptors': descriptors
            }
        except Exception as e:
            self.get_logger().error(f"Visual processing error: {e}")

    def audio_callback(self, msg):
        """Process audio data."""
        try:
            # Extract audio features
            audio_features = self.auditory_perceptor.process_audio(msg)
            self.current_scene['audio_features'] = audio_features

            # Classify sounds
            sound_class = self.auditory_perceptor.classify_sound(audio_features)
            self.current_scene['detected_sound'] = sound_class
        except Exception as e:
            self.get_logger().error(f"Audio processing error: {e}")

    def wrench_callback(self, msg):
        """Process tactile/force data."""
        try:
            # Process force/torque data
            tactile_data = self.tactile_perceptor.process_wrench_data(msg)
            self.current_scene['tactile_data'] = tactile_data

            # Detect contact and slip
            if tactile_data['contact']:
                self.current_scene['contact_detected'] = True
            if tactile_data['slip']:
                self.current_scene['slip_detected'] = True
        except Exception as e:
            self.get_logger().error(f"Tactile processing error: {e}")

    def process_scene(self):
        """Process and fuse multimodal scene data."""
        if not self.current_scene:
            return

        # Perform multimodal fusion
        fused_interpretation = self.fuse_scene_data()

        # Generate scene description
        scene_description = self.generate_scene_description(fused_interpretation)

        # Publish results
        description_msg = String()
        description_msg.data = scene_description
        self.scene_description_pub.publish(description_msg)

        # Update scene history
        self.scene_history.append({
            'timestamp': self.get_clock().now().to_msg(),
            'interpretation': fused_interpretation,
            'description': scene_description
        })

        # Keep only recent history
        if len(self.scene_history) > 100:
            self.scene_history = self.scene_history[-50:]

    def fuse_scene_data(self):
        """Fuse data from all modalities."""
        fusion_result = {
            'objects': [],
            'sounds': [],
            'interactions': [],
            'environment_state': 'unknown'
        }

        # Process visual objects
        if 'visual_objects' in self.current_scene:
            fusion_result['objects'].extend(self.current_scene['visual_objects'])

        # Process audio
        if 'detected_sound' in self.current_scene:
            fusion_result['sounds'].append(self.current_scene['detected_sound'])

        # Process tactile interactions
        if 'contact_detected' in self.current_scene:
            fusion_result['interactions'].append('contact')
        if 'slip_detected' in self.current_scene:
            fusion_result['interactions'].append('slip')

        # Determine environment state based on modalities
        if (self.current_scene.get('contact_detected', False) and
            len(fusion_result['objects']) > 0):
            fusion_result['environment_state'] = 'object_interaction'
        elif len(fusion_result['sounds']) > 0:
            fusion_result['environment_state'] = 'active_environment'
        else:
            fusion_result['environment_state'] = 'passive_sensing'

        return fusion_result

    def generate_scene_description(self, fused_data):
        """Generate natural language description of the scene."""
        description_parts = []

        # Describe objects
        if fused_data['objects']:
            obj_count = len(fused_data['objects'])
            description_parts.append(f"{obj_count} object{'s' if obj_count > 1 else ''} detected")

        # Describe sounds
        if fused_data['sounds']:
            description_parts.append(f"Sounds: {', '.join(fused_data['sounds'])}")

        # Describe interactions
        if fused_data['interactions']:
            description_parts.append(f"Interactions: {', '.join(fused_data['interactions'])}")

        # Describe overall state
        description_parts.append(f"Environment: {fused_data['environment_state']}")

        return "; ".join(description_parts)

    def get_current_scene_interpretation(self):
        """Get the current interpreted scene state."""
        if self.scene_history:
            return self.scene_history[-1]['interpretation']
        return None


def main(args=None):
    rclpy.init(args=args)
    scene_node = MultimodalSceneUnderstandingNode()

    # Example: Add a simple test
    def test_scene_understanding():
        """Test the scene understanding system."""
        scene_node.get_logger().info("Testing multimodal scene understanding...")

        # Simulate some scenario
        scene_node.current_scene = {
            'visual_objects': [{'class': 'cup', 'position': [0.5, 0.2, 1.0], 'confidence': 0.85}],
            'detected_sound': 'object_tap',
            'contact_detected': True
        }

        # Process the scene
        scene_node.process_scene()

        # Get interpretation
        interpretation = scene_node.get_current_scene_interpretation()
        if interpretation:
            scene_node.get_logger().info(f"Scene interpretation: {interpretation['environment_state']}")

    # Run test after a short delay
    scene_node.create_timer(2.0, test_scene_understanding)

    rclpy.spin(scene_node)
    scene_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## 11.7 Advanced Topics

### 11.7.1 Attention Mechanisms in Multimodal Perception
Using attention to focus on relevant sensory inputs at appropriate times.

### 11.7.2 Learning from Multimodal Correlations
Discovering relationships between different sensory modalities.

## 11.8 Best Practices

1. **Calibration**: Ensure proper calibration of all sensors
2. **Synchronization**: Align data from different modalities in time
3. **Robustness**: Handle missing or unreliable sensor data
4. **Efficiency**: Optimize for real-time performance
5. **Validation**: Test perception system in varied conditions

## Practical Exercise

### Exercise 11.1: Multimodal Object Recognition System
**Objective**: Create a system that recognizes objects using visual, auditory, and tactile information

1. Implement individual perception modules for each modality
2. Design a fusion algorithm that combines information from all modalities
3. Create a learning mechanism to improve recognition over time
4. Test the system with different objects and environmental conditions
5. Evaluate the improvement from multimodal fusion over single modality
6. Analyze the contribution of each modality to overall performance

**Deliverable**: Complete multimodal object recognition system with performance analysis.

## Summary

Week 11 covered multimodal perception in robotics, including sensor fusion, cross-modal learning, and integrated perception systems. You learned to combine information from visual, auditory, tactile, and other sensors to create robust environmental understanding. This capability is essential for robots operating in complex, dynamic environments.

[Next: Week 12 - Language Grounding & Decision Making →](./week12-language-grounding.md) | [Previous: Week 10 - Locomotion & Manipulation Integration ←](./week10-locomotion-manipulation-integration.md)