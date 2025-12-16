---
sidebar_position: 14
---

# Week 13: System Integration & Deployment

## Learning Objectives

By the end of this week, you will be able to:
- Integrate all components of a Physical AI system into a cohesive whole
- Design deployment architectures for robotic systems
- Implement system monitoring and maintenance procedures
- Validate integrated system performance and safety
- Create deployment documentation and operational procedures

## 13.1 Introduction to System Integration

System integration in Physical AI and robotics involves combining all the specialized subsystems (perception, planning, control, manipulation, locomotion) into a unified, operational system. This integration presents unique challenges:

- **Real-time constraints**: All subsystems must operate within timing requirements
- **Resource sharing**: Multiple processes compete for computational resources
- **Communication protocols**: Different subsystems use various communication patterns
- **Error propagation**: Failures in one subsystem can affect the entire system
- **Safety requirements**: Integrated safety across all components

### 13.1.1 Integration Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                 Physical AI System                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │              ROS 2 Middleware Layer                 │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │   │
│  │  │ Perception  │ │ Planning    │ │ Control     │   │   │
│  │  │   Nodes     │ │   Nodes     │ │   Nodes     │   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                │                   │           │
│           ▼                ▼                   ▼           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  │   Hardware      │ │   Simulation    │ │   Monitoring    │
│  │   Interface     │ │   Interface     │ │   & Logging     │
│  │   Layer         │ │   Layer         │ │   Layer         │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘
│           │                │                   │           │
│           └────────────────┼───────────────────┘           │
│                            ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Robot Hardware Platform                │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │   │
│  │  │  Sensors    │ │   Motors    │ │  Processors │   │   │
│  │  │ • Cameras   │ │ • Actuators │ │ • GPU       │   │   │
│  │  │ • LiDAR     │ │ • Drives    │ │ • CPU       │   │   │
│  │  │ • IMU       │ │ • Grippers  │ │ • FPGA      │   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 13.2 Integration Strategies

### 13.2.1 Component-Based Integration
```python
# system_integration.py
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import threading
import queue
import time
from typing import Dict, List, Any, Optional


class ComponentManager:
    """Manages integration of system components."""
    def __init__(self):
        self.components = {}
        self.connections = []
        self.health_status = {}
        self.resource_usage = {}
        self.integration_state = 'uninitialized'

    def register_component(self, name: str, component: Any):
        """Register a system component."""
        self.components[name] = {
            'instance': component,
            'status': 'registered',
            'dependencies': [],
            'resources': {'cpu': 0, 'memory': 0, 'bandwidth': 0}
        }
        self.health_status[name] = 'unknown'

    def define_dependency(self, dependent: str, dependency: str):
        """Define dependency relationship between components."""
        if dependent in self.components:
            self.components[dependent]['dependencies'].append(dependency)

    def connect_components(self, source: str, target: str, topic: str):
        """Define communication connection between components."""
        self.connections.append({
            'source': source,
            'target': target,
            'topic': topic,
            'status': 'pending'
        })

    def initialize_components(self):
        """Initialize all registered components in dependency order."""
        # Sort components by dependency
        init_order = self._topological_sort()

        for component_name in init_order:
            if component_name in self.components:
                try:
                    component = self.components[component_name]['instance']
                    if hasattr(component, 'initialize'):
                        component.initialize()
                    self.components[component_name]['status'] = 'initialized'
                    self.health_status[component_name] = 'healthy'
                except Exception as e:
                    self.components[component_name]['status'] = 'error'
                    self.health_status[component_name] = f'error: {str(e)}'
                    return False

        self.integration_state = 'initialized'
        return True

    def _topological_sort(self) -> List[str]:
        """Sort components in dependency order."""
        # Simple topological sort implementation
        visited = set()
        result = []

        def visit(node):
            if node not in visited:
                visited.add(node)
                if node in self.components:
                    for dep in self.components[node]['dependencies']:
                        visit(dep)
                result.append(node)

        for component_name in self.components:
            visit(component_name)

        return result


class IntegrationNode(Node):
    """ROS 2 node that manages system integration."""
    def __init__(self):
        super().__init__('integration_node')

        # Initialize component manager
        self.component_manager = ComponentManager()

        # Publishers and subscribers
        self.status_pub = self.create_publisher(String, '/system_status', 10)
        self.health_pub = self.create_publisher(String, '/system_health', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)

        # System state
        self.system_state = 'idle'
        self.emergency_stop = False

        # Integration timer
        self.integration_timer = self.create_timer(1.0, self.integration_callback)

    def register_component(self, name: str, component: Any):
        """Register a component with the integration system."""
        self.component_manager.register_component(name, component)
        self.get_logger().info(f"Registered component: {name}")

    def integration_callback(self):
        """Main integration callback - monitor system health and status."""
        if self.component_manager.integration_state == 'initialized':
            # Monitor component health
            all_healthy = True
            for comp_name, status in self.component_manager.health_status.items():
                if 'error' in status:
                    all_healthy = False
                    self.get_logger().error(f"Component {comp_name} has error: {status}")

            # Publish system status
            status_msg = String()
            status_msg.data = f"System state: {self.system_state}, All healthy: {all_healthy}"
            self.status_pub.publish(status_msg)

            # Publish health status
            health_msg = String()
            health_msg.data = str(self.component_manager.health_status)
            self.health_pub.publish(health_msg)

            # Check for emergency conditions
            if self.check_emergency_conditions():
                self.trigger_emergency_stop()

    def check_emergency_conditions(self) -> bool:
        """Check for conditions requiring emergency stop."""
        # Check various emergency conditions
        if self.emergency_stop:
            return True

        # Check for component failures
        for status in self.component_manager.health_status.values():
            if 'critical' in status:
                return True

        # Check for safety violations (e.g., collision detection)
        # This would interface with collision detection systems
        # For now, return False
        return False

    def trigger_emergency_stop(self):
        """Trigger emergency stop and halt all operations."""
        if not self.emergency_stop:
            self.emergency_stop = True
            self.system_state = 'emergency_stop'

            # Publish emergency stop command
            stop_msg = Bool()
            stop_msg.data = True
            self.emergency_stop_pub.publish(stop_msg)

            self.get_logger().fatal("EMERGENCY STOP TRIGGERED")
```

### 13.2.2 Microservice Architecture for Robotics
```python
# microservice_integration.py
import asyncio
import aiohttp
from typing import Dict, Any, Callable
import json


class MicroserviceIntegration:
    """Manages integration using microservice architecture."""
    def __init__(self):
        self.services = {}
        self.service_endpoints = {}
        self.message_broker = None
        self.service_mesh = {}

    async def register_service(self, name: str, endpoint: str, health_check: str = None):
        """Register a microservice with the integration system."""
        self.services[name] = {
            'endpoint': endpoint,
            'health_check': health_check,
            'status': 'unknown',
            'dependencies': []
        }
        self.service_endpoints[name] = endpoint

        # Perform initial health check
        if health_check:
            await self.check_service_health(name)

    async def check_service_health(self, service_name: str):
        """Check the health of a registered service."""
        if service_name not in self.services:
            return False

        endpoint = self.services[service_name]['health_check']
        if not endpoint:
            return True  # Assume healthy if no health check defined

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint) as response:
                    if response.status == 200:
                        self.services[service_name]['status'] = 'healthy'
                        return True
                    else:
                        self.services[service_name]['status'] = 'unhealthy'
                        return False
        except Exception as e:
            self.services[service_name]['status'] = f'error: {str(e)}'
            return False

    async def call_service(self, service_name: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call a registered microservice."""
        if service_name not in self.service_endpoints:
            raise ValueError(f"Service {service_name} not registered")

        endpoint = self.service_endpoints[service_name]
        url = f"{endpoint}/{method}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    result = await response.json()
                    return result
        except Exception as e:
            raise Exception(f"Service call failed: {str(e)}")

    def define_service_dependency(self, service: str, depends_on: str):
        """Define dependency relationship between services."""
        if service in self.services:
            self.services[service]['dependencies'].append(depends_on)

    async def initialize_services(self):
        """Initialize all services in dependency order."""
        # Check all services are healthy
        all_healthy = True
        for service_name in self.services:
            is_healthy = await self.check_service_health(service_name)
            if not is_healthy:
                all_healthy = False
                self.get_logger().error(f"Service {service_name} is not healthy")

        return all_healthy

    def get_logger(self):
        """Get logger instance (placeholder)."""
        class Logger:
            def error(self, msg):
                print(f"ERROR: {msg}")
        return Logger()


class ServiceOrchestrator:
    """Orchestrates communication between microservices."""
    def __init__(self, integration: MicroserviceIntegration):
        self.integration = integration
        self.workflow_definitions = {}
        self.active_workflows = {}

    async def define_workflow(self, name: str, steps: List[Dict[str, Any]]):
        """Define a workflow of service calls."""
        self.workflow_definitions[name] = steps

    async def execute_workflow(self, workflow_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a defined workflow."""
        if workflow_name not in self.workflow_definitions:
            raise ValueError(f"Workflow {workflow_name} not defined")

        workflow = self.workflow_definitions[workflow_name]
        context = input_data.copy()

        for step in workflow:
            service_name = step['service']
            method = step['method']
            params = step.get('params', {})

            # Merge context with step parameters
            step_params = {**params, **context}

            try:
                result = await self.integration.call_service(service_name, method, step_params)
                # Update context with results
                context.update(result)
            except Exception as e:
                # Handle step failure
                error_result = {
                    'error': str(e),
                    'step': step,
                    'context': context
                }
                return error_result

        return context
```

## 13.3 Deployment Architecture

### 13.3.1 Hardware Abstraction Layer
```python
# hardware_abstraction.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import time


class HardwareInterface(ABC):
    """Abstract base class for hardware interfaces."""

    @abstractmethod
    def initialize(self):
        """Initialize the hardware interface."""
        pass

    @abstractmethod
    def read_sensors(self) -> Dict[str, Any]:
        """Read current sensor values."""
        pass

    @abstractmethod
    def write_actuators(self, commands: Dict[str, Any]):
        """Send commands to actuators."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current hardware status."""
        pass


class RobotHardwareInterface(HardwareInterface):
    """Concrete implementation for robot hardware."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
        self.last_read_time = 0
        self.hardware_status = {
            'motors': {},
            'sensors': {},
            'power': {},
            'temperature': {}
        }

    def initialize(self):
        """Initialize robot hardware."""
        # Initialize motors
        for motor_config in self.config.get('motors', []):
            motor_id = motor_config['id']
            self.hardware_status['motors'][motor_id] = {
                'status': 'initializing',
                'position': 0.0,
                'velocity': 0.0,
                'effort': 0.0
            }

        # Initialize sensors
        for sensor_config in self.config.get('sensors', []):
            sensor_id = sensor_config['id']
            self.hardware_status['sensors'][sensor_id] = {
                'status': 'initializing',
                'value': 0.0
            }

        self.is_initialized = True
        self.last_read_time = time.time()

    def read_sensors(self) -> Dict[str, Any]:
        """Read current sensor values from hardware."""
        if not self.is_initialized:
            raise RuntimeError("Hardware not initialized")

        # Simulate reading from actual hardware
        # In practice, this would interface with real hardware drivers
        sensor_data = {}

        for sensor_id, status in self.hardware_status['sensors'].items():
            # Simulate sensor reading
            if 'camera' in sensor_id:
                sensor_data[sensor_id] = {'image': 'simulated_image_data', 'timestamp': time.time()}
            elif 'lidar' in sensor_id:
                sensor_data[sensor_id] = {'ranges': [1.0] * 360, 'timestamp': time.time()}
            elif 'imu' in sensor_id:
                sensor_data[sensor_id] = {
                    'orientation': [0.0, 0.0, 0.0, 1.0],
                    'angular_velocity': [0.0, 0.0, 0.0],
                    'linear_acceleration': [0.0, 0.0, 9.81],
                    'timestamp': time.time()
                }
            else:
                sensor_data[sensor_id] = {'value': 0.0, 'timestamp': time.time()}

        self.last_read_time = time.time()
        return sensor_data

    def write_actuators(self, commands: Dict[str, Any]):
        """Send commands to actuators."""
        if not self.is_initialized:
            raise RuntimeError("Hardware not initialized")

        # Process motor commands
        for motor_id, command in commands.get('motors', {}).items():
            if motor_id in self.hardware_status['motors']:
                self.hardware_status['motors'][motor_id].update({
                    'command': command,
                    'timestamp': time.time()
                })

        # Process other actuator commands
        for actuator_id, command in commands.get('actuators', {}).items():
            self.hardware_status['motors'][actuator_id] = {
                'command': command,
                'timestamp': time.time()
            }

    def get_status(self) -> Dict[str, Any]:
        """Get current hardware status."""
        return {
            'initialized': self.is_initialized,
            'last_read_time': self.last_read_time,
            'motors': self.hardware_status['motors'],
            'sensors': self.hardware_status['sensors'],
            'system_time': time.time()
        }


class SimulationHardwareInterface(HardwareInterface):
    """Hardware interface for simulation environment."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
        self.simulation_time = 0.0
        self.simulation_state = {}

    def initialize(self):
        """Initialize simulation interface."""
        self.is_initialized = True
        self.simulation_time = 0.0

    def read_sensors(self) -> Dict[str, Any]:
        """Read sensor values from simulation."""
        if not self.is_initialized:
            raise RuntimeError("Simulation not initialized")

        # Generate simulated sensor data
        sensor_data = {}
        for sensor_config in self.config.get('sensors', []):
            sensor_id = sensor_config['id']
            sensor_type = sensor_config['type']

            if sensor_type == 'camera':
                sensor_data[sensor_id] = {
                    'image': 'simulated_camera_data',
                    'timestamp': self.simulation_time
                }
            elif sensor_type == 'lidar':
                sensor_data[sensor_id] = {
                    'ranges': [2.0 + 0.1 * i for i in range(360)],  # Simulated ranges
                    'timestamp': self.simulation_time
                }
            elif sensor_type == 'imu':
                sensor_data[sensor_id] = {
                    'orientation': [0.0, 0.0, 0.0, 1.0],
                    'angular_velocity': [0.0, 0.0, 0.0],
                    'linear_acceleration': [0.0, 0.0, 9.81],
                    'timestamp': self.simulation_time
                }

        self.simulation_time += 0.01  # Increment simulation time
        return sensor_data

    def write_actuators(self, commands: Dict[str, Any]):
        """Send commands to simulated actuators."""
        if not self.is_initialized:
            raise RuntimeError("Simulation not initialized")

        # Update simulation state based on commands
        self.simulation_state.update(commands)

    def get_status(self) -> Dict[str, Any]:
        """Get simulation status."""
        return {
            'initialized': self.is_initialized,
            'simulation_time': self.simulation_time,
            'simulation_state': self.simulation_state
        }
```

### 13.3.2 Containerized Deployment
```python
# containerized_deployment.py
import docker
import yaml
from typing import Dict, List, Any
import subprocess
import os


class ContainerizedDeployment:
    """Manages deployment using containerization."""
    def __init__(self, config_file: str = None):
        self.client = docker.from_env()
        self.config = self.load_config(config_file) if config_file else {}
        self.containers = {}
        self.networks = {}

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load deployment configuration from file."""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def create_network(self, network_name: str, driver: str = 'bridge') -> str:
        """Create a Docker network for the deployment."""
        try:
            network = self.client.networks.create(
                name=network_name,
                driver=driver
            )
            self.networks[network_name] = network
            return network.id
        except docker.errors.APIError as e:
            print(f"Error creating network: {e}")
            return None

    def deploy_component(self, name: str, image: str, config: Dict[str, Any]) -> str:
        """Deploy a component as a container."""
        try:
            # Create container with specified configuration
            container = self.client.containers.run(
                image=image,
                name=name,
                detach=True,
                network=config.get('network', 'bridge'),
                environment=config.get('environment', {}),
                volumes=config.get('volumes', {}),
                ports=config.get('ports', {}),
                restart_policy={'Name': 'unless-stopped'},
                **config.get('extra_args', {})
            )

            self.containers[name] = container
            print(f"Deployed component {name}")
            return container.id
        except docker.errors.APIError as e:
            print(f"Error deploying component {name}: {e}")
            return None

    def deploy_system(self, system_config: Dict[str, Any]):
        """Deploy the entire system from configuration."""
        # Create network if specified
        if 'network' in system_config:
            network_name = system_config['network']
            self.create_network(network_name)

        # Deploy each component
        for component_name, component_config in system_config.get('components', {}).items():
            self.deploy_component(component_name, component_config['image'], component_config)

    def monitor_system(self) -> Dict[str, Any]:
        """Monitor the deployed system."""
        status = {}

        for name, container in self.containers.items():
            try:
                container.reload()
                status[name] = {
                    'status': container.status,
                    'image': container.image.tags[0] if container.image.tags else 'unknown',
                    'created': container.attrs['Created'],
                    'ports': container.ports
                }
            except docker.errors.NotFound:
                status[name] = {'status': 'not_found', 'error': 'Container not found'}
            except Exception as e:
                status[name] = {'status': 'error', 'error': str(e)}

        return status

    def scale_component(self, name: str, replicas: int):
        """Scale a component to specified number of replicas."""
        # In a real system, this would create/destroy containers
        # For now, we'll just log the scaling request
        print(f"Scaling component {name} to {replicas} replicas")

    def rollback_component(self, name: str):
        """Rollback a component to previous version."""
        print(f"Rolling back component {name}")


class KubernetesDeployment:
    """Alternative deployment using Kubernetes."""
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.kubectl_cmd = 'kubectl'

    def deploy_from_yaml(self, yaml_file: str):
        """Deploy from Kubernetes YAML file."""
        cmd = [self.kubectl_cmd, 'apply', '-f', yaml_file]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Deployment failed: {result.stderr}")
            return False
        else:
            print(f"Deployment successful: {result.stdout}")
            return True

    def create_deployment(self, name: str, image: str, replicas: int = 1) -> Dict[str, Any]:
        """Create a Kubernetes deployment."""
        deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      containers:
      - name: {name}
        image: {image}
        ports:
        - containerPort: 80
"""

        # Write to temporary file and deploy
        temp_file = f"/tmp/{name}_deployment.yaml"
        with open(temp_file, 'w') as f:
            f.write(deployment_yaml)

        success = self.deploy_from_yaml(temp_file)
        os.remove(temp_file)  # Clean up temp file

        return {'success': success, 'deployment_name': name}
```

## 13.4 System Validation and Testing

### 13.4.1 Integration Testing Framework
```python
# integration_testing.py
import unittest
import time
from typing import Dict, Any, List
import threading


class IntegrationTestFramework:
    """Framework for testing integrated systems."""
    def __init__(self, system_interface):
        self.system_interface = system_interface
        self.test_results = {}
        self.test_history = []

    def run_integration_tests(self, test_suite: List[str]) -> Dict[str, Any]:
        """Run a suite of integration tests."""
        results = {}

        for test_name in test_suite:
            print(f"Running test: {test_name}")
            test_result = self.run_single_test(test_name)
            results[test_name] = test_result
            self.test_history.append({
                'test': test_name,
                'result': test_result,
                'timestamp': time.time()
            })

        return results

    def run_single_test(self, test_name: str) -> Dict[str, Any]:
        """Run a single integration test."""
        start_time = time.time()

        try:
            # Execute the specific test
            if test_name == 'perception_integration':
                result = self.test_perception_integration()
            elif test_name == 'control_integration':
                result = self.test_control_integration()
            elif test_name == 'navigation_integration':
                result = self.test_navigation_integration()
            elif test_name == 'manipulation_integration':
                result = self.test_manipulation_integration()
            elif test_name == 'system_stability':
                result = self.test_system_stability()
            else:
                result = {'success': False, 'error': f'Unknown test: {test_name}'}

            result['duration'] = time.time() - start_time
            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }

    def test_perception_integration(self) -> Dict[str, Any]:
        """Test integration of perception components."""
        # Test that perception pipeline works end-to-end
        try:
            # Trigger perception pipeline
            sensor_data = self.system_interface.read_sensors()

            # Process through perception stack
            perception_results = self.system_interface.process_perception(sensor_data)

            # Validate results
            required_keys = ['objects', 'obstacles', 'landmarks']
            success = all(key in perception_results for key in required_keys)

            return {
                'success': success,
                'results': perception_results,
                'metrics': {
                    'object_count': len(perception_results.get('objects', [])),
                    'processing_time': perception_results.get('processing_time', 0)
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_control_integration(self) -> Dict[str, Any]:
        """Test integration of control components."""
        try:
            # Send control command
            command = {'linear_velocity': 0.5, 'angular_velocity': 0.1}
            self.system_interface.send_control_command(command)

            # Monitor response
            response = self.system_interface.get_control_response()

            # Validate response
            success = (abs(response.get('linear_velocity', 0) - command['linear_velocity']) < 0.1 and
                      abs(response.get('angular_velocity', 0) - command['angular_velocity']) < 0.1)

            return {
                'success': success,
                'command': command,
                'response': response,
                'error': abs(response.get('linear_velocity', 0) - command['linear_velocity'])
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_navigation_integration(self) -> Dict[str, Any]:
        """Test integration of navigation components."""
        try:
            # Set navigation goal
            goal = {'x': 1.0, 'y': 1.0, 'theta': 0.0}
            self.system_interface.set_navigation_goal(goal)

            # Monitor navigation progress
            start_time = time.time()
            timeout = 30.0  # 30 second timeout

            while time.time() - start_time < timeout:
                status = self.system_interface.get_navigation_status()
                if status.get('status') == 'reached':
                    return {'success': True, 'final_pose': status.get('pose')}
                elif status.get('status') == 'failed':
                    return {'success': False, 'reason': 'Navigation failed'}
                time.sleep(0.1)

            return {'success': False, 'error': 'Navigation timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_manipulation_integration(self) -> Dict[str, Any]:
        """Test integration of manipulation components."""
        try:
            # Command manipulation task
            task = {
                'action': 'pick',
                'object': 'red_cup',
                'location': 'table'
            }
            self.system_interface.execute_manipulation_task(task)

            # Monitor task progress
            start_time = time.time()
            timeout = 60.0

            while time.time() - start_time < timeout:
                status = self.system_interface.get_manipulation_status()
                if status.get('status') == 'completed':
                    return {'success': True, 'result': status}
                elif status.get('status') == 'failed':
                    return {'success': False, 'reason': status.get('error')}
                time.sleep(0.1)

            return {'success': False, 'error': 'Manipulation timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_system_stability(self) -> Dict[str, Any]:
        """Test overall system stability."""
        try:
            # Run system under load for extended period
            test_duration = 60.0  # 1 minute test
            start_time = time.time()

            while time.time() - start_time < test_duration:
                # Continuously monitor system health
                health = self.system_interface.get_system_health()

                if not health.get('all_systems_operational', True):
                    return {
                        'success': False,
                        'error': 'System instability detected',
                        'health': health
                    }

                time.sleep(1.0)  # Check every second

            return {
                'success': True,
                'test_duration': test_duration,
                'final_health': health
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def generate_test_report(self) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("=== INTEGRATION TEST REPORT ===\n")

        total_tests = len(self.test_history)
        passed_tests = sum(1 for test in self.test_history if test['result']['success'])
        failed_tests = total_tests - passed_tests

        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {failed_tests}")
        report.append(f"Success Rate: {passed_tests/total_tests*100:.1f}%\n")

        for test in self.test_history:
            status = "PASS" if test['result']['success'] else "FAIL"
            duration = test['result'].get('duration', 0)
            report.append(f"{status}: {test['test']} ({duration:.2f}s)")

            if not test['result']['success']:
                error = test['result'].get('error', 'Unknown error')
                report.append(f"  Error: {error}")

        return "\n".join(report)


class SystemValidator:
    """Validates the integrated system against requirements."""
    def __init__(self, integration_framework: IntegrationTestFramework):
        self.framework = integration_framework
        self.requirements = {}
        self.validation_results = {}

    def define_requirements(self, requirements: Dict[str, Any]):
        """Define system requirements to validate against."""
        self.requirements = requirements

    def validate_system(self) -> Dict[str, Any]:
        """Validate the system against defined requirements."""
        validation_results = {}

        # Run integration tests
        test_results = self.framework.run_integration_tests([
            'perception_integration',
            'control_integration',
            'navigation_integration',
            'manipulation_integration',
            'system_stability'
        ])

        # Validate against requirements
        for req_id, requirement in self.requirements.items():
            validation_results[req_id] = self.validate_requirement(req_id, requirement, test_results)

        self.validation_results = validation_results
        return validation_results

    def validate_requirement(self, req_id: str, requirement: Dict[str, Any], test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a specific requirement against test results."""
        req_type = requirement.get('type', 'functional')
        req_condition = requirement.get('condition', lambda x: True)

        if req_type == 'performance':
            # Validate performance requirements
            metric = requirement.get('metric')
            threshold = requirement.get('threshold')

            # Check if metric meets threshold
            success = self.check_performance_metric(metric, threshold, test_results)

        elif req_type == 'safety':
            # Validate safety requirements
            success = self.check_safety_conditions(test_results)

        elif req_type == 'functional':
            # Validate functional requirements
            success = self.check_functional_requirements(requirement, test_results)
        else:
            success = True  # Default to success for unknown requirement types

        return {
            'requirement': requirement,
            'success': success,
            'test_results': test_results
        }

    def check_performance_metric(self, metric: str, threshold: float, test_results: Dict[str, Any]) -> bool:
        """Check if performance metric meets threshold."""
        # This is a simplified example
        # In practice, you'd extract the specific metric from test results
        if metric == 'processing_time' and threshold:
            for test_result in test_results.values():
                if 'metrics' in test_result:
                    proc_time = test_result['metrics'].get('processing_time', float('inf'))
                    if proc_time > threshold:
                        return False
        return True

    def check_safety_conditions(self, test_results: Dict[str, Any]) -> bool:
        """Check if safety conditions are met."""
        # Check that emergency stop works, no collisions, etc.
        stability_result = test_results.get('system_stability', {})
        return stability_result.get('success', True)

    def check_functional_requirements(self, requirement: Dict[str, Any], test_results: Dict[str, Any]) -> bool:
        """Check if functional requirements are met."""
        # This would check specific functional test results
        # against the requirement condition
        return True
```

### 13.4.2 Safety and Reliability Validation
```python
# safety_validation.py
import time
import threading
from typing import Dict, Any, List
import logging


class SafetyValidator:
    """Validates safety aspects of the integrated system."""
    def __init__(self):
        self.safety_limits = {
            'velocity': {'linear': 1.0, 'angular': 1.0},  # m/s, rad/s
            'acceleration': {'linear': 2.0, 'angular': 2.0},  # m/s², rad/s²
            'force': {'gripper': 50.0},  # Newtons
            'temperature': {'motors': 80.0, 'cpu': 85.0}  # Celsius
        }
        self.safety_monitoring = True
        self.violation_log = []
        self.emergency_procedures = []

    def start_safety_monitoring(self):
        """Start continuous safety monitoring."""
        self.monitoring_thread = threading.Thread(target=self.safety_monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def safety_monitor_loop(self):
        """Continuous safety monitoring loop."""
        while self.safety_monitoring:
            try:
                self.check_safety_conditions()
                time.sleep(0.1)  # Check every 100ms
            except Exception as e:
                logging.error(f"Safety monitoring error: {e}")

    def check_safety_conditions(self):
        """Check all safety conditions."""
        # This would interface with the actual system
        # For simulation, we'll check simulated conditions
        system_state = self.get_system_state_simulation()

        # Check velocity limits
        if self.check_velocity_limits(system_state):
            self.log_safety_violation('velocity_limit_exceeded', system_state)

        # Check acceleration limits
        if self.check_acceleration_limits(system_state):
            self.log_safety_violation('acceleration_limit_exceeded', system_state)

        # Check force limits
        if self.check_force_limits(system_state):
            self.log_safety_violation('force_limit_exceeded', system_state)

        # Check temperature limits
        if self.check_temperature_limits(system_state):
            self.log_safety_violation('temperature_limit_exceeded', system_state)

        # Check collision conditions
        if self.check_collision_conditions(system_state):
            self.log_safety_violation('collision_detected', system_state)

    def get_system_state_simulation(self) -> Dict[str, Any]:
        """Get simulated system state for safety checking."""
        # In practice, this would read from the actual system
        return {
            'velocity': {'linear': 0.5, 'angular': 0.2},
            'acceleration': {'linear': 1.0, 'angular': 0.5},
            'forces': {'gripper': 10.0},
            'temperatures': {'motors': 45.0, 'cpu': 50.0},
            'position': [1.0, 1.0, 0.0],
            'obstacles': []  # No obstacles in simulation
        }

    def check_velocity_limits(self, state: Dict[str, Any]) -> bool:
        """Check if velocity limits are exceeded."""
        linear_vel = state['velocity'].get('linear', 0)
        angular_vel = state['velocity'].get('angular', 0)

        return (abs(linear_vel) > self.safety_limits['velocity']['linear'] or
                abs(angular_vel) > self.safety_limits['velocity']['angular'])

    def check_acceleration_limits(self, state: Dict[str, Any]) -> bool:
        """Check if acceleration limits are exceeded."""
        linear_acc = state['acceleration'].get('linear', 0)
        angular_acc = state['acceleration'].get('angular', 0)

        return (abs(linear_acc) > self.safety_limits['acceleration']['linear'] or
                abs(angular_acc) > self.safety_limits['acceleration']['angular'])

    def check_force_limits(self, state: Dict[str, Any]) -> bool:
        """Check if force limits are exceeded."""
        gripper_force = state['forces'].get('gripper', 0)
        max_force = self.safety_limits['force']['gripper']

        return abs(gripper_force) > max_force

    def check_temperature_limits(self, state: Dict[str, Any]) -> bool:
        """Check if temperature limits are exceeded."""
        motor_temp = state['temperatures'].get('motors', 0)
        cpu_temp = state['temperatures'].get('cpu', 0)

        motor_limit = self.safety_limits['temperature']['motors']
        cpu_limit = self.safety_limits['temperature']['cpu']

        return (motor_temp > motor_limit or cpu_temp > cpu_limit)

    def check_collision_conditions(self, state: Dict[str, Any]) -> bool:
        """Check for collision conditions."""
        # Check if obstacles are too close
        obstacles = state.get('obstacles', [])
        for obstacle in obstacles:
            distance = obstacle.get('distance', float('inf'))
            if distance < 0.3:  # 30cm safety margin
                return True
        return False

    def log_safety_violation(self, violation_type: str, state: Dict[str, Any]):
        """Log a safety violation."""
        violation = {
            'timestamp': time.time(),
            'type': violation_type,
            'state': state.copy(),
            'severity': self.assess_violation_severity(violation_type)
        }
        self.violation_log.append(violation)

        # Trigger appropriate emergency procedure
        self.trigger_emergency_procedure(violation_type)

    def assess_violation_severity(self, violation_type: str) -> str:
        """Assess the severity of a safety violation."""
        high_severity = ['collision_detected', 'temperature_limit_exceeded']
        medium_severity = ['force_limit_exceeded', 'acceleration_limit_exceeded']

        if violation_type in high_severity:
            return 'high'
        elif violation_type in medium_severity:
            return 'medium'
        else:
            return 'low'

    def trigger_emergency_procedure(self, violation_type: str):
        """Trigger appropriate emergency procedure."""
        if violation_type == 'collision_detected':
            self.emergency_stop()
        elif violation_type == 'temperature_limit_exceeded':
            self.reduce_power()
        elif violation_type == 'force_limit_exceeded':
            self.release_gripper()

    def emergency_stop(self):
        """Execute emergency stop procedure."""
        print("EMERGENCY STOP TRIGGERED")
        # In practice, this would send stop commands to all actuators

    def reduce_power(self):
        """Reduce system power to prevent overheating."""
        print("Reducing system power due to temperature limit exceeded")

    def release_gripper(self):
        """Release gripper due to force limit exceeded."""
        print("Releasing gripper due to force limit exceeded")

    def get_safety_report(self) -> Dict[str, Any]:
        """Generate safety validation report."""
        return {
            'violation_count': len(self.violation_log),
            'violations': self.violation_log[-10:],  # Last 10 violations
            'current_status': 'safe' if not self.violation_log else 'violations_detected',
            'safety_limits': self.safety_limits
        }
```

## 13.5 Deployment and Maintenance

### 13.5.1 Deployment Automation
```python
# deployment_automation.py
import subprocess
import sys
import os
from typing import Dict, Any, List
import json
import shutil


class DeploymentAutomator:
    """Automates the deployment process."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.deployment_steps = []
        self.deployment_log = []

    def add_deployment_step(self, name: str, command: str, description: str = ""):
        """Add a step to the deployment process."""
        self.deployment_steps.append({
            'name': name,
            'command': command,
            'description': description,
            'executed': False,
            'success': False,
            'output': ''
        })

    def execute_deployment(self) -> Dict[str, Any]:
        """Execute the complete deployment process."""
        results = {
            'deployment_id': self.generate_deployment_id(),
            'start_time': time.time(),
            'steps': [],
            'overall_success': True
        }

        for step in self.deployment_steps:
            print(f"Executing step: {step['name']}")
            print(f"Description: {step['description']}")

            try:
                result = subprocess.run(
                    step['command'],
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per step
                )

                step['executed'] = True
                step['success'] = result.returncode == 0
                step['output'] = result.stdout
                step['error'] = result.stderr
                step['returncode'] = result.returncode
                step['timestamp'] = time.time()

                if not step['success']:
                    results['overall_success'] = False
                    print(f"Step failed: {step['name']}")
                    print(f"Error: {result.stderr}")
                    break  # Stop deployment on failure

                print(f"Step completed successfully: {step['name']}")

            except subprocess.TimeoutExpired:
                step['executed'] = True
                step['success'] = False
                step['error'] = 'Step timed out'
                step['timestamp'] = time.time()
                results['overall_success'] = False
                print(f"Step timed out: {step['name']}")
                break
            except Exception as e:
                step['executed'] = True
                step['success'] = False
                step['error'] = str(e)
                step['timestamp'] = time.time()
                results['overall_success'] = False
                print(f"Step failed with exception: {step['name']}, Error: {e}")
                break

            results['steps'].append(step.copy())

        results['end_time'] = time.time()
        results['duration'] = results['end_time'] - results['start_time']

        # Log deployment
        self.deployment_log.append(results)

        return results

    def generate_deployment_id(self) -> str:
        """Generate a unique deployment ID."""
        import uuid
        return str(uuid.uuid4())

    def setup_deployment_environment(self):
        """Setup the deployment environment."""
        # Add common deployment steps
        self.add_deployment_step(
            'check_prerequisites',
            'python3 --version && docker --version && git --version',
            'Verify required tools are installed'
        )

        self.add_deployment_step(
            'create_directories',
            f'mkdir -p {self.config.get("deployment_dir", "/tmp/deployment")}',
            'Create deployment directories'
        )

        self.add_deployment_step(
            'setup_virtual_env',
            f'python3 -m venv {self.config.get("venv_path", "/tmp/deployment/venv")} && ' +
            f'source {self.config.get("venv_path", "/tmp/deployment/venv")}/bin/activate && ' +
            'pip install --upgrade pip',
            'Setup Python virtual environment'
        )

        if self.config.get('use_docker', False):
            self.add_deployment_step(
                'build_docker_images',
                f'cd {self.config.get("source_dir", ".")} && docker-compose build',
                'Build Docker images'
            )

        if self.config.get('use_kubernetes', False):
            self.add_deployment_step(
                'apply_k8s_manifests',
                f'kubectl apply -f {self.config.get("k8s_manifests_dir", "k8s/")}',
                'Apply Kubernetes manifests'
            )

    def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback to a previous deployment."""
        # Find the deployment to rollback
        target_deployment = None
        for deployment in reversed(self.deployment_log):
            if deployment['deployment_id'] == deployment_id:
                target_deployment = deployment
                break

        if not target_deployment:
            print(f"Deployment {deployment_id} not found")
            return False

        print(f"Rolling back to deployment {deployment_id}")
        # Implementation would depend on the deployment strategy
        # For now, return True as a placeholder
        return True


class ConfigurationManager:
    """Manages system configuration for deployment."""
    def __init__(self):
        self.config = {}
        self.config_templates = {}
        self.parameter_schemas = {}

    def load_config_template(self, name: str, template_path: str):
        """Load a configuration template."""
        with open(template_path, 'r') as f:
            self.config_templates[name] = f.read()

    def generate_config(self, template_name: str, parameters: Dict[str, Any]) -> str:
        """Generate configuration from template and parameters."""
        if template_name not in self.config_templates:
            raise ValueError(f"Template {template_name} not found")

        template = self.config_templates[template_name]

        # Replace placeholders with actual values
        for key, value in parameters.items():
            placeholder = f"{{{key}}}"
            template = template.replace(placeholder, str(value))

        return template

    def validate_config(self, config: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """Validate configuration against schema."""
        schema = self.parameter_schemas.get(schema_name, {})

        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        for key, constraints in schema.items():
            if key not in config:
                if constraints.get('required', False):
                    validation_result['errors'].append(f"Required parameter '{key}' missing")
                    validation_result['valid'] = False
                continue

            value = config[key]
            expected_type = constraints.get('type')
            min_val = constraints.get('min')
            max_val = constraints.get('max')

            # Type checking
            if expected_type and not isinstance(value, expected_type):
                validation_result['errors'].append(
                    f"Parameter '{key}' has wrong type. Expected {expected_type}, got {type(value)}"
                )
                validation_result['valid'] = False

            # Range checking
            if min_val is not None and isinstance(value, (int, float)) and value < min_val:
                validation_result['warnings'].append(
                    f"Parameter '{key}' is below minimum value {min_val}"
                )

            if max_val is not None and isinstance(value, (int, float)) and value > max_val:
                validation_result['warnings'].append(
                    f"Parameter '{key}' is above maximum value {max_val}"
                )

        return validation_result

    def save_config(self, config: Dict[str, Any], file_path: str):
        """Save configuration to file."""
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)

    def load_config(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        with open(file_path, 'r') as f:
            return json.load(f)
```

### 13.5.2 Monitoring and Maintenance
```python
# monitoring_maintenance.py
import psutil
import GPUtil
import time
from datetime import datetime
from typing import Dict, Any, List
import logging
import smtplib
from email.mime.text import MIMEText


class SystemMonitor:
    """Monitors system performance and health."""
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self.health_thresholds = {
            'cpu_usage': 80,      # percentage
            'memory_usage': 85,   # percentage
            'disk_usage': 90,     # percentage
            'gpu_usage': 85,      # percentage
            'temperature': 80,    # Celsius
            'network_error_rate': 0.01  # percentage
        }
        self.notification_config = {}

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'usage': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'load_average': psutil.getloadavg()
            },
            'memory': {
                'usage': psutil.virtual_memory().percent,
                'available': psutil.virtual_memory().available,
                'total': psutil.virtual_memory().total
            },
            'disk': {
                'usage': psutil.disk_usage('/').percent,
                'free': psutil.disk_usage('/').free,
                'total': psutil.disk_usage('/').total
            },
            'network': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv,
                'packets_sent': psutil.net_io_counters().packets_sent,
                'packets_recv': psutil.net_io_counters().packets_recv
            }
        }

        # GPU metrics if available
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Use first GPU
            metrics['gpu'] = {
                'usage': gpu.load * 100,
                'memory_usage': gpu.memoryUtil * 100,
                'temperature': gpu.temperature
            }

        # Add to history
        self.metrics_history.append(metrics)

        # Check for threshold violations
        self.check_thresholds(metrics)

        return metrics

    def check_thresholds(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts."""
        for metric_name, threshold in self.health_thresholds.items():
            if self.is_threshold_violated(metric_name, metrics, threshold):
                alert = {
                    'timestamp': metrics['timestamp'],
                    'metric': metric_name,
                    'value': self.get_metric_value(metric_name, metrics),
                    'threshold': threshold,
                    'severity': 'high' if self.is_critical_violation(metric_name, metrics, threshold) else 'medium'
                }
                self.alerts.append(alert)

                # Send notification
                self.send_alert_notification(alert)

    def is_threshold_violated(self, metric_name: str, metrics: Dict[str, Any], threshold: float) -> bool:
        """Check if a specific metric violates its threshold."""
        if metric_name == 'cpu_usage':
            return metrics['cpu']['usage'] > threshold
        elif metric_name == 'memory_usage':
            return metrics['memory']['usage'] > threshold
        elif metric_name == 'disk_usage':
            return metrics['disk']['usage'] > threshold
        elif metric_name == 'gpu_usage':
            gpu_metrics = metrics.get('gpu', {})
            return gpu_metrics.get('usage', 0) > threshold
        elif metric_name == 'temperature':
            gpu_metrics = metrics.get('gpu', {})
            return gpu_metrics.get('temperature', 0) > threshold
        return False

    def is_critical_violation(self, metric_name: str, metrics: Dict[str, Any], threshold: float) -> bool:
        """Check if threshold violation is critical."""
        value = self.get_metric_value(metric_name, metrics)
        return value > threshold * 1.2  # 20% above threshold is critical

    def get_metric_value(self, metric_name: str, metrics: Dict[str, Any]) -> float:
        """Get the value of a specific metric."""
        if metric_name == 'cpu_usage':
            return metrics['cpu']['usage']
        elif metric_name == 'memory_usage':
            return metrics['memory']['usage']
        elif metric_name == 'disk_usage':
            return metrics['disk']['usage']
        elif metric_name == 'gpu_usage':
            return metrics.get('gpu', {}).get('usage', 0)
        elif metric_name == 'temperature':
            return metrics.get('gpu', {}).get('temperature', 0)
        return 0.0

    def send_alert_notification(self, alert: Dict[str, Any]):
        """Send alert notification."""
        print(f"ALERT: {alert['metric']} threshold exceeded!")
        print(f"Value: {alert['value']}, Threshold: {alert['threshold']}")
        print(f"Time: {alert['timestamp']}")
        print(f"Severity: {alert['severity']}")

        # In a real system, this would send emails, SMS, or other notifications
        if self.notification_config.get('email_enabled'):
            self.send_email_notification(alert)

    def send_email_notification(self, alert: Dict[str, Any]):
        """Send email notification."""
        # This is a placeholder - in practice, configure SMTP settings
        pass

    def generate_health_report(self) -> Dict[str, Any]:
        """Generate system health report."""
        if not self.metrics_history:
            return {'error': 'No metrics collected yet'}

        latest_metrics = self.metrics_history[-1]
        historical_metrics = self.metrics_history[-10:]  # Last 10 readings

        # Calculate averages
        avg_cpu = sum(m['cpu']['usage'] for m in historical_metrics) / len(historical_metrics)
        avg_memory = sum(m['memory']['usage'] for m in historical_metrics) / len(historical_metrics)
        avg_disk = sum(m['disk']['usage'] for m in historical_metrics) / len(historical_metrics)

        return {
            'latest_metrics': latest_metrics,
            'averages': {
                'cpu_usage': avg_cpu,
                'memory_usage': avg_memory,
                'disk_usage': avg_disk
            },
            'alert_count': len(self.alerts),
            'recent_alerts': self.alerts[-5:],  # Last 5 alerts
            'system_uptime': self.calculate_uptime()
        }

    def calculate_uptime(self) -> float:
        """Calculate system uptime."""
        if not self.metrics_history:
            return 0.0

        start_time = datetime.fromisoformat(self.metrics_history[0]['timestamp'])
        current_time = datetime.fromisoformat(self.metrics_history[-1]['timestamp'])
        return (current_time - start_time).total_seconds()


class MaintenanceScheduler:
    """Schedules and manages system maintenance tasks."""
    def __init__(self):
        self.scheduled_tasks = []
        self.completed_tasks = []
        self.maintenance_windows = []

    def schedule_task(self, task_name: str, command: str, schedule: str, description: str = ""):
        """Schedule a maintenance task."""
        task = {
            'name': task_name,
            'command': command,
            'schedule': schedule,  # Cron-like schedule
            'description': description,
            'last_run': None,
            'status': 'scheduled'
        }
        self.scheduled_tasks.append(task)

    def run_scheduled_tasks(self):
        """Run scheduled tasks that are due."""
        current_time = datetime.now()

        for task in self.scheduled_tasks:
            if self.is_task_due(task, current_time):
                self.execute_task(task)

    def is_task_due(self, task: Dict[str, Any], current_time: datetime) -> bool:
        """Check if a task is due for execution."""
        # Simplified schedule checking
        # In practice, use a proper cron parser
        return False  # Placeholder

    def execute_task(self, task: Dict[str, Any]):
        """Execute a scheduled task."""
        try:
            result = subprocess.run(
                task['command'],
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )

            task['last_run'] = datetime.now().isoformat()
            task['status'] = 'completed' if result.returncode == 0 else 'failed'
            task['output'] = result.stdout
            task['error'] = result.stderr

            self.completed_tasks.append(task.copy())

        except Exception as e:
            task['status'] = 'error'
            task['error'] = str(e)
            self.completed_tasks.append(task.copy())

    def add_maintenance_window(self, start_time: str, end_time: str, description: str = ""):
        """Add a maintenance window."""
        window = {
            'start_time': start_time,
            'end_time': end_time,
            'description': description,
            'active': False
        }
        self.maintenance_windows.append(window)

    def is_in_maintenance_window(self) -> bool:
        """Check if current time is within a maintenance window."""
        current_time = datetime.now().strftime('%H:%M')

        for window in self.maintenance_windows:
            if window['start_time'] <= current_time <= window['end_time']:
                window['active'] = True
                return True

        return False
```

## 13.6 Implementation Example: Complete System Integration

```python
# complete_integration_example.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import numpy as np
import threading
import time


class CompleteIntegrationNode(Node):
    """Complete integration node combining all system components."""
    def __init__(self):
        super().__init__('complete_integration_node')

        # Initialize all system components
        self.component_manager = ComponentManager()
        self.integration_framework = IntegrationTestFramework(self)
        self.safety_validator = SafetyValidator()
        self.deployment_automator = DeploymentAutomator({})
        self.system_monitor = SystemMonitor()

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/system_status', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10)

        # Internal state
        self.bridge = CvBridge()
        self.current_joint_state = JointState()
        self.current_imu_data = Imu()
        self.current_scan_data = LaserScan()
        self.current_image = None
        self.system_state = 'idle'
        self.emergency_stop_active = False

        # Integration timer
        self.integration_timer = self.create_timer(0.1, self.integration_callback)  # 10 Hz

        # Initialize safety monitoring
        self.safety_validator.start_safety_monitoring()

        # System metrics collection timer
        self.metrics_timer = self.create_timer(5.0, self.collect_system_metrics)  # Every 5 seconds

        self.get_logger().info("Complete Integration Node initialized")

    def joint_state_callback(self, msg):
        """Handle joint state updates."""
        self.current_joint_state = msg

    def imu_callback(self, msg):
        """Handle IMU data updates."""
        self.current_imu_data = msg

    def scan_callback(self, msg):
        """Handle laser scan updates."""
        self.current_scan_data = msg

    def image_callback(self, msg):
        """Handle camera image updates."""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def integration_callback(self):
        """Main integration callback - coordinate all system components."""
        if self.emergency_stop_active:
            # Only run safety checks during emergency stop
            self.run_safety_checks()
            return

        try:
            # 1. Collect sensor data
            sensor_data = self.collect_sensor_data()

            # 2. Run perception pipeline
            perception_results = self.run_perception_pipeline(sensor_data)

            # 3. Plan actions based on perception
            actions = self.plan_actions(perception_results)

            # 4. Execute actions safely
            self.execute_actions_safely(actions)

            # 5. Update system status
            self.update_system_status()

        except Exception as e:
            self.get_logger().error(f"Integration error: {e}")
            self.trigger_emergency_stop()

    def collect_sensor_data(self) -> Dict[str, Any]:
        """Collect data from all sensors."""
        sensor_data = {
            'joint_state': {
                'position': list(self.current_joint_state.position),
                'velocity': list(self.current_joint_state.velocity),
                'effort': list(self.current_joint_state.effort)
            },
            'imu': {
                'orientation': [
                    self.current_imu_data.orientation.x,
                    self.current_imu_data.orientation.y,
                    self.current_imu_data.orientation.z,
                    self.current_imu_data.orientation.w
                ],
                'angular_velocity': [
                    self.current_imu_data.angular_velocity.x,
                    self.current_imu_data.angular_velocity.y,
                    self.current_imu_data.angular_velocity.z
                ],
                'linear_acceleration': [
                    self.current_imu_data.linear_acceleration.x,
                    self.current_imu_data.linear_acceleration.y,
                    self.current_imu_data.linear_acceleration.z
                ]
            },
            'laser_scan': {
                'ranges': list(self.current_scan_data.ranges),
                'intensities': list(self.current_scan_data.intensities)
            },
            'image': self.current_image
        }

        return sensor_data

    def run_perception_pipeline(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete perception pipeline."""
        results = {}

        # Process camera data for object detection
        if sensor_data['image'] is not None:
            results['objects'] = self.detect_objects_in_image(sensor_data['image'])
            results['features'] = self.extract_visual_features(sensor_data['image'])

        # Process laser scan for obstacle detection
        results['obstacles'] = self.detect_obstacles_in_scan(sensor_data['laser_scan']['ranges'])

        # Process IMU for state estimation
        results['pose_estimate'] = self.estimate_pose_from_imu(sensor_data['imu'])

        # Process joint state for kinematic state
        results['kinematic_state'] = self.estimate_kinematic_state(sensor_data['joint_state'])

        return results

    def detect_objects_in_image(self, image):
        """Detect objects in image (simulated)."""
        # In practice, use object detection models
        # For simulation, return mock detections
        return [
            {'class': 'person', 'bbox': [100, 100, 200, 200], 'confidence': 0.9},
            {'class': 'cup', 'bbox': [300, 150, 350, 200], 'confidence': 0.8}
        ]

    def extract_visual_features(self, image):
        """Extract visual features (simulated)."""
        # In practice, use feature extraction models
        return {'features': np.random.random(512).tolist(), 'timestamp': time.time()}

    def detect_obstacles_in_scan(self, ranges):
        """Detect obstacles from laser scan."""
        obstacles = []
        min_distance = 1.0  # meters

        for i, range_val in enumerate(ranges):
            if 0.1 < range_val < min_distance:  # Valid range and close
                angle = i * (2 * np.pi / len(ranges))  # Assuming 360-degree scan
                obstacles.append({
                    'angle': angle,
                    'distance': range_val,
                    'x': range_val * np.cos(angle),
                    'y': range_val * np.sin(angle)
                })

        return obstacles

    def estimate_pose_from_imu(self, imu_data):
        """Estimate pose from IMU data."""
        # Simplified pose estimation
        return {
            'orientation': imu_data['orientation'],
            'angular_velocity': imu_data['angular_velocity'],
            'linear_acceleration': imu_data['linear_acceleration']
        }

    def estimate_kinematic_state(self, joint_state):
        """Estimate kinematic state from joint data."""
        return {
            'joint_positions': joint_state['position'],
            'joint_velocities': joint_state['velocity'],
            'joint_efforts': joint_state['effort']
        }

    def plan_actions(self, perception_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan actions based on perception results."""
        actions = []

        # Example action planning logic
        obstacles = perception_results.get('obstacles', [])
        objects = perception_results.get('objects', [])

        if obstacles:
            # Plan obstacle avoidance
            closest_obstacle = min(obstacles, key=lambda o: o['distance'])
            if closest_obstacle['distance'] < 0.5:  # 50cm threshold
                actions.append({
                    'type': 'avoid_obstacle',
                    'obstacle_info': closest_obstacle,
                    'preferred_direction': 'left'  # or 'right'
                })

        if objects:
            # Plan object interaction
            person = next((obj for obj in objects if obj['class'] == 'person'), None)
            if person and person['confidence'] > 0.8:
                actions.append({
                    'type': 'greet_person',
                    'person_location': person['bbox']
                })

        return actions

    def execute_actions_safely(self, actions: List[Dict[str, Any]]):
        """Execute actions with safety checks."""
        for action in actions:
            if self.check_action_safety(action):
                self.execute_single_action(action)
            else:
                self.get_logger().warn(f"Action {action['type']} failed safety check, skipping")

    def check_action_safety(self, action: Dict[str, Any]) -> bool:
        """Check if an action is safe to execute."""
        # Check current system state for safety
        if self.emergency_stop_active:
            return False

        # Check for immediate dangers
        obstacles = self.detect_obstacles_in_scan(self.current_scan_data.ranges)
        if action['type'] == 'avoid_obstacle' and obstacles:
            # This is a safe action
            return True

        # Check other safety conditions
        return True

    def execute_single_action(self, action: Dict[str, Any]):
        """Execute a single action."""
        if action['type'] == 'avoid_obstacle':
            self.execute_avoidance_maneuver(action['obstacle_info'], action['preferred_direction'])
        elif action['type'] == 'greet_person':
            self.execute_greeting_action(action['person_location'])

    def execute_avoidance_maneuver(self, obstacle_info: Dict[str, Any], direction: str):
        """Execute obstacle avoidance maneuver."""
        cmd = Twist()

        if direction == 'left':
            cmd.angular.z = 0.5  # Turn left
        else:
            cmd.angular.z = -0.5  # Turn right

        cmd.linear.x = 0.2  # Move forward slowly during turn

        self.cmd_vel_pub.publish(cmd)

    def execute_greeting_action(self, person_location: List[int]):
        """Execute greeting action for detected person."""
        # In practice, this might trigger speech, LED patterns, etc.
        self.get_logger().info(f"Greeting person at location: {person_location}")

    def update_system_status(self):
        """Update system status and publish."""
        status_msg = String()
        status_msg.data = f"System operational. State: {self.system_state}"
        self.status_pub.publish(status_msg)

    def run_safety_checks(self):
        """Run safety checks during emergency stop."""
        # Publish zero velocity to stop robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

    def trigger_emergency_stop(self):
        """Trigger emergency stop."""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            self.system_state = 'emergency_stop'

            # Publish emergency stop command
            stop_msg = Bool()
            stop_msg.data = True
            self.emergency_stop_pub.publish(stop_msg)

            self.get_logger().fatal("EMERGENCY STOP ACTIVATED")

    def collect_system_metrics(self):
        """Collect and log system metrics."""
        metrics = self.system_monitor.collect_metrics()
        self.get_logger().info(f"System metrics collected. CPU: {metrics['cpu']['usage']}%")

    def process_perception(self, sensor_data):
        """Process perception pipeline (for testing framework)."""
        return self.run_perception_pipeline(sensor_data)

    def send_control_command(self, command):
        """Send control command (for testing framework)."""
        cmd = Twist()
        cmd.linear.x = command.get('linear_velocity', 0.0)
        cmd.angular.z = command.get('angular_velocity', 0.0)
        self.cmd_vel_pub.publish(cmd)

    def get_control_response(self):
        """Get control response (for testing framework)."""
        # In practice, this would read actual motor feedback
        return {
            'linear_velocity': self.current_joint_state.velocity[0] if self.current_joint_state.velocity else 0.0,
            'angular_velocity': self.current_joint_state.velocity[1] if len(self.current_joint_state.velocity) > 1 else 0.0
        }

    def get_system_health(self):
        """Get system health status (for testing framework)."""
        return {
            'all_systems_operational': not self.emergency_stop_active,
            'component_status': self.component_manager.health_status if hasattr(self.component_manager, 'health_status') else {},
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent
        }


def main(args=None):
    rclpy.init(args=args)
    integration_node = CompleteIntegrationNode()

    try:
        rclpy.spin(integration_node)
    except KeyboardInterrupt:
        pass
    finally:
        integration_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## 13.7 Advanced Deployment Topics

### 13.7.1 Over-the-Air Updates
Implementing secure, reliable updates for deployed robotic systems.

### 13.7.2 Edge Computing Integration
Deploying AI models and processing at the edge for real-time performance.

## 13.8 Best Practices

1. **Gradual Integration**: Integrate components incrementally rather than all at once
2. **Comprehensive Testing**: Test at every level of integration
3. **Safety First**: Implement safety measures before functionality
4. **Monitoring**: Implement comprehensive monitoring and alerting
5. **Documentation**: Maintain detailed documentation for maintenance
6. **Rollback Plans**: Always have a way to revert to previous states

## Practical Exercise

### Exercise 13.1: Complete System Integration Project
**Objective**: Integrate all components developed throughout the course into a complete Physical AI system

1. Implement the system integration architecture
2. Create deployment automation scripts
3. Implement comprehensive monitoring and safety validation
4. Conduct end-to-end system testing
5. Document the complete system architecture and operation procedures
6. Perform validation tests to ensure all requirements are met

**Deliverable**: Fully integrated Physical AI system with deployment, monitoring, and validation capabilities.

## Summary

Week 13 completed the 13-week curriculum by covering system integration and deployment of Physical AI systems. You learned to combine all specialized subsystems into a cohesive whole, implement deployment architectures, establish monitoring procedures, and validate system performance and safety. This integration phase is critical for creating operational robotic systems that can function reliably in real-world environments.

[Previous: Week 12 - Language Grounding & Decision Making ←](./week12-language-grounding.md)