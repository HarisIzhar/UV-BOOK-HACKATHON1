---
sidebar_position: 13
---

# Week 12: Language Grounding & Decision Making

## Learning Objectives

By the end of this week, you will be able to:
- Understand language grounding in robotics and AI systems
- Implement natural language understanding for robot control
- Design decision-making systems that incorporate language input
- Create multimodal language processing pipelines
- Integrate language understanding with robotic action planning

## 12.1 Introduction to Language Grounding

Language grounding is the process of connecting natural language to the physical world, enabling robots to understand and act upon linguistic instructions. This involves:

- **Semantic parsing**: Converting language to structured meaning
- **World modeling**: Connecting language to environmental concepts
- **Action mapping**: Translating language to robot behaviors
- **Feedback integration**: Using sensor data to refine language understanding

### 12.1.1 Language Grounding Architecture
```
┌─────────────────────────────────────────────────────────────┐
│              Language Grounding System                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Natural     │    │ Semantic    │    │ Action      │     │
│  │ Language    │───▶│ Parser      │───▶│ Generator   │     │
│  │ Input       │    │             │    │             │     │
│  │ • Commands  │    │ • Intent    │    │ • Task      │     │
│  │ • Queries   │    │   Extraction│    │   Planning  │     │
│  │ • Dialog    │    │ • Entity    │    │ • Motion    │     │
│  │             │    │   Linking   │    │   Planning  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                   │                   │          │
│         ▼                   ▼                   ▼          │
│  ┌─────────────────────────────────────────────────┐       │
│  │              World Model                        │       │
│  │  ┌─────────────┐  ┌─────────────┐             │       │
│  │  │ Environment │  │ Object      │             │       │
│  │  │ Model       │  │ Knowledge   │             │       │
│  │  │ • Spatial   │  │ • Properties│             │       │
│  │  │ • Dynamics  │  │ • Relations │             │       │
│  │  └─────────────┘  └─────────────┘             │       │
│  └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## 12.2 Natural Language Understanding for Robotics

### 12.2.1 Command Parsing and Interpretation
```python
# command_parsing.py
import re
import spacy
from typing import Dict, List, Tuple, Any


class CommandParser:
    def __init__(self):
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install en_core_web_sm: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Define command patterns and templates
        self.command_patterns = {
            'move': [
                r'go to (.+)',
                r'move to (.+)',
                r'go over to (.+)',
                r'walk to (.+)'
            ],
            'grasp': [
                r'pick up (.+)',
                r'grab (.+)',
                r'take (.+)',
                r'get (.+)'
            ],
            'place': [
                r'put (.+) on (.+)',
                r'place (.+) on (.+)',
                r'drop (.+) on (.+)'
            ],
            'describe': [
                r'what is (.+)',
                r'describe (.+)',
                r'tell me about (.+)'
            ]
        }

        # Define spatial relations
        self.spatial_relations = {
            'near', 'close to', 'next to', 'beside', 'in front of',
            'behind', 'left of', 'right of', 'on', 'under', 'above'
        }

        # Define action verbs and their semantic roles
        self.action_verbs = {
            'move': ['go', 'move', 'walk', 'navigate', 'approach'],
            'grasp': ['pick', 'grab', 'take', 'get', 'lift'],
            'place': ['put', 'place', 'drop', 'set'],
            'describe': ['what', 'describe', 'tell', 'explain']
        }

    def parse_command(self, command: str) -> Dict[str, Any]:
        """Parse natural language command into structured format."""
        if self.nlp is None:
            return self.parse_simple(command)

        doc = self.nlp(command.lower())

        # Extract entities and their relationships
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Identify main action
        action = self.identify_action(doc)

        # Extract arguments
        arguments = self.extract_arguments(doc, action)

        # Determine spatial relationships
        spatial_info = self.extract_spatial_info(doc)

        return {
            'command': command,
            'action': action,
            'entities': entities,
            'arguments': arguments,
            'spatial_info': spatial_info,
            'raw_tokens': [token.text for token in doc]
        }

    def parse_simple(self, command: str) -> Dict[str, Any]:
        """Simple rule-based parsing when spaCy is not available."""
        command_lower = command.lower()

        # Match against predefined patterns
        for action, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command_lower)
                if match:
                    return {
                        'command': command,
                        'action': action,
                        'arguments': list(match.groups()),
                        'entities': [],
                        'spatial_info': {},
                        'raw_tokens': command.split()
                    }

        # If no pattern matches, return generic parse
        return {
            'command': command,
            'action': 'unknown',
            'arguments': [command],
            'entities': [],
            'spatial_info': {},
            'raw_tokens': command.split()
        }

    def identify_action(self, doc) -> str:
        """Identify the main action in the command."""
        for token in doc:
            if token.pos_ == 'VERB':
                for action, verbs in self.action_verbs.items():
                    if token.lemma_ in verbs:
                        return action
        return 'unknown'

    def extract_arguments(self, doc, action: str) -> Dict[str, str]:
        """Extract arguments for the identified action."""
        arguments = {}

        # Use dependency parsing to extract arguments
        for token in doc:
            if token.dep_ == 'dobj':  # Direct object
                arguments['object'] = token.text
            elif token.dep_ == 'pobj':  # Object of preposition
                arguments['location'] = token.text
            elif token.dep_ == 'prep':  # Prepositional modifier
                arguments['preposition'] = token.text

        return arguments

    def extract_spatial_info(self, doc) -> Dict[str, str]:
        """Extract spatial relationships from the command."""
        spatial_info = {}

        for token in doc:
            if token.text in self.spatial_relations:
                # Look for the object of the spatial relation
                for child in token.children:
                    if child.dep_ == 'pobj':
                        spatial_info[token.text] = child.text
                    elif child.dep_ == 'pcomp':
                        spatial_info[token.text] = child.text

        return spatial_info


class SemanticFrame:
    """Represents the semantic structure of a command."""
    def __init__(self, action: str, arguments: Dict[str, str], spatial_info: Dict[str, str]):
        self.action = action
        self.arguments = arguments
        self.spatial_info = spatial_info
        self.confidence = 1.0  # Confidence in interpretation

    def to_robot_command(self) -> Dict[str, Any]:
        """Convert semantic frame to robot-executable command."""
        robot_cmd = {
            'action': self.action,
            'parameters': {},
            'constraints': {}
        }

        # Map semantic arguments to robot parameters
        if 'object' in self.arguments:
            robot_cmd['parameters']['target_object'] = self.arguments['object']

        if 'location' in self.arguments:
            robot_cmd['parameters']['target_location'] = self.arguments['location']

        # Add spatial constraints
        robot_cmd['constraints'].update(self.spatial_info)

        return robot_cmd
```

### 12.2.2 Entity Recognition and Grounding
```python
# entity_grounding.py
import numpy as np
from typing import List, Dict, Tuple


class EntityGrounding:
    def __init__(self):
        self.object_database = {}  # Maps names to object properties
        self.spatial_memory = {}   # Maps locations to coordinates
        self.association_matrix = {}  # Tracks co-occurrence of entities

    def ground_entities(self, parsed_command: Dict[str, Any], world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Ground linguistic entities to physical objects in the world."""
        grounded_command = parsed_command.copy()

        # Ground object entities
        if 'arguments' in parsed_command:
            if 'object' in parsed_command['arguments']:
                object_name = parsed_command['arguments']['object']
                grounded_object = self.find_object_in_world(object_name, world_state)
                grounded_command['arguments']['grounded_object'] = grounded_object

            if 'location' in parsed_command['arguments']:
                location_name = parsed_command['arguments']['location']
                grounded_location = self.find_location_in_world(location_name, world_state)
                grounded_command['arguments']['grounded_location'] = grounded_location

        # Ground spatial relations
        if 'spatial_info' in parsed_command:
            grounded_spatial = {}
            for relation, entity in parsed_command['spatial_info'].items():
                grounded_entity = self.find_object_in_world(entity, world_state)
                grounded_spatial[relation] = grounded_entity
            grounded_command['spatial_info'] = grounded_spatial

        return grounded_command

    def find_object_in_world(self, object_name: str, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Find an object in the world that matches the name."""
        # Search in current world state
        if 'objects' in world_state:
            for obj in world_state['objects']:
                if self.match_object_name(object_name, obj):
                    return obj

        # If not found, return the name as a reference
        return {
            'name': object_name,
            'type': 'unknown',
            'position': None,
            'properties': {}
        }

    def match_object_name(self, query_name: str, object_info: Dict[str, Any]) -> bool:
        """Check if object name matches the query."""
        # Simple string matching (in practice, use more sophisticated matching)
        obj_name = object_info.get('name', '').lower()
        obj_type = object_info.get('type', '').lower()
        query = query_name.lower()

        return query in obj_name or query in obj_type

    def find_location_in_world(self, location_name: str, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Find a location in the world that matches the name."""
        # Search in spatial memory or world map
        if 'locations' in world_state:
            for loc in world_state['locations']:
                if location_name.lower() in loc.get('name', '').lower():
                    return loc

        # If not found, treat as relative location
        return {
            'name': location_name,
            'type': 'relative',
            'position': None,
            'properties': {}
        }

    def update_object_database(self, object_name: str, properties: Dict[str, Any]):
        """Update the object database with new information."""
        if object_name not in self.object_database:
            self.object_database[object_name] = []

        self.object_database[object_name].append(properties)

    def learn_entity_associations(self, entities: List[str]):
        """Learn associations between entities that co-occur."""
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j:
                    key = (entity1, entity2)
                    if key not in self.association_matrix:
                        self.association_matrix[key] = 0
                    self.association_matrix[key] += 1


class SpatialGrounding:
    def __init__(self):
        self.spatial_reference_system = {}  # Maps spatial terms to coordinates
        self.relative_positioning = {}      # Stores relative positions

    def ground_spatial_relations(self, spatial_info: Dict[str, str], world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Ground spatial relations to actual positions."""
        grounded_spatial = {}

        for relation, entity in spatial_info.items():
            if relation in ['near', 'close to', 'next to']:
                # Find position that is near the entity
                entity_pos = self.get_entity_position(entity, world_state)
                if entity_pos is not None:
                    grounded_spatial[relation] = self.find_nearby_position(entity_pos)

            elif relation in ['in front of', 'behind']:
                # Use robot's current position and orientation
                robot_pos = world_state.get('robot_position', [0, 0, 0])
                robot_orient = world_state.get('robot_orientation', [0, 0, 0])
                grounded_spatial[relation] = self.calculate_directional_position(
                    robot_pos, robot_orient, relation)

            elif relation in ['left of', 'right of']:
                # Calculate position to the left/right
                entity_pos = self.get_entity_position(entity, world_state)
                if entity_pos is not None:
                    grounded_spatial[relation] = self.calculate_lateral_position(
                        entity_pos, relation)

        return grounded_spatial

    def get_entity_position(self, entity: str, world_state: Dict[str, Any]) -> List[float]:
        """Get the position of an entity in the world."""
        if 'objects' in world_state:
            for obj in world_state['objects']:
                if obj.get('name', '').lower() in entity.lower():
                    return obj.get('position', [0, 0, 0])
        return None

    def find_nearby_position(self, reference_pos: List[float], distance: float = 0.5) -> List[float]:
        """Find a position nearby the reference position."""
        # Add small random offset
        offset = np.random.uniform(-distance, distance, 3)
        return [a + b for a, b in zip(reference_pos, offset)]

    def calculate_directional_position(self, robot_pos: List[float], robot_orient: List[float],
                                     direction: str) -> List[float]:
        """Calculate position in front of or behind robot."""
        # Simplified: assume robot_orient is [roll, pitch, yaw]
        yaw = robot_orient[2] if len(robot_orient) > 2 else 0

        # Calculate direction vector
        if direction == 'in front of':
            direction_vec = [np.cos(yaw), np.sin(yaw), 0]
        else:  # behind
            direction_vec = [-np.cos(yaw), -np.sin(yaw), 0]

        # Calculate position at distance
        distance = 1.0  # meters
        target_pos = [a + distance * b for a, b in zip(robot_pos, direction_vec)]
        return target_pos

    def calculate_lateral_position(self, reference_pos: List[float], direction: str) -> List[float]:
        """Calculate position to the left or right of reference."""
        # Rotate 90 degrees to get lateral direction
        if direction == 'left of':
            lateral_offset = [-0.5, 0, 0]  # Simplified
        else:  # right of
            lateral_offset = [0.5, 0, 0]

        return [a + b for a, b in zip(reference_pos, lateral_offset)]
```

## 12.3 Decision Making with Language Input

### 12.3.1 Planning from Natural Language
```python
# language_planning.py
from typing import List, Dict, Any
import heapq


class LanguagePlanner:
    def __init__(self):
        self.action_library = {
            'move_to': {'cost': 10, 'preconditions': ['robot_exists'], 'effects': ['robot_at']},
            'grasp_object': {'cost': 5, 'preconditions': ['robot_at', 'object_present'], 'effects': ['holding_object']},
            'place_object': {'cost': 5, 'preconditions': ['holding_object'], 'effects': ['object_placed']},
            'describe_object': {'cost': 2, 'preconditions': ['object_visible'], 'effects': ['description_provided']},
            'navigate_around': {'cost': 15, 'preconditions': ['obstacle_detected'], 'effects': ['path_found']}
        }

    def plan_from_command(self, semantic_frame: SemanticFrame, world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a plan to execute a command given the current world state."""
        # Convert semantic frame to planning problem
        goal_state = self.semantic_to_goal(semantic_frame, world_state)
        current_state = self.world_to_state(world_state)

        # Use A* search to find optimal plan
        plan = self.a_star_search(current_state, goal_state)

        return plan

    def semantic_to_goal(self, semantic_frame: SemanticFrame, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert semantic frame to goal state."""
        goal = {}

        if semantic_frame.action == 'move_to':
            if 'grounded_location' in semantic_frame.arguments:
                location = semantic_frame.arguments['grounded_location']
                goal['robot_at'] = location.get('position', [0, 0, 0])

        elif semantic_frame.action == 'grasp_object':
            if 'grounded_object' in semantic_frame.arguments:
                obj = semantic_frame.arguments['grounded_object']
                goal['holding_object'] = obj.get('name', 'unknown')

        elif semantic_frame.action == 'place_object':
            if 'grounded_location' in semantic_frame.arguments:
                location = semantic_frame.arguments['grounded_location']
                goal['object_placed_at'] = location.get('position', [0, 0, 0])

        return goal

    def world_to_state(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert world state to planning state representation."""
        state = {
            'robot_position': world_state.get('robot_position', [0, 0, 0]),
            'robot_orientation': world_state.get('robot_orientation', [0, 0, 0]),
            'objects': world_state.get('objects', []),
            'obstacles': world_state.get('obstacles', []),
            'holding': world_state.get('holding', None)
        }

        return state

    def a_star_search(self, start_state: Dict[str, Any], goal_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """A* search algorithm for planning."""
        # Priority queue: (cost, state, path)
        frontier = [(0, start_state, [])]
        explored = set()

        while frontier:
            cost, current_state, path = heapq.heappop(frontier)

            # Check if goal is reached
            if self.is_goal_reached(current_state, goal_state):
                return path

            # Generate successor states
            for action in self.get_applicable_actions(current_state):
                successor_state = self.apply_action(current_state, action)
                new_cost = cost + action['cost']
                new_path = path + [action]

                # Create state signature for cycle detection
                state_sig = self.state_signature(successor_state)

                if state_sig not in explored:
                    explored.add(state_sig)
                    priority = new_cost + self.heuristic(successor_state, goal_state)
                    heapq.heappush(frontier, (priority, successor_state, new_path))

        # No plan found
        return []

    def is_goal_reached(self, current_state: Dict[str, Any], goal_state: Dict[str, Any]) -> bool:
        """Check if goal state is reached."""
        for key, value in goal_state.items():
            if key not in current_state or current_state[key] != value:
                return False
        return True

    def get_applicable_actions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get actions that can be applied in the current state."""
        applicable = []

        for action_name, action_def in self.action_library.items():
            # Check preconditions
            if self.check_preconditions(action_def, state):
                action = action_def.copy()
                action['name'] = action_name
                applicable.append(action)

        return applicable

    def check_preconditions(self, action_def: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """Check if action preconditions are satisfied."""
        for precondition in action_def.get('preconditions', []):
            # Simplified precondition checking
            if precondition == 'robot_exists':
                return 'robot_position' in state
            elif precondition == 'object_present':
                return len(state.get('objects', [])) > 0
            elif precondition == 'holding_object':
                return state.get('holding') is not None
            elif precondition == 'object_visible':
                return len(state.get('objects', [])) > 0

        return True

    def apply_action(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply action to state and return new state."""
        new_state = state.copy()

        action_name = action['name']
        if action_name == 'move_to':
            new_state['robot_position'] = action.get('target_position', [0, 0, 0])
        elif action_name == 'grasp_object':
            new_state['holding'] = action.get('target_object', 'unknown')
        elif action_name == 'place_object':
            new_state['holding'] = None

        return new_state

    def state_signature(self, state: Dict[str, Any]) -> str:
        """Create a signature for state to detect cycles."""
        # Simplified signature
        pos = tuple(state.get('robot_position', [0, 0, 0]))
        holding = state.get('holding', 'none')
        return f"{pos}_{holding}"

    def heuristic(self, state: Dict[str, Any], goal: Dict[str, Any]) -> float:
        """Calculate heuristic distance to goal."""
        if 'robot_at' in goal:
            goal_pos = goal['robot_at']
            current_pos = state.get('robot_position', [0, 0, 0])
            return self.euclidean_distance(current_pos, goal_pos)
        return 0

    def euclidean_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return sum((a - b)**2 for a, b in zip(pos1, pos2))**0.5


class TaskExecutor:
    def __init__(self):
        self.planner = LanguagePlanner()
        self.current_plan = []
        self.plan_step = 0

    def execute_command(self, semantic_frame: SemanticFrame, world_state: Dict[str, Any]) -> bool:
        """Execute a command by generating and following a plan."""
        # Generate plan
        plan = self.planner.plan_from_command(semantic_frame, world_state)

        if not plan:
            print(f"Could not generate plan for command: {semantic_frame.action}")
            return False

        # Execute plan step by step
        success = self.execute_plan(plan, world_state)
        return success

    def execute_plan(self, plan: List[Dict[str, Any]], world_state: Dict[str, Any]) -> bool:
        """Execute a sequence of actions."""
        for i, action in enumerate(plan):
            print(f"Executing action {i+1}/{len(plan)}: {action['name']}")

            success = self.execute_single_action(action, world_state)
            if not success:
                print(f"Action failed: {action['name']}")
                return False

        return True

    def execute_single_action(self, action: Dict[str, Any], world_state: Dict[str, Any]) -> bool:
        """Execute a single action."""
        action_name = action['name']

        # In practice, this would interface with the robot's action system
        print(f"Executing: {action_name}")

        # Simulate action execution
        if action_name == 'move_to':
            # Move robot to target position
            target = action.get('target_position', [0, 0, 0])
            world_state['robot_position'] = target
        elif action_name == 'grasp_object':
            # Grasp the object
            obj = action.get('target_object', 'unknown')
            world_state['holding'] = obj
        elif action_name == 'place_object':
            # Place the object
            world_state['holding'] = None

        return True  # Assume success for simulation
```

### 12.3.2 Uncertainty and Confidence Handling
```python
# uncertainty_handling.py
class UncertaintyHandler:
    def __init__(self):
        self.understanding_confidence = {}
        self.action_success_probability = {}
        self.world_state_uncertainty = {}

    def calculate_command_confidence(self, parsed_command: Dict[str, Any]) -> float:
        """Calculate confidence in command understanding."""
        confidence = 1.0

        # Confidence decreases with ambiguous elements
        if parsed_command.get('action') == 'unknown':
            confidence *= 0.3

        # Check entity grounding confidence
        if 'arguments' in parsed_command:
            for arg_name, arg_value in parsed_command['arguments'].items():
                if 'grounded_' in arg_name and arg_value.get('position') is None:
                    confidence *= 0.7  # Reduce confidence for ungrounded entities

        # Check for ambiguous spatial relations
        if 'spatial_info' in parsed_command and not parsed_command['spatial_info']:
            confidence *= 0.8

        return max(0.1, confidence)  # Minimum confidence of 0.1

    def handle_ambiguous_command(self, command: str, alternatives: List[str]) -> Dict[str, Any]:
        """Handle commands with multiple possible interpretations."""
        print(f"Ambiguous command: {command}")
        print("Possible interpretations:")
        for i, alt in enumerate(alternatives):
            print(f"  {i+1}. {alt}")

        # In interactive systems, ask for clarification
        # For autonomous systems, choose the most probable
        chosen_interpretation = alternatives[0]  # Choose first as default
        confidence = 0.8 / len(alternatives)  # Distribute confidence

        return {
            'command': chosen_interpretation,
            'confidence': confidence,
            'is_ambiguous': True,
            'alternatives': alternatives
        }

    def update_world_uncertainty(self, sensor_data: Dict[str, Any], action_result: Dict[str, Any]):
        """Update uncertainty about world state based on sensor data and action outcomes."""
        # Update position uncertainty based on odometry and sensor data
        if 'position_estimate' in sensor_data:
            pos = sensor_data['position_estimate']
            uncertainty = sensor_data.get('position_uncertainty', 0.1)
            self.world_state_uncertainty['position'] = {
                'estimate': pos,
                'variance': uncertainty**2
            }

        # Update object location uncertainty
        if 'object_detections' in sensor_data:
            for detection in sensor_data['object_detections']:
                obj_name = detection.get('name', 'unknown')
                obj_pos = detection.get('position', [0, 0, 0])
                obj_uncertainty = detection.get('uncertainty', 0.1)

                self.world_state_uncertainty[f'object_{obj_name}'] = {
                    'estimate': obj_pos,
                    'variance': obj_uncertainty**2
                }

    def predict_action_outcome(self, action: Dict[str, Any], world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the outcome of an action with uncertainty."""
        predicted_state = world_state.copy()
        success_probability = 0.9  # Base success probability

        # Adjust based on world state uncertainty
        if 'object_uncertainty' in self.world_state_uncertainty:
            # Higher uncertainty may lead to lower success probability
            avg_uncertainty = np.mean(list(self.world_state_uncertainty['object_uncertainty'].values()))
            success_probability *= (1 - avg_uncertainty)

        # Adjust based on action complexity
        if action.get('name') == 'grasp_object':
            # Grasping is more complex and error-prone
            success_probability *= 0.8

        return {
            'predicted_state': predicted_state,
            'success_probability': success_probability,
            'confidence_intervals': self.calculate_confidence_intervals(action, world_state)
        }

    def calculate_confidence_intervals(self, action: Dict[str, Any], world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence intervals for action outcomes."""
        # Simplified confidence intervals
        return {
            'position_error': 0.1,  # meters
            'orientation_error': 0.1,  # radians
            'success_range': [0.7, 1.0]  # probability range
        }
```

## 12.4 Multimodal Language Processing

### 12.4.1 Vision-Language Integration
```python
# vision_language_integration.py
import numpy as np
from typing import Dict, List, Any


class VisionLanguageIntegrator:
    def __init__(self):
        self.visual_features = {}
        self.language_features = {}
        self.cross_modal_aligner = self.initialize_aligner()

    def initialize_aligner(self):
        """Initialize cross-modal alignment model."""
        # In practice, use pre-trained models like CLIP
        # For this example, use a simple alignment mechanism
        return SimpleAligner()

    def process_visual_language_input(self, image_features: np.ndarray,
                                    text_description: str) -> Dict[str, Any]:
        """Process combined visual and language input."""
        # Extract visual features
        visual_repr = self.encode_visual_features(image_features)

        # Extract language features
        lang_repr = self.encode_language_features(text_description)

        # Align visual and language representations
        alignment_score = self.cross_modal_aligner.align(visual_repr, lang_repr)

        # Generate multimodal understanding
        understanding = {
            'visual_content': self.interpret_visual_content(image_features),
            'language_content': self.interpret_language_content(text_description),
            'alignment_score': alignment_score,
            'cross_modal_attention': self.compute_cross_modal_attention(visual_repr, lang_repr)
        }

        return understanding

    def encode_visual_features(self, image_features: np.ndarray) -> np.ndarray:
        """Encode visual features for multimodal processing."""
        # In practice, use CNN features or ViT features
        # For this example, use simplified encoding
        return np.mean(image_features, axis=(0, 1))  # Global average pooling

    def encode_language_features(self, text: str) -> np.ndarray:
        """Encode language features for multimodal processing."""
        # In practice, use BERT, RoBERTa, or similar
        # For this example, use simplified encoding
        # Convert text to a simple feature vector
        feature_vector = np.zeros(512)  # Fixed size for simplicity
        words = text.lower().split()
        for word in words[:10]:  # Limit to first 10 words
            hash_val = hash(word) % 512
            feature_vector[hash_val] += 1.0
        return feature_vector

    def interpret_visual_content(self, image_features: np.ndarray) -> Dict[str, Any]:
        """Interpret the content of visual features."""
        # In practice, use object detection, segmentation, etc.
        # For this example, return simplified interpretation
        return {
            'objects': ['object1', 'object2'],  # Placeholder
            'colors': ['red', 'blue'],         # Placeholder
            'spatial_relations': ['left_of', 'above']  # Placeholder
        }

    def interpret_language_content(self, text: str) -> Dict[str, Any]:
        """Interpret the content of language text."""
        # Use the command parser for language interpretation
        parser = CommandParser()
        return parser.parse_command(text)

    def compute_cross_modal_attention(self, visual_repr: np.ndarray,
                                    lang_repr: np.ndarray) -> np.ndarray:
        """Compute attention between visual and language modalities."""
        # Simple dot product attention
        attention_weights = np.dot(visual_repr, lang_repr.T)
        return attention_weights


class SimpleAligner:
    def __init__(self):
        # Simple alignment matrix (in practice, this would be learned)
        self.alignment_matrix = np.eye(512)  # Identity for simplicity

    def align(self, visual_features: np.ndarray, language_features: np.ndarray) -> float:
        """Compute alignment score between visual and language features."""
        # Compute similarity using the alignment matrix
        aligned_lang = np.dot(self.alignment_matrix, language_features)
        similarity = np.dot(visual_features, aligned_lang) / (
            np.linalg.norm(visual_features) * np.linalg.norm(aligned_lang)
        )
        return similarity


class GroundedLanguageUnderstanding:
    def __init__(self):
        self.vision_language_integrator = VisionLanguageIntegrator()
        self.entity_grounding = EntityGrounding()
        self.spatial_grounding = SpatialGrounding()

    def understand_command_with_vision(self, command: str, visual_features: np.ndarray,
                                     world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Understand a language command grounded in visual context."""
        # Parse the command
        parser = CommandParser()
        parsed_command = parser.parse_command(command)

        # Process visual-language integration
        vision_lang_understanding = self.vision_language_integrator.process_visual_language_input(
            visual_features, command)

        # Ground entities in the current world state
        grounded_command = self.entity_grounding.ground_entities(parsed_command, world_state)

        # Ground spatial relations
        if 'spatial_info' in grounded_command:
            grounded_spatial = self.spatial_grounding.ground_spatial_relations(
                grounded_command['spatial_info'], world_state)
            grounded_command['spatial_info'] = grounded_spatial

        # Combine all understanding components
        complete_understanding = {
            'parsed_command': parsed_command,
            'vision_language_alignment': vision_lang_understanding,
            'grounded_entities': grounded_command,
            'semantic_frame': self.create_semantic_frame(grounded_command),
            'confidence': self.calculate_understanding_confidence(grounded_command)
        }

        return complete_understanding

    def create_semantic_frame(self, grounded_command: Dict[str, Any]) -> SemanticFrame:
        """Create a semantic frame from grounded command."""
        action = grounded_command.get('action', 'unknown')
        arguments = grounded_command.get('arguments', {})
        spatial_info = grounded_command.get('spatial_info', {})

        return SemanticFrame(action, arguments, spatial_info)

    def calculate_understanding_confidence(self, grounded_command: Dict[str, Any]) -> float:
        """Calculate confidence in the complete understanding."""
        # Base confidence from parsing
        base_confidence = 0.8

        # Reduce confidence if entities are not grounded
        if 'grounded_object' not in grounded_command.get('arguments', {}):
            base_confidence *= 0.7

        # Reduce confidence if spatial relations are not grounded
        if not grounded_command.get('spatial_info'):
            base_confidence *= 0.8

        return base_confidence
```

### 12.4.2 Language-Guided Decision Making
```python
# language_guided_decision.py
from enum import Enum
import random


class DecisionPolicy(Enum):
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    BALANCED = "balanced"


class LanguageGuidedDecisionMaker:
    def __init__(self):
        self.policy = DecisionPolicy.BALANCED
        self.context_memory = []
        self.preference_learning = {}

    def make_decision(self, command_understanding: Dict[str, Any],
                     world_state: Dict[str, Any],
                     available_actions: List[str]) -> Dict[str, Any]:
        """Make a decision based on language understanding and world state."""
        semantic_frame = command_understanding['semantic_frame']

        # Evaluate each available action
        action_scores = {}
        for action in available_actions:
            score = self.evaluate_action(action, semantic_frame, world_state, command_understanding)
            action_scores[action] = score

        # Select best action based on policy
        best_action = self.select_best_action(action_scores)

        return {
            'selected_action': best_action,
            'action_scores': action_scores,
            'reasoning': self.generate_reasoning(semantic_frame, best_action, world_state),
            'confidence': self.calculate_decision_confidence(action_scores, best_action)
        }

    def evaluate_action(self, action: str, semantic_frame: SemanticFrame,
                       world_state: Dict[str, Any], understanding: Dict[str, Any]) -> float:
        """Evaluate how well an action satisfies the command."""
        score = 0.0

        # Check if action matches the requested command
        if action == semantic_frame.action:
            score += 0.5  # Base match score

        # Check preconditions
        preconditions_met = self.check_action_preconditions(action, world_state)
        if preconditions_met:
            score += 0.3

        # Check spatial compatibility
        if self.is_spatially_compatible(action, semantic_frame, world_state):
            score += 0.2

        # Apply policy-specific adjustments
        if self.policy == DecisionPolicy.OPTIMISTIC:
            score *= 1.1
        elif self.policy == DecisionPolicy.PESSIMISTIC:
            score *= 0.9

        # Apply uncertainty penalties
        understanding_confidence = understanding.get('confidence', 1.0)
        score *= understanding_confidence

        return min(score, 1.0)  # Cap at 1.0

    def check_action_preconditions(self, action: str, world_state: Dict[str, Any]) -> bool:
        """Check if action preconditions are met in the world state."""
        # Define preconditions for common actions
        preconditions = {
            'move_to': 'robot_exists',
            'grasp_object': 'object_visible',
            'place_object': 'holding_object'
        }

        required = preconditions.get(action, '')
        if required == 'robot_exists':
            return 'robot_position' in world_state
        elif required == 'object_visible':
            return len(world_state.get('objects', [])) > 0
        elif required == 'holding_object':
            return world_state.get('holding') is not None

        return True

    def is_spatially_compatible(self, action: str, semantic_frame: SemanticFrame,
                              world_state: Dict[str, Any]) -> bool:
        """Check if action is spatially compatible with command."""
        if action in ['move_to', 'grasp_object'] and 'grounded_location' in semantic_frame.arguments:
            target_pos = semantic_frame.arguments['grounded_location'].get('position')
            robot_pos = world_state.get('robot_position', [0, 0, 0])

            if target_pos and self.is_reachable(robot_pos, target_pos):
                return True

        return True

    def is_reachable(self, robot_pos: List[float], target_pos: List[float],
                    max_distance: float = 3.0) -> bool:
        """Check if target position is reachable."""
        distance = sum((a - b)**2 for a, b in zip(robot_pos, target_pos))**0.5
        return distance <= max_distance

    def select_best_action(self, action_scores: Dict[str, float]) -> str:
        """Select the best action based on scores and policy."""
        if not action_scores:
            return 'idle'

        # Sort actions by score
        sorted_actions = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)

        if self.policy == DecisionPolicy.OPTIMISTIC:
            # Choose highest scoring action
            return sorted_actions[0][0]
        elif self.policy == DecisionPolicy.PESSIMISTIC:
            # Add some randomness to avoid getting stuck
            if random.random() < 0.2:  # 20% chance to try alternative
                return sorted_actions[min(1, len(sorted_actions)-1)][0]
            else:
                return sorted_actions[0][0]
        else:  # BALANCED
            # Use softmax to select action probabilistically
            scores = np.array(list(action_scores.values()))
            probs = self.softmax(scores)
            actions = list(action_scores.keys())
            return np.random.choice(actions, p=probs)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax of array."""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)

    def generate_reasoning(self, semantic_frame: SemanticFrame, action: str,
                          world_state: Dict[str, Any]) -> str:
        """Generate natural language explanation for the decision."""
        if semantic_frame.action == 'move_to' and action == 'move_to':
            return f"Moving to the requested location as commanded."
        elif semantic_frame.action == 'grasp_object' and action == 'grasp_object':
            obj_name = semantic_frame.arguments.get('object', 'unknown object')
            return f"Grasping the {obj_name} as requested."
        elif action == 'ask_for_clarification':
            return f"The command is ambiguous, asking for clarification."
        else:
            return f"Taking action {action} to fulfill the command."

    def calculate_decision_confidence(self, action_scores: Dict[str, float],
                                   selected_action: str) -> float:
        """Calculate confidence in the decision."""
        selected_score = action_scores.get(selected_action, 0.0)

        # Calculate confidence based on score margin over alternatives
        other_scores = [score for action, score in action_scores.items()
                       if action != selected_action]
        if other_scores:
            max_other_score = max(other_scores)
            confidence = (selected_score - max_other_score + 1.0) / 2.0
        else:
            confidence = selected_score

        return min(confidence, 1.0)


class InteractiveDecisionMaker(LanguageGuidedDecisionMaker):
    def __init__(self):
        super().__init__()
        self.user_feedback_history = []

    def handle_ambiguous_command(self, command: str, alternatives: List[str],
                               world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ambiguous commands by asking for clarification."""
        print(f"Ambiguous command detected: {command}")
        print("Could you please clarify?")
        for i, alt in enumerate(alternatives):
            print(f"  {i+1}. {alt}")

        # In a real system, this would wait for user input
        # For simulation, we'll pick the first alternative
        selected_alternative = alternatives[0]

        # Update preference learning
        self.update_preferences_from_interaction(command, selected_alternative)

        # Reparse the clarified command
        parser = CommandParser()
        clarified_command = parser.parse_command(selected_alternative)

        # Return decision based on clarified command
        available_actions = ['move_to', 'grasp_object', 'place_object']  # Example actions
        return self.make_decision(
            {'semantic_frame': SemanticFrame(clarified_command['action'],
                                           clarified_command.get('arguments', {}),
                                           clarified_command.get('spatial_info', {})),
             'confidence': 0.9},
            world_state, available_actions
        )

    def update_preferences_from_interaction(self, original_command: str,
                                          clarified_command: str):
        """Update preference model based on user interaction."""
        if original_command not in self.preference_learning:
            self.preference_learning[original_command] = []

        self.preference_learning[original_command].append(clarified_command)
```

## 12.5 Implementation Example: Language-Guided Robot Control

```python
# language_control_example.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np


class LanguageControlNode(Node):
    def __init__(self):
        super().__init__('language_control_node')

        # Publishers
        self.response_pub = self.create_publisher(String, '/language_response', 10)
        self.action_pub = self.create_publisher(String, '/robot_action', 10)
        self.status_pub = self.create_publisher(String, '/language_status', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/user_command', self.command_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10)

        # Components
        self.command_parser = CommandParser()
        self.entity_grounding = EntityGrounding()
        self.spatial_grounding = SpatialGrounding()
        self.planner = LanguagePlanner()
        self.executor = TaskExecutor()
        self.uncertainty_handler = UncertaintyHandler()
        self.vision_language_integrator = VisionLanguageIntegrator()
        self.decision_maker = InteractiveDecisionMaker()

        # Internal state
        self.bridge = CvBridge()
        self.current_world_state = {
            'robot_position': [0.0, 0.0, 0.0],
            'robot_orientation': [0.0, 0.0, 0.0],
            'objects': [],
            'holding': None
        }
        self.last_image_features = None
        self.command_history = []

    def command_callback(self, msg):
        """Handle incoming language commands."""
        command_text = msg.data
        self.get_logger().info(f"Received command: {command_text}")

        # Update status
        status_msg = String()
        status_msg.data = f"Processing command: {command_text}"
        self.status_pub.publish(status_msg)

        try:
            # Parse the command
            parsed_command = self.command_parser.parse_command(command_text)

            # Check understanding confidence
            confidence = self.uncertainty_handler.calculate_command_confidence(parsed_command)
            self.get_logger().info(f"Command understanding confidence: {confidence:.2f}")

            if confidence < 0.5:
                # Handle low confidence with clarification request
                response = String()
                response.data = f"I'm not sure I understood. Could you rephrase '{command_text}'?"
                self.response_pub.publish(response)
                return

            # Ground entities in the current world state
            grounded_command = self.entity_grounding.ground_entities(
                parsed_command, self.current_world_state)

            # If we have visual features, integrate vision-language understanding
            if self.last_image_features is not None:
                vision_lang_understanding = self.vision_language_integrator.process_visual_language_input(
                    self.last_image_features, command_text)

                # Update world state with visual information
                self.update_world_state_from_vision(vision_lang_understanding)

            # Create semantic frame
            semantic_frame = SemanticFrame(
                grounded_command['action'],
                grounded_command.get('arguments', {}),
                grounded_command.get('spatial_info', {})
            )

            # Make decision about how to respond
            available_actions = ['execute_plan', 'request_clarification', 'report_status']
            decision = self.decision_maker.make_decision(
                {
                    'semantic_frame': semantic_frame,
                    'confidence': confidence
                },
                self.current_world_state,
                available_actions
            )

            # Execute the decision
            if decision['selected_action'] == 'execute_plan':
                success = self.executor.execute_command(semantic_frame, self.current_world_state)

                response = String()
                if success:
                    response.data = f"Successfully executed command: {command_text}"
                else:
                    response.data = f"Could not execute command: {command_text}"

                self.response_pub.publish(response)

            elif decision['selected_action'] == 'request_clarification':
                response = String()
                response.data = f"Could you clarify: {command_text}?"
                self.response_pub.publish(response)

            elif decision['selected_action'] == 'report_status':
                response = String()
                response.data = f"Current status: Robot at {self.current_world_state['robot_position']}"
                self.response_pub.publish(response)

            # Update command history
            self.command_history.append({
                'command': command_text,
                'parsed': parsed_command,
                'decision': decision,
                'timestamp': self.get_clock().now().to_msg()
            })

        except Exception as e:
            self.get_logger().error(f"Error processing command: {e}")
            response = String()
            response.data = f"Error processing command: {str(e)}"
            self.response_pub.publish(response)

    def image_callback(self, msg):
        """Process incoming images for vision-language integration."""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Extract visual features (simplified)
            # In practice, use a CNN or similar
            features = np.mean(cv_image, axis=(0, 1))  # Simplified feature extraction
            self.last_image_features = features

            # Update world state with detected objects
            self.update_world_state_from_image(cv_image)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def update_world_state_from_image(self, cv_image):
        """Update world state based on image processing."""
        # In practice, run object detection, segmentation, etc.
        # For this example, we'll simulate object detection
        simulated_objects = [
            {
                'name': 'red_cup',
                'position': [1.0, 0.5, 0.0],
                'type': 'cup',
                'color': 'red'
            },
            {
                'name': 'blue_box',
                'position': [-0.5, 1.0, 0.0],
                'type': 'box',
                'color': 'blue'
            }
        ]

        self.current_world_state['objects'] = simulated_objects

    def update_world_state_from_vision(self, vision_lang_understanding):
        """Update world state based on vision-language integration."""
        # Update with vision-language understanding
        vision_objects = vision_lang_understanding.get('visual_content', {}).get('objects', [])

        # Merge with existing objects
        existing_names = {obj['name'] for obj in self.current_world_state['objects']}
        for obj in vision_objects:
            if obj['name'] not in existing_names:
                self.current_world_state['objects'].append(obj)

    def get_current_world_state(self):
        """Get the current world state."""
        return self.current_world_state.copy()


def main(args=None):
    rclpy.init(args=args)
    language_node = LanguageControlNode()

    # Example: Add some simulated objects to the world
    language_node.current_world_state['objects'] = [
        {
            'name': 'red_cup',
            'position': [1.0, 0.5, 0.0],
            'type': 'cup',
            'color': 'red'
        },
        {
            'name': 'blue_box',
            'position': [-0.5, 1.0, 0.0],
            'type': 'box',
            'color': 'blue'
        }
    ]

    # Example: Simulate a command
    def test_command():
        """Test the language control system."""
        cmd_msg = String()
        cmd_msg.data = "pick up the red cup"
        language_node.command_callback(cmd_msg)

    # Run test after a short delay
    language_node.create_timer(3.0, test_command)

    try:
        rclpy.spin(language_node)
    except KeyboardInterrupt:
        pass
    finally:
        language_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## 12.6 Advanced Topics

### 12.6.1 Conversational AI for Robotics
Creating systems that can engage in natural conversations with users to better understand their intentions.

### 12.6.2 Learning from Language Instructions
Using language commands as training data to improve robot capabilities.

## 12.7 Best Practices

1. **Robust Parsing**: Handle various ways users might express the same intent
2. **Uncertainty Management**: Clearly communicate when the robot is uncertain
3. **Context Awareness**: Consider the current context when interpreting commands
4. **Safety First**: Ensure all language-guided actions are safe
5. **User Feedback**: Learn from user corrections and preferences

## Practical Exercise

### Exercise 12.1: Language-Guided Manipulation System
**Objective**: Create a system that accepts natural language commands to control a robot arm

1. Implement a command parser for manipulation tasks
2. Design entity grounding for objects in the environment
3. Create a planner that generates manipulation sequences from language
4. Implement uncertainty handling for ambiguous commands
5. Integrate with a simulated robot arm
6. Test with various natural language commands
7. Evaluate success rate and user satisfaction

**Deliverable**: Complete language-guided manipulation system with evaluation results.

## Summary

Week 12 covered language grounding and decision making in robotics, including natural language understanding, semantic parsing, and language-guided action planning. You learned to connect linguistic commands to physical actions and handle the uncertainties inherent in natural language communication. This capability enables more natural human-robot interaction.

[Next: Week 13 - System Integration & Deployment →](./week13-capstone-integration.md) | [Previous: Week 11 - Multimodal Perception ←](./week11-multimodal-perception.md)