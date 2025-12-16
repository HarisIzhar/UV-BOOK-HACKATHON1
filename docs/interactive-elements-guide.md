---
sidebar_position: 106
---

# Interactive Element Framework for MDX Content

## Overview

This guide provides a framework for incorporating interactive elements into the Physical AI & Humanoid Robotics book using MDX (Markdown + React). Interactive elements enhance the learning experience by allowing students to engage with concepts dynamically.

## 1. Interactive Element Types

### 1.1 Code Playground
Allow students to modify and run code examples directly in the browser.

```mdx
import CodePlayground from '@site/src/components/CodePlayground';

<CodePlayground
  code={`# ROS 2 Simple Publisher
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World'
        self.publisher.publish(msg)

def main():
    rclpy.init()
    publisher = MinimalPublisher()
    rclpy.spin(publisher)

if __name__ == '__main__':
    main()
`}
  language="python"
  title="ROS 2 Publisher Example"
/>
```

### 1.2 Simulation Viewer
Embed simulation environments for visualization.

```mdx
import SimulationViewer from '@site/src/components/SimulationViewer';

<SimulationViewer
  simulationId="robot-navigation"
  title="Robot Navigation Simulation"
  description="Watch a robot navigate through an environment using path planning algorithms"
/>
```

### 1.3 Concept Demonstrator
Visual demonstrations of complex concepts.

```mdx
import ConceptDemo from '@site/src/components/ConceptDemo';

<ConceptDemo
  type="perception-action-loop"
  title="Perception-Action Loop"
  description="Interactive demonstration of how perception, cognition, and action work together in robotics"
/>
```

## 2. Component Development

### 2.1 Creating Custom Components

Create interactive components in the `src/components/` directory:

```jsx
// src/components/CodePlayground.jsx
import React, { useState } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

const CodePlayground = ({ code, language, title }) => {
  const [currentCode, setCurrentCode] = useState(code);

  const executeCode = () => {
    // In a real implementation, this would execute the code
    // in a sandboxed environment
    console.log('Executing code:', currentCode);
  };

  return (
    <div className="code-playground">
      <h4>{title}</h4>
      <textarea
        value={currentCode}
        onChange={(e) => setCurrentCode(e.target.value)}
        rows="15"
        cols="80"
      />
      <button onClick={executeCode}>Run Code</button>
    </div>
  );
};

export default CodePlayground;
```

### 2.2 Docusaurus Configuration

Add components to Docusaurus configuration:

```js
// docusaurus.config.js (already configured for MDX)
module.exports = {
  // ... existing config
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/your-org/uvbook/tree/main/',
          // Enable MDX
          remarkPlugins: [
            // Any custom remark plugins
          ],
          rehypePlugins: [
            // Any custom rehype plugins
          ],
        },
        // ... other configs
      },
    ],
  ],
};
```

## 3. Sample Interactive Components

### 3.1 Architecture Diagram Component
```jsx
// src/components/ArchitectureDiagram.jsx
import React from 'react';

const ArchitectureDiagram = ({
  title = "System Architecture",
  diagram = "default"
}) => {
  const diagrams = {
    ros2: {
      title: "ROS 2 Architecture",
      content: (
        <div className="architecture-diagram">
          <div className="layer">Application</div>
          <div className="layer">ROS 2 Client Library</div>
          <div className="layer">ROS 2 Middleware</div>
        </div>
      )
    },
    perception: {
      title: "Perception System",
      content: (
        <div className="architecture-diagram">
          <div className="component">Sensors</div>
          <div className="arrow">→</div>
          <div className="component">Processing</div>
          <div className="arrow">→</div>
          <div className="component">Understanding</div>
        </div>
      )
    }
  };

  const currentDiagram = diagrams[diagram] || diagrams.default;

  return (
    <div className="architecture-container">
      <h4>{currentDiagram.title}</h4>
      <div className="diagram-content">
        {currentDiagram.content}
      </div>
    </div>
  );
};

export default ArchitectureDiagram;
```

### 3.2 Concept Visualization Component
```jsx
// src/components/ConceptVisualizer.jsx
import React, { useState } from 'react';

const ConceptVisualizer = ({ type = "default", title = "Concept Visualizer" }) => {
  const [speed, setSpeed] = useState(1);
  const [isRunning, setIsRunning] = useState(false);

  const conceptVisualizations = {
    kinematics: (
      <div className="kinematics-visualizer">
        <div className="robot-arm">
          <div className="joint" style={{transform: `rotate(${isRunning ? '45deg' : '0'})`}}>
            <div className="link"></div>
          </div>
        </div>
      </div>
    ),
    pathplanning: (
      <div className="path-planning-visualizer">
        <div className="environment">
          <div className="start">Start</div>
          <div className="goal">Goal</div>
          {isRunning && <div className="path">→ → →</div>}
        </div>
      </div>
    )
  };

  const visualization = conceptVisualizations[type] || conceptVisualizations.default;

  return (
    <div className="concept-visualizer">
      <h4>{title}</h4>
      <div className="controls">
        <button onClick={() => setIsRunning(!isRunning)}>
          {isRunning ? 'Stop' : 'Start'}
        </button>
        <label>
          Speed:
          <input
            type="range"
            min="0.5"
            max="3"
            step="0.5"
            value={speed}
            onChange={(e) => setSpeed(e.target.value)}
          />
          {speed}x
        </label>
      </div>
      <div className="visualization">
        {visualization}
      </div>
    </div>
  );
};

export default ConceptVisualizer;
```

## 4. Integration with Docusaurus

### 4.1 Import in MDX Files

Interactive components can be imported and used in any MDX document:

```mdx
---
sidebar_position: 1
---

# Week 2: ROS 2 Architecture

import ArchitectureDiagram from '@site/src/components/ArchitectureDiagram';
import ConceptVisualizer from '@site/src/components/ConceptVisualizer';

## ROS 2 Architecture

Understanding the ROS 2 architecture is crucial for building robust robotic systems.

<ArchitectureDiagram type="ros2" />

## Perception-Action Loop

<ConceptVisualizer type="kinematics" title="Robot Kinematics" />

The above visualization shows how joint angles affect the end-effector position.
```

### 4.2 Styling Interactive Elements

Add CSS for interactive elements:

```css
/* src/css/interactive.css */
.code-playground {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 16px;
  margin: 16px 0;
  background-color: #f9f9f9;
}

.architecture-diagram {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 16px;
}

.layer {
  background-color: #e3f2fd;
  padding: 8px 16px;
  border-radius: 4px;
  border: 1px solid #bbdefb;
}

.component {
  display: inline-block;
  background-color: #fff3e0;
  padding: 8px 12px;
  border-radius: 4px;
  border: 1px solid #ffe0b2;
}

.arrow {
  font-size: 1.5em;
  margin: 0 8px;
}

.concept-visualizer {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 16px;
  margin: 16px 0;
  text-align: center;
}

.controls {
  margin-bottom: 16px;
}

.kinematics-visualizer {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
}

.robot-arm {
  display: flex;
  align-items: center;
}

.joint {
  width: 20px;
  height: 20px;
  background-color: #1976d2;
  border-radius: 50%;
  transition: transform 0.3s ease;
}

.link {
  width: 100px;
  height: 10px;
  background-color: #1976d2;
  margin-left: 10px;
}
```

## 5. Best Practices

### 5.1 Performance Considerations
- Keep interactive elements lightweight
- Use memoization for complex components
- Implement proper cleanup for state and effects
- Consider lazy loading for heavy components

### 5.2 Accessibility
- Ensure all interactive elements are keyboard accessible
- Provide alternative text for visualizations
- Support screen readers with proper ARIA labels
- Maintain color contrast for visual elements

### 5.3 Educational Value
- Focus on enhancing understanding, not just entertainment
- Provide clear instructions for interacting with elements
- Include explanations of what students should observe
- Link interactive elements to learning objectives

## 6. Advanced Examples

### 6.1 Simulation Component with Props
```mdx
import SimulationComponent from '@site/src/components/SimulationComponent';

<SimulationComponent
  type="path-planning"
  environment="maze"
  algorithm="a-star"
  showGrid={true}
  animationSpeed={2}
  title="A* Path Planning Algorithm"
  description="Watch the A* algorithm find the shortest path through a maze"
/>
```

### 6.2 Exercise Component
```mdx
import ExerciseComponent from '@site/src/components/ExerciseComponent';

<ExerciseComponent
  title="Modify the Publisher"
  instructions="Change the message text from 'Hello World' to 'Robotics is Fun!' and adjust the timer to publish every 2 seconds"
  starterCode={`import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # Change this to 2.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World'  # Change this message
        self.publisher.publish(msg)
`}
  solution={`import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 2.0  # Changed to 2 seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Robotics is Fun!'  # Updated message
        self.publisher.publish(msg)
`}
/>
```

## 7. Implementation Roadmap

### Phase 1: Basic Components
- [ ] Code playground component
- [ ] Architecture diagram component
- [ ] Basic concept visualizer

### Phase 2: Advanced Components
- [ ] Simulation viewer
- [ ] Interactive exercises
- [ ] 3D visualizations

### Phase 3: Integration
- [ ] Consistent styling across components
- [ ] Performance optimization
- [ ] Accessibility improvements

This interactive element framework provides the foundation for creating engaging, educational content that enhances the learning experience in the Physical AI & Humanoid Robotics book.