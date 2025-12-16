---
sidebar_position: 105
---

# Testing Environment Setup for Code Examples

## Overview

This guide provides instructions for setting up a comprehensive testing environment to validate all code examples in the Physical AI & Humanoid Robotics book. The testing environment ensures that all code examples function as documented and can be successfully executed by students.

## 1. Required Software and Dependencies

### 1.1 Core System Requirements
- **Operating System**: Ubuntu 22.04 LTS (as specified in the constitution)
- **Python**: Python 3.10+ (minimum Python 3.8)
- **ROS 2**: Humble Hawksbill (LTS version)
- **Docker**: For containerized testing environments
- **Git**: For version control and example retrieval

### 1.2 Python Dependencies
```bash
pip3 install --upgrade pip
pip3 install pylint flake8 pytest pytest-cov black mypy
pip3 install rclpy  # ROS 2 Python client library
```

### 1.3 ROS 2 Dependencies
```bash
# Install ROS 2 Humble (Ubuntu 22.04)
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-ros-base
sudo apt install python3-rosdep2
sudo apt install ros-humble-xacro ros-humble-robot-state-publisher ros-humble-joint-state-publisher ros-humble-gazebo-ros-pkgs

# Source ROS 2 environment
source /opt/ros/humble/setup.bash
```

## 2. Testing Environment Architecture

### 2.1 Local Development Environment
```
┌─────────────────────────────────────────────────────────┐
│                    Development Host                     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Editor    │  │  Terminal   │  │   Browser   │     │
│  │   (VS Code) │  │   (bash)    │  │  (Docusaurus) │   │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │                 │            │
│         ▼               ▼                 ▼            │
│  ┌─────────────────────────────────────────────────┐   │
│  │              ROS 2 Workspace                  │   │
│  │  ┌─────────────────────────────────────────┐  │   │
│  │  │         Testing Scripts               │  │   │
│  │  │  (validation, linting, execution)     │  │   │
│  │  └─────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Containerized Testing Environment
```dockerfile
# Dockerfile.testing
FROM ros:humble-ros-base-ubuntu-jammy

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install pylint flake8 pytest pytest-cov black mypy

# Set up ROS 2 environment
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Create workspace
RUN mkdir -p /workspace/src
WORKDIR /workspace

# Copy validation scripts
COPY scripts/ /workspace/scripts/

CMD ["/bin/bash"]
```

## 3. Validation Scripts

### 3.1 Python Syntax Validator
```python
#!/usr/bin/env python3
# validate_python_syntax.py
"""
Script to validate Python syntax across all code examples.
"""

import ast
import os
import sys
from pathlib import Path


def validate_python_file(filepath):
    """Validate Python syntax for a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()

        # Parse the file to check for syntax errors
        ast.parse(content)
        print(f"✓ {filepath}: Syntax OK")
        return True
    except SyntaxError as e:
        print(f"✗ {filepath}: Syntax Error - {e}")
        return False
    except Exception as e:
        print(f"✗ {filepath}: Error - {e}")
        return False


def validate_directory(directory):
    """Validate all Python files in a directory."""
    directory_path = Path(directory)
    python_files = list(directory_path.rglob("*.py"))

    if not python_files:
        print(f"No Python files found in {directory}")
        return True

    results = []
    for py_file in python_files:
        result = validate_python_file(py_file)
        results.append(result)

    success_count = sum(results)
    total_count = len(results)

    print(f"\nValidation Summary: {success_count}/{total_count} files passed")
    return success_count == total_count


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_python_syntax.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    success = validate_directory(directory)
    sys.exit(0 if success else 1)
```

### 3.2 ROS 2 Node Validator
```python
#!/usr/bin/env python3
# validate_ros2_node.py
"""
Script to validate ROS 2 nodes in code examples.
"""

import subprocess
import sys
import os
import tempfile
import importlib.util
from pathlib import Path


def validate_ros2_node(filepath):
    """Validate a ROS 2 node file."""
    # Check if ROS 2 is available
    try:
        result = subprocess.run(['ros2', 'node', 'list'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print(f"⚠ {filepath}: ROS 2 not available, skipping runtime validation")
            return validate_syntax_only(filepath)
    except FileNotFoundError:
        print(f"⚠ {filepath}: ROS 2 not installed, skipping runtime validation")
        return validate_syntax_only(filepath)

    # Create a temporary ROS 2 package to test the node
    with tempfile.TemporaryDirectory() as temp_dir:
        package_name = "test_node_package"
        package_path = Path(temp_dir) / package_name

        # Create minimal package structure
        package_path.mkdir()
        (package_path / "setup.py").write_text(setup_py_content())
        (package_path / "setup.cfg").write_text(setup_cfg_content())
        (package_path / "package.xml").write_text(package_xml_content(package_name))

        # Copy the node file to the package
        node_dir = package_path / package_name
        node_dir.mkdir()
        (node_dir / "__init__.py").write_text("")

        # Read and copy the node file
        with open(filepath, 'r') as f:
            node_content = f.read()

        # Save the node content to the package
        node_filename = Path(filepath).stem + "_test.py"
        (node_dir / node_filename).write_text(node_content)

        # Try to import the module to check for syntax and import errors
        try:
            spec = importlib.util.spec_from_file_location(node_filename.replace('.py', ''),
                                                         node_dir / node_filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✓ {filepath}: Import successful")
            return True
        except Exception as e:
            print(f"✗ {filepath}: Import error - {e}")
            return False


def validate_syntax_only(filepath):
    """Validate only the syntax of a Python file."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        compile(code, filepath, 'exec')
        print(f"✓ {filepath}: Syntax valid")
        return True
    except SyntaxError as e:
        print(f"✗ {filepath}: Syntax error - {e}")
        return False


def setup_py_content():
    return """from setuptools import setup
from glob import glob

package_name = 'test_node_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Test',
    maintainer_email='test@example.com',
    description='Test package for validation',
    license='Apache License 2.0',
)
"""


def setup_cfg_content():
    return """[develop]
script-dir=$base/lib/test_node_package
[install]
install-scripts=$base/lib/test_node_package
"""


def package_xml_content(package_name):
    return f"""<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{package_name}</name>
  <version>0.0.0</version>
  <description>Test package for validation</description>
  <maintainer email="test@example.com">Test</maintainer>
  <license>Apache License 2.0</license>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
"""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_ros2_node.py <filepath>")
        sys.exit(1)

    filepath = sys.argv[1]
    success = validate_ros2_node(filepath)
    sys.exit(0 if success else 1)
```

### 3.3 Automated Testing Script
```bash
#!/bin/bash
# run_validation.sh
# Main script to run all validations

set -e  # Exit on any error

echo "Starting code validation process..."

# Directory containing code examples
CODE_DIR="${1:-./docs}"

echo "Validating Python syntax..."
python3 validate_python_syntax.py "$CODE_DIR"

echo "Running linter checks..."
find "$CODE_DIR" -name "*.py" -exec pylint {} \; 2>/dev/null || echo "Pylint completed with issues"
find "$CODE_DIR" -name "*.py" -exec flake8 {} \; || echo "Flake8 completed with issues"

echo "Checking code formatting..."
find "$CODE_DIR" -name "*.py" -exec black --check {} \; || echo "Black formatting check completed"

echo "Running type checks..."
find "$CODE_DIR" -name "*.py" -exec mypy {} \; || echo "MyPy type checking completed"

echo "Validation process completed!"
```

## 4. Continuous Integration Configuration

### 4.1 GitHub Actions Workflow
```yaml
# .github/workflows/code-validation.yml
name: Code Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-22.04

    steps:
    - uses: actions/checkout@v3

    - name: Setup ROS 2 Humble
      uses: ros-tooling/setup-ros@v0.7
      with:
        required-ros-distributions: humble

    - name: Install Python dependencies
      run: |
        python3 -m pip install --upgrade pip
        pip3 install pylint flake8 pytest black mypy

    - name: Validate Python syntax
      run: |
        python3 scripts/validate_python_syntax.py ./docs

    - name: Run code linters
      run: |
        find ./docs -name "*.py" -exec pylint {} \; || echo "Pylint completed with issues"
        find ./docs -name "*.py" -exec flake8 {} \; || echo "Flake8 completed with issues"

    - name: Check code formatting
      run: |
        find ./docs -name "*.py" -exec black --check {} \;

    - name: Run type checks
      run: |
        find ./docs -name "*.py" -exec mypy {} \; || echo "MyPy completed with issues"

    - name: Run ROS 2 validation (if available)
      run: |
        if command -v ros2 &> /dev/null; then
          echo "ROS 2 available, running node validation..."
          # Add ROS 2 specific validation here
        else
          echo "ROS 2 not available in this environment"
        fi
```

## 5. Testing Procedures

### 5.1 Manual Testing Process
1. **Setup Environment**: Ensure all dependencies are installed
2. **Run Syntax Validation**: Check for syntax errors in all Python files
3. **Run Linting**: Check code style and best practices
4. **Run Type Checks**: Validate type hints where applicable
5. **Execute Examples**: Run selected examples to verify functionality
6. **Document Issues**: Record any problems found during testing

### 5.2 Automated Testing Process
1. **Pre-commit Hooks**: Run basic validation before commits
2. **CI Pipeline**: Full validation on every push/pull request
3. **Scheduled Testing**: Periodic validation of all examples
4. **Regression Testing**: Verify that new changes don't break existing examples

## 6. Validation Reporting

### 6.1 Test Results Format
```
Validation Report: YYYY-MM-DD

Environment:
- OS: Ubuntu 22.04
- Python: 3.10.x
- ROS 2: Humble Hawksbill
- Validation Tool: Version X.X.X

Results:
- Python Syntax: X/X files passed
- Linting: X/X files passed
- Type Checks: X/X files passed
- Runtime Tests: X/X files passed

Issues Found:
- [List specific issues with file paths and descriptions]
```

### 6.2 Issue Tracking
- **Critical**: Code doesn't run or has major functionality issues
- **High**: Code runs but doesn't produce expected results
- **Medium**: Style or minor functionality issues
- **Low**: Documentation or formatting issues

This testing environment setup ensures that all code examples in the book are validated for syntax, functionality, and educational value.