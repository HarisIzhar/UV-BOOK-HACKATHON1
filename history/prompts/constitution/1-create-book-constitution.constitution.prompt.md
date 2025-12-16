---
id: 1
title: Create Book Constitution
stage: constitution
date: 2025-12-15
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: Claude Code
command: /sp.constitution
labels: [constitution,book,physical-ai,humanoid-robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
---

## Prompt

You are defining the Constitution for an AI-driven book project.

Project Name:
"Physical AI & Humanoid Robotics: Embodied Intelligence in the Real World"

Purpose:
Create a university-level, capstone-quality technical book focused on Physical AI and Humanoid Robotics, bridging digital AI systems with real-world embodied intelligence.

Core Principles:
1. Accuracy First: All robotics, ROS 2, Gazebo, NVIDIA Isaac, and AI concepts must be technically correct and industry-aligned.
2. Learn-by-Building: Every concept should be reinforced with practical examples, code snippets, or simulations.
3. Simulation-to-Real Mindset: Emphasize sim-to-real transfer, hardware constraints, and deployment realities.
4. Open-Source Friendly: All examples must use open standards and tools (ROS 2, URDF, Gazebo, Isaac ROS).
5. Progressive Complexity: Begin with foundational Physical AI concepts and progress toward full humanoid autonomy.
6. Reproducibility: Steps, commands, and configurations should be reproducible on Ubuntu 22.04.
7. Student-Centered: Written for advanced AI/CS students transitioning into robotics.

Non-Goals:
- No focus on purely digital AI (chatbots-only systems).
- No low-level motor driver electronics unless required for understanding control flow.
- No marketing or vendor lock-in language.

Output Format:
This constitution will guide all future specs, plans, and tasks.

## Response snapshot

Created the constitution file for the Physical AI & Humanoid Robotics book project. The constitution includes seven core principles based on the user's requirements:

1. Accuracy First - Technical correctness and industry alignment
2. Learn-by-Building - Practical examples and simulations
3. Simulation-to-Real Mindset - Focus on sim-to-real transfer
4. Open-Source Friendly - Use of open standards and tools
5. Progressive Complexity - Logical progression from basics to advanced
6. Reproducibility - Ensuring examples work on target platform
7. Student-Centered - Focused on advanced students transitioning to robotics

Additional sections cover technology stack requirements, content standards, development workflow, and governance.

## Outcome

- ✅ Impact: Created initial constitution file with all required principles
- 🧪 Tests: N/A
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: Continue with spec creation using /sp.specify
- 🧠 Reflection: The constitution properly captures the educational and technical goals of the book project

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin creating the book specification