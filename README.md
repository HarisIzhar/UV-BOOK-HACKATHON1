# UVBook - Robotics, AI, and VLA Systems

A comprehensive guide to modern robotics and vision-language-action systems built with Docusaurus and Spec-Kit Plus.

## Overview

This book covers everything from foundational robotics concepts to advanced AI and Vision-Language-Action (VLA) systems. It follows a Spec-Driven Development approach using Claude Code and Spec-Kit Plus for systematic content creation.

## Structure

The book is organized into four main phases:

1. **Foundation & Architecture** - Core concepts and setup
2. **Core Robotics & Simulation** - Building robotic systems and simulators
3. **AI, VLA, and Capstone** - Advanced AI and VLA systems
4. **Review, Polish, Deploy** - Finalization and publication

## Development Workflow

This project uses Spec-Kit Plus for structured development:

- `/sp.specify` - Create or update chapter specifications
- `/sp.plan` - Generate implementation plans
- `/sp.tasks` - Create task breakdowns
- `/sp.implement` - Execute implementation plans
- `/sp.analyze` - Perform consistency analysis
- `/sp.phr` - Create prompt history records

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. Open [http://localhost:3000](http://localhost:3000) to view the book in development mode.

## Building for Production

To build the book for production:

```bash
npm run build
```

The static files will be generated in the `build` folder and can be deployed to any static hosting service.

## Deployment

This project is configured for deployment on Vercel. The `vercel.json` file is already configured for Docusaurus deployment.

## Contributing

1. Create a new branch for your chapter/module
2. Follow the Spec-Driven Development workflow
3. Ensure all validation checks pass
4. Submit a pull request for review

## License

This book is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.