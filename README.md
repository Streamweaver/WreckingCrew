# Ovierview

This repository contains an experimental codebase for bulding different multi-agent AI services to accomplish different tasks.

Each Crew is maintained under it's own subdirectory and shoudl be run as a separate project.

## Core Technologies

- **CrewAI**: A framework for orchestrating and coordinating multiple AI agents to work collaboratively on tasks.


## Setup Notes

### Python Poetry

These sub-projects use python poetry dependency management and virtual environment manager.  

From the sub-project folder
> poetry install --no-root

To activate the appropriate environment.
> poetry shell

To run a specifid file, cd into the appropriate subdirectory.
> poetry run <pythonfilename>

## License
This project is licensed under the Apache License 2.0, an open-source license. See the LICENSE file for more details.


