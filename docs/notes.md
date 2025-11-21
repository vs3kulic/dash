# Commands

This file describes the essentials for setting up a small dev project.

## Project setup

- create new virtual env (conda): conda create -n dashboard
- activate conda env: conda activate dashboard
- git init // git add . // git commit -m "Initial commit"
- git remote add origin https://github.com/{username}/{project_name}.git
- git branch -M main
- git push origin main
- added dependencies to requirements.txt
- pip install -r requirements.txt
- pip install -e .
- create README file

## Editable installs & PYTHONPATH

- For local development, we install the project in "editable" mode:
	- `pip install -e .`
- If we use a `src/` layout, set your PYTHONPATH so imports work:
	- `export PYTHONPATH=$PWD/src:$PYTHONPATH`
- This lets us import modules without installing every change.
