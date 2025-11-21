# Commands

This file describes the essentials for setting up a small dev project.

## Project setup

- create new virtual env: python3 -m venv .venv
- git init // git add . // git commit -m "Initial commit"
- git remote add origin https://github.com/{username}/{project_name}.git
- git branch -M main
- git push origin main
- added dependencies to requirements.txt
- pip install -r requirements.txt
- create README file

## Editable installs & PYTHONPATH

- For local development, you can install your project in "editable" mode:
	- `pip install -e .`
- If you use a `src/` layout, set your PYTHONPATH so imports work:
	- `export PYTHONPATH=$PWD/src:$PYTHONPATH`
- This lets you import your modules without installing every change.
