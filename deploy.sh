#!/bin/bash

# Define variables
REPO_DIR="/Users/macbookair/Documents/bayesian_dataclass"   # Replace with the path to your local repo
REMOTE_NAME="origin"
BRANCH_NAME="master"
SCRIPT_PATH="src/main.py"             # Path to your Python script

# Navigate to the repository directory
cd $REPO_DIR

# Check for uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo "You have uncommitted changes. Please commit or stash them before proceeding."
    exit 1
fi

# Upload changes to GitHub
git add .
git commit -m "Automated commit"  # Customize the commit message as needed
git push $REMOTE_NAME $BRANCH_NAME

# Pull latest changes on Oracle server
ssh username@oracle_server "cd /path/to/your/project && git pull $REMOTE_NAME $BRANCH_NAME"

# Run the Python script
ssh username@oracle_server "cd /path/to/your/project && python3 $SCRIPT_PATH"
