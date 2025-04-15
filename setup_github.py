#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper script to setup and push to GitHub.
This script initializes a git repository, adds all files, commits them,
and provides instructions for pushing to GitHub.
"""

import os
import subprocess
import sys

def run_command(command, print_output=True):
    """Run a shell command and print its output."""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              text=True, capture_output=True)
        if print_output and result.stdout:
            print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error details: {e.stderr}")
        return False, e.stderr

def setup_git_repository():
    """Initialize git repository and make initial commit."""
    # Check if git is installed
    success, _ = run_command("git --version", print_output=False)
    if not success:
        print("Git is not installed or not in PATH. Please install git and try again.")
        return False
    
    # Check if .git directory already exists
    if os.path.exists(".git"):
        print("Git repository already initialized.")
    else:
        # Initialize git repository
        print("Initializing git repository...")
        success, _ = run_command("git init")
        if not success:
            return False
    
    # Add all files
    print("Adding files to git...")
    success, _ = run_command("git add .")
    if not success:
        return False
    
    # Commit changes
    print("Committing changes...")
    success, _ = run_command('git commit -m "Initial commit of facial expression recognition project"')
    if not success:
        # If commit fails, try setting up user config
        print("Commit failed. Trying to setup git user configuration...")
        run_command('git config --global user.email "you@example.com"')
        run_command('git config --global user.name "Your Name"')
        
        # Try committing again
        success, _ = run_command('git commit -m "Initial commit of facial expression recognition project"')
        if not success:
            return False
    
    return True

def github_instructions():
    """Print instructions for pushing to GitHub."""
    print("\n" + "="*80)
    print("GITHUB PUSH INSTRUCTIONS")
    print("="*80)
    print("\nTo push this repository to GitHub:")
    
    print("\n1. Create a new repository on GitHub at:")
    print("   https://github.com/new")
    
    print("\n2. Connect your local repository to GitHub:")
    print("   git remote add origin https://github.com/Danielmark001/facial_recognition_personal_project.git")
    
    print("\n3. Push your code to GitHub:")
    print("   git push -u origin main")
    
    print("\nIf your default branch is 'master' instead of 'main', use:")
    print("   git push -u origin master")
    
    print("\nIf you encounter authentication issues, you might need to:")
    print("   - Set up SSH keys for GitHub")
    print("   - Use a personal access token")
    print("   - Use GitHub CLI")
    
    print("\nFor more information, visit:")
    print("   https://docs.github.com/en/get-started/quickstart/create-a-repo")
    print("="*80)

def main():
    """Main function."""
    print("Setting up GitHub repository for the facial expression recognition project...")
    
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Setup git repository
    if setup_git_repository():
        print("\nGit repository setup successfully!")
        github_instructions()
    else:
        print("\nFailed to setup git repository. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
