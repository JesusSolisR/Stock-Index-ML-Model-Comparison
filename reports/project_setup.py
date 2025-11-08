import sys
import os

def setup_project_root():
    """
    Finds the project root by going one level up from this file's directory
    (which is /report) and adds the root (/project) to sys.path.
    """
    current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
        print(f"Added project root to path")

setup_project_root()