"""
Databricks Setup Script - Python Version
This script configures Databricks CLI and uploads your files
"""

import os
import subprocess
import sys

WORKSPACE_URL = "https://adb-1889721579194859.19.azuredatabricks.net"
PROJECT_DIR = "/Users/abhishektanguturi/Projects/env/Assignment/INFS7203"
PYTHON_PATH = "/Users/abhishektanguturi/Projects/env/bin/python"
DATABRICKS_CLI = "/Users/abhishektanguturi/Projects/env/bin/databricks"

def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def run_command(cmd):
    """Run a shell command and return output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return result.stdout.strip()

def configure_databricks_cli():
    """Guide user through Databricks CLI configuration"""
    print_header("Databricks CLI Configuration")
    
    print(f"\nYour Databricks Workspace URL: {WORKSPACE_URL}\n")
    print("To create a Personal Access Token:")
    print(f"1. Open: {WORKSPACE_URL}")
    print("2. Click your user profile (top right) → Settings")
    print("3. Go to 'Developer' → 'Access tokens'")
    print("4. Click 'Generate new token'")
    print("5. Name: 'cli-token', Lifetime: 90 days")
    print("6. Click 'Generate' and COPY THE TOKEN")
    
    input("\nPress Enter when you have your token ready...")
    
    print("\nConfiguring Databricks CLI...")
    print(f"When prompted, enter:")
    print(f"  Host: {WORKSPACE_URL}")
    print(f"  Token: [paste your copied token]\n")
    
    os.system(f"{DATABRICKS_CLI} configure --token")
    print("\n✓ Databricks CLI configured!")

def find_data_files():
    """Locate train.csv and test_data.csv"""
    print_header("Locating Data Files")
    
    os.chdir(PROJECT_DIR)
    
    # Find train.csv
    train_file = None
    test_file = None
    
    possible_locations = [
        "train.csv",
        "INFS7203/train.csv",
        "../train.csv"
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            train_file = location
            print(f"✓ Found train.csv at: {location}")
            break
    
    if not train_file:
        train_file = input("Enter path to train.csv: ")
    
    # Find test_data.csv
    possible_test_locations = [
        "test_data.csv",
        "INFS7203/test_data.csv",
        "../test_data.csv"
    ]
    
    for location in possible_test_locations:
        if os.path.exists(location):
            test_file = location
            print(f"✓ Found test_data.csv at: {location}")
            break
    
    if not test_file:
        test_file = input("Enter path to test_data.csv: ")
    
    return train_file, test_file

def upload_files(train_file, test_file):
    """Upload data files to DBFS"""
    print_header("Uploading Files to DBFS")
    
    # Upload train.csv
    print("\nUploading train.csv...")
    cmd = f"{DATABRICKS_CLI} fs cp {train_file} dbfs:/FileStore/tables/train.csv --overwrite"
    result = run_command(cmd)
    if result is not None:
        print("✓ train.csv uploaded successfully")
    
    # Upload test_data.csv
    print("\nUploading test_data.csv...")
    cmd = f"{DATABRICKS_CLI} fs cp {test_file} dbfs:/FileStore/tables/test_data.csv --overwrite"
    result = run_command(cmd)
    if result is not None:
        print("✓ test_data.csv uploaded successfully")
    
    # Verify uploads
    print("\nVerifying uploads...")
    cmd = f"{DATABRICKS_CLI} fs ls dbfs:/FileStore/tables/"
    output = run_command(cmd)
    if output:
        print(output)

def upload_notebook():
    """Upload the notebook to workspace"""
    print_header("Uploading Notebook")
    
    notebook_path = f"{PROJECT_DIR}/databricks_notebook.py"
    username = os.environ.get('USER', 'user')
    
    print(f"\nUploading databricks_notebook.py...")
    cmd = f"{DATABRICKS_CLI} workspace import {notebook_path} /Users/{username}/ml_classification_pipeline --language PYTHON --format SOURCE --overwrite"
    result = run_command(cmd)
    
    if result is not None:
        print(f"✓ Notebook uploaded to: /Users/{username}/ml_classification_pipeline")
    
    return username

def print_next_steps(username):
    """Print final instructions"""
    print_header("✓ Setup Complete!")
    
    print(f"""
Next Steps:

1. Open your Databricks workspace:
   {WORKSPACE_URL}

2. Create a compute cluster:
   - Navigate to: Compute → Create Cluster
   - Cluster name: ml-classification-cluster
   - Runtime: 13.3 LTS ML or 14.3 LTS ML
   - Node type: Standard_DS3_v2 (4 cores, 14GB RAM)
   - Workers: 2-4 (with autoscaling enabled)
   - Click "Create Cluster"

3. Open your notebook:
   - Go to: Workspace → Users → {username} → ml_classification_pipeline
   
4. Run the pipeline:
   - Attach notebook to your cluster (dropdown at top)
   - Click "Run All" or run cells sequentially
   
5. Download results:
   - Results will be at: /dbfs/FileStore/output/s4845110.infs4203
   - Download via Data browser or CLI

Data files uploaded to:
  • /dbfs/FileStore/tables/train.csv
  • /dbfs/FileStore/tables/test_data.csv

Estimated runtime: 5-15 minutes (depending on cluster size)
Expected F1 Score: > 0.65
""")

def main():
    """Main execution function"""
    print_header("Databricks Setup & Configuration")
    print("This script will:")
    print("1. Configure Databricks CLI")
    print("2. Upload data files to DBFS")
    print("3. Upload the ML pipeline notebook")
    
    try:
        # Step 1: Configure CLI
        configure_databricks_cli()
        
        # Step 2: Find data files
        train_file, test_file = find_data_files()
        
        # Step 3: Upload files
        upload_files(train_file, test_file)
        
        # Step 4: Upload notebook
        username = upload_notebook()
        
        # Step 5: Print next steps
        print_next_steps(username)
        
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
