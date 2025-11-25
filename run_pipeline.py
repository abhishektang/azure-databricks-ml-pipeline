#!/usr/bin/env python
"""
Run the complete ETL → ML pipeline on Azure Databricks
"""

import json
import time
import subprocess
import sys

# Configuration
CLUSTER_ID = "1124-132744-47atqg0y"
DATABRICKS_CLI = "/Users/abhishektanguturi/Projects/env/bin/databricks"
OUTPUT_DIR = "/Users/abhishektanguturi/Projects/env/Assignment/INFS7203"
WORKSPACE_URL = "https://adb-1889721579194859.19.azuredatabricks.net"

def run_databricks_command(cmd):
    """Execute databricks CLI command"""
    # Disable automatic execution of newer version to use Python-based CLI
    env = subprocess.os.environ.copy()
    env['DATABRICKS_CLI_DO_NOT_EXECUTE_NEWER_VERSION'] = '1'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    return result.stdout, result.stderr, result.returncode

def submit_job(notebook_path, run_name):
    """Submit a Databricks job"""
    job_config = {
        "run_name": run_name,
        "existing_cluster_id": CLUSTER_ID,
        "notebook_task": {
            "notebook_path": notebook_path,
            "source": "WORKSPACE"
        },
        "timeout_seconds": 3600
    }
    
    cmd = f'{DATABRICKS_CLI} runs submit --json \'{json.dumps(job_config)}\''
    stdout, stderr, code = run_databricks_command(cmd)
    
    if code != 0:
        print(f"✗ Failed to submit job: {stderr}")
        return None
    
    try:
        response = json.loads(stdout)
        return response.get('run_id')
    except:
        print(f"✗ Failed to parse response: {stdout}")
        return None

def monitor_job(run_id, job_name):
    """Monitor job execution"""
    print(f"  Monitoring {job_name} (Run ID: {run_id})...")
    
    while True:
        cmd = f'{DATABRICKS_CLI} runs get --run-id {run_id}'
        stdout, stderr, code = run_databricks_command(cmd)
        
        if code != 0:
            print(f"  ⚠ Error checking status: {stderr}")
            time.sleep(10)
            continue
        
        try:
            run_info = json.loads(stdout)
            state = run_info.get('state', {})
            life_cycle_state = state.get('life_cycle_state', 'UNKNOWN')
            result_state = state.get('result_state', '')
            state_message = state.get('state_message', '')
            
            print(f"  [{time.strftime('%H:%M:%S')}] Status: {life_cycle_state}", end='')
            if state_message:
                print(f" - {state_message}", end='')
            print()
            
            if life_cycle_state == 'TERMINATED':
                if result_state == 'SUCCESS':
                    print(f"✓ {job_name} completed successfully!")
                    return True
                else:
                    print(f"✗ {job_name} failed: {result_state}")
                    print(f"  Check logs at: {WORKSPACE_URL}")
                    return False
            
        except Exception as e:
            print(f"  ⚠ Error parsing status: {e}")
        
        time.sleep(20)

def download_results():
    """Download results from DBFS"""
    print("\nDownloading results...")
    cmd = f'{DATABRICKS_CLI} fs cp dbfs:/FileStore/output/s4845110.infs4203 {OUTPUT_DIR}/s4845110.infs4203 --overwrite'
    stdout, stderr, code = run_databricks_command(cmd)
    
    if code == 0:
        print(f"✓ Results saved to: {OUTPUT_DIR}/s4845110.infs4203")
        
        # Show preview
        try:
            with open(f"{OUTPUT_DIR}/s4845110.infs4203", 'r') as f:
                lines = f.readlines()
            print("\nPreview:")
            for line in lines[:5]:
                print(f"  {line.strip()}")
            print("  ...")
            for line in lines[-2:]:
                print(f"  {line.strip()}")
        except:
            pass
    else:
        print(f"⚠ Failed to download results: {stderr}")

def main():
    """Run the complete pipeline"""
    print("=" * 60)
    print("COMPLETE DATA PIPELINE EXECUTION")
    print("=" * 60)
    print("\nPipeline Steps:")
    print("  1. ETL Pipeline (Extract → Transform → Load)")
    print("  2. ML Classification Pipeline (Train → Predict)")
    print()
    
    # Step 1: Run ETL Pipeline
    print("STEP 1: Running ETL Pipeline...")
    print("-" * 40)
    
    etl_run_id = submit_job(
        "/Users/t.abhishek45699@yahoo.com/etl_pipeline",
        "ETL Pipeline - Complete Run"
    )
    
    if not etl_run_id:
        print("✗ Failed to submit ETL pipeline")
        sys.exit(1)
    
    print(f"✓ ETL Pipeline submitted (Run ID: {etl_run_id})")
    
    if not monitor_job(etl_run_id, "ETL Pipeline"):
        sys.exit(1)
    
    print()
    
    # Step 2: Run ML Pipeline
    print("STEP 2: Running ML Classification Pipeline...")
    print("-" * 40)
    
    ml_run_id = submit_job(
        "/Users/t.abhishek45699@yahoo.com/ml_classification_pipeline",
        "ML Classification Pipeline - Complete Run"
    )
    
    if not ml_run_id:
        print("✗ Failed to submit ML pipeline")
        sys.exit(1)
    
    print(f"✓ ML Pipeline submitted (Run ID: {ml_run_id})")
    
    if not monitor_job(ml_run_id, "ML Pipeline"):
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)
    print()
    
    # Download results
    download_results()
    
    print()
    print("Pipeline Summary:")
    print(f"  ETL Run ID: {etl_run_id}")
    print(f"  ML Run ID: {ml_run_id}")
    print(f"  Output: {OUTPUT_DIR}/s4845110.infs4203")
    print(f"  Workspace: {WORKSPACE_URL}")

if __name__ == "__main__":
    main()
