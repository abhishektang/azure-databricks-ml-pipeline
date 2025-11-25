#!/bin/bash

# Run the ML pipeline notebook on Databricks
CLUSTER_ID="1124-131544-a3kfqmqq"
NOTEBOOK_PATH="/Users/t.abhishek45699@yahoo.com/ml_classification_pipeline"
DATABRICKS_CLI="/Users/abhishektanguturi/Projects/env/bin/databricks"

echo "Waiting for cluster to be ready..."

# Wait for cluster to be in RUNNING state
while true; do
    STATE=$($DATABRICKS_CLI clusters get --cluster-id $CLUSTER_ID | grep '"state"' | head -1 | cut -d'"' -f4)
    echo "Current state: $STATE"
    
    if [ "$STATE" = "RUNNING" ]; then
        echo "✓ Cluster is ready!"
        break
    elif [ "$STATE" = "ERROR" ] || [ "$STATE" = "TERMINATED" ]; then
        echo "✗ Cluster failed to start. State: $STATE"
        exit 1
    fi
    
    sleep 10
done

echo ""
echo "Running notebook: $NOTEBOOK_PATH"
echo "This will take 10-15 minutes..."
echo ""

# Run the notebook
$DATABRICKS_CLI runs submit --json "{
  \"run_name\": \"ML Classification Pipeline\",
  \"existing_cluster_id\": \"$CLUSTER_ID\",
  \"notebook_task\": {
    \"notebook_path\": \"$NOTEBOOK_PATH\",
    \"source\": \"WORKSPACE\"
  },
  \"timeout_seconds\": 3600
}"

echo ""
echo "Notebook execution started!"
echo "Check progress at: https://adb-1889721579194859.19.azuredatabricks.net"
