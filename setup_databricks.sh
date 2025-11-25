#!/bin/bash

# Databricks Setup and File Upload Script
# This script will help you configure Databricks CLI and upload your files

set -e

echo "============================================"
echo "Databricks Configuration & File Upload"
echo "============================================"
echo ""

WORKSPACE_URL="https://adb-1889721579194859.19.azuredatabricks.net"

echo "Your Databricks Workspace URL: $WORKSPACE_URL"
echo ""
echo "To complete the setup, you need a Databricks personal access token."
echo ""
echo "Steps to create a token:"
echo "1. Open: $WORKSPACE_URL"
echo "2. Click your user profile (top right) → Settings"
echo "3. Go to 'Developer' → 'Access tokens'"
echo "4. Click 'Generate new token'"
echo "5. Give it a name (e.g., 'cli-token')"
echo "6. Set lifetime (e.g., 90 days)"
echo "7. Click 'Generate'"
echo "8. COPY THE TOKEN (you won't see it again!)"
echo ""
echo "Press Enter when you have your token ready..."
read

echo ""
echo "Configuring Databricks CLI..."
echo ""
echo "When prompted:"
echo "  - Host: $WORKSPACE_URL"
echo "  - Token: [paste your token]"
echo ""

/Users/abhishektanguturi/Projects/env/bin/databricks configure --token

echo ""
echo "✓ Databricks CLI configured!"
echo ""
echo "============================================"
echo "Checking for data files..."
echo "============================================"

cd /Users/abhishektanguturi/Projects/env/Assignment/INFS7203

if [ -f "train.csv" ]; then
    echo "✓ Found train.csv"
    TRAIN_FILE="train.csv"
elif [ -f "INFS7203/train.csv" ]; then
    echo "✓ Found train.csv in INFS7203 folder"
    TRAIN_FILE="INFS7203/train.csv"
else
    echo "⚠ train.csv not found in current directory"
    read -p "Enter path to train.csv: " TRAIN_FILE
fi

if [ -f "test_data.csv" ]; then
    echo "✓ Found test_data.csv"
    TEST_FILE="test_data.csv"
elif [ -f "INFS7203/test_data.csv" ]; then
    echo "✓ Found test_data.csv in INFS7203 folder"
    TEST_FILE="INFS7203/test_data.csv"
else
    echo "⚠ test_data.csv not found in current directory"
    read -p "Enter path to test_data.csv: " TEST_FILE
fi

echo ""
echo "============================================"
echo "Uploading files to DBFS..."
echo "============================================"

echo "Uploading train.csv..."
/Users/abhishektanguturi/Projects/env/bin/databricks fs cp "$TRAIN_FILE" dbfs:/FileStore/tables/train.csv --overwrite
echo "✓ train.csv uploaded"

echo "Uploading test_data.csv..."
/Users/abhishektanguturi/Projects/env/bin/databricks fs cp "$TEST_FILE" dbfs:/FileStore/tables/test_data.csv --overwrite
echo "✓ test_data.csv uploaded"

echo ""
echo "Verifying uploads..."
/Users/abhishektanguturi/Projects/env/bin/databricks fs ls dbfs:/FileStore/tables/

echo ""
echo "============================================"
echo "Uploading notebook..."
echo "============================================"

echo "Uploading databricks_notebook.py to workspace..."
/Users/abhishektanguturi/Projects/env/bin/databricks workspace import \
    databricks_notebook.py \
    /Users/$(whoami)/ml_classification_pipeline \
    --language PYTHON \
    --format SOURCE \
    --overwrite

echo "✓ Notebook uploaded to: /Users/$(whoami)/ml_classification_pipeline"

echo ""
echo "============================================"
echo "✓ Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Open: $WORKSPACE_URL"
echo "2. Create a cluster:"
echo "   - Go to Compute → Create Cluster"
echo "   - Runtime: 13.3 LTS ML or later"
echo "   - Node: Standard_DS3_v2"
echo "3. Open notebook: Workspace → Users → $(whoami) → ml_classification_pipeline"
echo "4. Attach to cluster and run!"
echo ""
echo "Data files are at:"
echo "  - /dbfs/FileStore/tables/train.csv"
echo "  - /dbfs/FileStore/tables/test_data.csv"
echo ""
