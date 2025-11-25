#!/bin/bash

# Azure Databricks Deployment Script
# Author: Abhishek Tanguturi

set -e

echo "======================================"
echo "Azure Databricks Deployment Setup"
echo "======================================"

# Check if Azure CLI is logged in
if ! az account show &> /dev/null; then
    echo "Error: Not logged in to Azure. Please run 'az login' first."
    exit 1
fi

echo "✓ Azure CLI is authenticated"

# Get subscription info
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
SUBSCRIPTION_NAME=$(az account show --query name -o tsv)

echo "Current Subscription: $SUBSCRIPTION_NAME"
echo "Subscription ID: $SUBSCRIPTION_ID"
echo ""

# Prompt for resource group
read -p "Enter Resource Group name (or press Enter to create new): " RESOURCE_GROUP
if [ -z "$RESOURCE_GROUP" ]; then
    RESOURCE_GROUP="rg-databricks-ml-project"
    echo "Using default: $RESOURCE_GROUP"
fi

# Check if resource group exists
if ! az group show --name "$RESOURCE_GROUP" &> /dev/null; then
    echo "Resource group '$RESOURCE_GROUP' does not exist."
    read -p "Enter location (e.g., eastus, westus2): " LOCATION
    if [ -z "$LOCATION" ]; then
        LOCATION="eastus"
    fi
    echo "Creating resource group '$RESOURCE_GROUP' in '$LOCATION'..."
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
    echo "✓ Resource group created"
else
    echo "✓ Resource group '$RESOURCE_GROUP' exists"
    LOCATION=$(az group show --name "$RESOURCE_GROUP" --query location -o tsv)
fi

# Prompt for workspace name
read -p "Enter Databricks workspace name (or press Enter for default): " WORKSPACE_NAME
if [ -z "$WORKSPACE_NAME" ]; then
    WORKSPACE_NAME="databricks-ml-workspace"
    echo "Using default: $WORKSPACE_NAME"
fi

# Check if workspace exists
if ! az databricks workspace show --resource-group "$RESOURCE_GROUP" --name "$WORKSPACE_NAME" &> /dev/null; then
    echo "Creating Databricks workspace '$WORKSPACE_NAME'..."
    az databricks workspace create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$WORKSPACE_NAME" \
        --location "$LOCATION" \
        --sku premium
    
    echo "✓ Databricks workspace created successfully"
else
    echo "✓ Databricks workspace '$WORKSPACE_NAME' already exists"
fi

# Get workspace URL
WORKSPACE_URL=$(az databricks workspace show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$WORKSPACE_NAME" \
    --query workspaceUrl -o tsv)

echo ""
echo "======================================"
echo "Deployment Summary"
echo "======================================"
echo "Resource Group: $RESOURCE_GROUP"
echo "Workspace Name: $WORKSPACE_NAME"
echo "Location: $LOCATION"
echo "Workspace URL: https://$WORKSPACE_URL"
echo ""
echo "======================================"
echo "Next Steps:"
echo "======================================"
echo "1. Upload your data files to DBFS:"
echo "   - train.csv → /dbfs/FileStore/tables/train.csv"
echo "   - test_data.csv → /dbfs/FileStore/tables/test_data.csv"
echo ""
echo "2. Import the notebook:"
echo "   - Open Databricks workspace: https://$WORKSPACE_URL"
echo "   - Go to Workspace → Import"
echo "   - Upload: databricks_notebook.py"
echo ""
echo "3. Create a cluster:"
echo "   - Runtime: 13.3 LTS or later"
echo "   - Python: 3.10+"
echo "   - Libraries: scikit-learn, pandas, numpy, scipy"
echo ""
echo "4. Run the notebook!"
echo "======================================"
