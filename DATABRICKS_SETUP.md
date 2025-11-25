# Azure Databricks ML Project Setup

## Prerequisites
- Azure CLI installed and logged in
- Azure subscription with permissions to create resources
- Data files: `train.csv` and `test_data.csv`

## Quick Start

### 1. Create Databricks Workspace

Run the deployment script:
```bash
cd /Users/abhishektanguturi/Projects/env/Assignment/INFS7203
chmod +x deploy_to_databricks.sh
./deploy_to_databricks.sh
```

Or manually create via Azure Portal:
1. Go to [Azure Portal](https://portal.azure.com)
2. Create Resource → Azure Databricks
3. Fill in details:
   - Workspace name: `databricks-ml-workspace`
   - Region: Choose nearest (e.g., East US)
   - Pricing tier: **Premium** (required for advanced features)

### 2. Upload Data to DBFS

**Option A: Using Databricks UI**
1. Open your Databricks workspace
2. Click **Data** in left sidebar
3. Click **Create Table**
4. Drop or browse for files:
   - Upload `train.csv`
   - Upload `test_data.csv`
5. Files will be at: `/dbfs/FileStore/tables/`

**Option B: Using Databricks CLI**
```bash
# Install Databricks CLI
pip install databricks-cli

# Configure
databricks configure --token

# Upload files
databricks fs cp train.csv dbfs:/FileStore/tables/train.csv
databricks fs cp test_data.csv dbfs:/FileStore/tables/test_data.csv
```

### 3. Import Notebook

1. Open Databricks workspace
2. Click **Workspace** → **Users** → [your email]
3. Click **⌄** → **Import**
4. Select `databricks_notebook.py`
5. Click **Import**

### 4. Create Compute Cluster

1. Click **Compute** in left sidebar
2. Click **Create Cluster**
3. Configure:
   - **Cluster name**: ml-classification-cluster
   - **Runtime**: 13.3 LTS or 14.3 LTS (includes ML libraries)
   - **Node type**: Standard_DS3_v2 or similar
   - **Workers**: 2-4 (adjust based on data size)
   - **Autoscaling**: Enabled

4. Click **Create Cluster**

### 5. Install Additional Libraries (if needed)

The Databricks ML runtime includes most libraries, but if needed:

1. Go to your cluster → **Libraries**
2. Click **Install New**
3. Select **PyPI**
4. Install:
   - `scikit-learn` (latest)
   - `imbalanced-learn` (if using SMOTE)

### 6. Run the Notebook

1. Open the imported notebook
2. Attach to your cluster (top-left dropdown)
3. **Verify file paths** in the Configuration cell:
   ```python
   TRAIN_DATA_PATH = "/dbfs/FileStore/tables/train.csv"
   TEST_DATA_PATH = "/dbfs/FileStore/tables/test_data.csv"
   ```
4. Click **Run All** or run cells sequentially

### 7. Download Results

After completion:
1. Results are saved to: `/dbfs/FileStore/output/s4845110.infs4203`
2. Download via:
   - **Databricks UI**: Data → browse to file → download
   - **CLI**: `databricks fs cp dbfs:/FileStore/output/s4845110.infs4203 ./`

## Notebook Structure

The notebook is organized into these sections:

1. **Import Libraries** - Load all required packages
2. **Configuration** - Set file paths and parameters
3. **Data Preprocessing** - Custom KNN imputation, outlier removal
4. **Model Training** - Voting classifier with RF, KNN, DT, NB
5. **Evaluation** - Cross-validation with threshold optimization
6. **Predictions** - Generate test predictions
7. **Output** - Create submission file

## Key Features

- **Custom KNN Imputation**: Handles missing values intelligently
- **Z-score Outlier Detection**: Removes data anomalies
- **Class Balancing**: Custom resampling for imbalanced data
- **Ensemble Learning**: Weighted voting of 4 classifiers
- **Threshold Optimization**: Maximizes F1 score
- **Cross-validation**: 5-fold stratified CV

## Expected Performance

- **F1 Score**: > 0.65
- **Accuracy**: ~0.85-0.90
- **Training Time**: 5-15 minutes (depends on cluster size)

## Troubleshooting

### File Not Found Error
```python
# Check files exist
import os
print(os.listdir("/dbfs/FileStore/tables/"))
```

### Memory Error
- Increase cluster size
- Reduce `n_estimators` in Random Forest
- Use fewer CV folds

### Library Import Error
```python
# Install missing library
%pip install library-name
dbutils.library.restartPython()
```

### Slow Performance
- Attach to a larger cluster
- Enable autoscaling
- Use Databricks Runtime 14.3 LTS ML

## Costs

Estimated Azure Databricks costs:
- **Standard_DS3_v2** (4 cores, 14GB RAM): ~$0.30-0.50/hour
- **Premium tier**: Additional $0.55/DBU
- **For this project**: Approximately $2-5 total

**Tip**: Stop cluster when not in use!

## Alternative: Local to Cloud Migration

If you want to run locally first:
```bash
# Run locally
cd /Users/abhishektanguturi/Projects/env/Assignment/INFS7203
python main.py

# Then upload results to cloud for storage
az storage blob upload \
  --account-name <storage-account> \
  --container-name results \
  --file s4845110.infs4203
```

## Support

For issues:
1. Check Databricks logs in notebook cells
2. Review Azure Databricks documentation
3. Check cluster event logs: Compute → [cluster] → Event Log

## Clean Up

To avoid charges:
```bash
# Delete resource group (removes everything)
az group delete --name rg-databricks-ml-project --yes

# Or just stop the cluster via UI
```
