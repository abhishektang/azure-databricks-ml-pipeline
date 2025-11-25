# Quick Start Guide - Azure Databricks Setup

## 🎯 Current Status
✅ Azure Databricks workspace created  
✅ Databricks CLI installed  
✅ Data files located  

## 📍 Your Workspace
**URL**: https://adb-1889721579194859.19.azuredatabricks.net  
**Resource Group**: Data_Mining_Project  
**Location**: East US  

---

## 🚀 Option 1: Manual Upload (Easiest - Recommended)

### Step 1: Access Your Workspace
1. Open: https://adb-1889721579194859.19.azuredatabricks.net
2. Sign in with your Azure account (t.abhishek45699@yahoo.com)

### Step 2: Upload Data Files via UI
1. In Databricks, click **Data** in the left sidebar
2. Click **Create Table** or **Add Data**
3. Click **Upload File**
4. Upload these files from: `/Users/abhishektanguturi/Projects/env/Assignment/INFS7203/`
   - ✅ `train.csv` 
   - ✅ `test_data.csv`
5. They will be uploaded to: `dbfs:/FileStore/tables/`

### Step 3: Import Notebook via UI
1. Click **Workspace** in the left sidebar
2. Navigate to **Users** → **[your email]**
3. Click the **⌄** dropdown → **Import**
4. Click **Browse** and select: `databricks_notebook.py`
5. The notebook will be imported as `ml_classification_pipeline`

### Step 4: Create Compute Cluster
1. Click **Compute** in the left sidebar
2. Click **Create Compute**
3. Configure:
   ```
   Cluster name: ml-classification-cluster
   Policy: Unrestricted
   Runtime: 13.3 LTS ML (or 14.3 LTS ML)
   Node type: Standard_DS3_v2
   Workers: Min 2, Max 4 (autoscaling enabled)
   ```
4. Click **Create Compute**
5. Wait 3-5 minutes for cluster to start

### Step 5: Run Your Pipeline
1. Navigate to **Workspace** → **Users** → [your email] → **ml_classification_pipeline**
2. In the notebook, click the **cluster dropdown** (top) → select your cluster
3. Click **Run all** or run cells one by one
4. Monitor progress in each cell

### Step 6: Download Results
After completion (~10-15 minutes):
1. Click **Data** in sidebar
2. Navigate to **DBFS** → **FileStore** → **output**
3. Find `s4845110.infs4203`
4. Click **⋮** → **Download**

---

## 🛠 Option 2: Using Databricks CLI

### Get Access Token
1. Open your workspace: https://adb-1889721579194859.19.azuredatabricks.net
2. Click your **profile icon** (top right) → **Settings**
3. Click **Developer** → **Access tokens**
4. Click **Generate new token**
5. Name: `cli-token`, Lifetime: `90 days`
6. Click **Generate**
7. **COPY THE TOKEN** (save it somewhere safe!)

### Configure CLI
```bash
cd /Users/abhishektanguturi/Projects/env/Assignment/INFS7203

# Configure with your token
/Users/abhishektanguturi/Projects/env/bin/databricks configure --token

# Enter when prompted:
# Host: https://adb-1889721579194859.19.azuredatabricks.net
# Token: [paste your token]
```

### Upload Files
```bash
# Upload data files
/Users/abhishektanguturi/Projects/env/bin/databricks fs cp train.csv dbfs:/FileStore/tables/train.csv --overwrite

/Users/abhishektanguturi/Projects/env/bin/databricks fs cp test_data.csv dbfs:/FileStore/tables/test_data.csv --overwrite

# Verify uploads
/Users/abhishektanguturi/Projects/env/bin/databricks fs ls dbfs:/FileStore/tables/
```

### Import Notebook
```bash
# Upload notebook to workspace
/Users/abhishektanguturi/Projects/env/bin/databricks workspace import \
    databricks_notebook.py \
    /Users/$(whoami)/ml_classification_pipeline \
    --language PYTHON \
    --format SOURCE \
    --overwrite
```

---

## 📊 Cluster Recommendations

### For Small Dataset (< 100MB)
- **Node**: Standard_DS3_v2 (4 cores, 14GB)
- **Workers**: 2
- **Cost**: ~$0.40/hour

### For Medium Dataset (100MB - 1GB)
- **Node**: Standard_DS4_v2 (8 cores, 28GB)
- **Workers**: 2-4
- **Cost**: ~$0.80/hour

### Cost Saving Tips
- ✅ Enable autoscaling
- ✅ Set auto-termination to 30 minutes
- ✅ Stop cluster when not in use
- ✅ Use spot instances (if available)

---

## 🔍 Troubleshooting

### File Path Issues
If you see "File not found" errors in the notebook:
1. Verify files are uploaded: Data → FileStore → tables
2. Check paths in notebook match:
   ```python
   TRAIN_DATA_PATH = "/dbfs/FileStore/tables/train.csv"
   TEST_DATA_PATH = "/dbfs/FileStore/tables/test_data.csv"
   ```

### Memory Errors
- Increase cluster size
- Reduce `n_estimators` in Random Forest (line ~280)
- Use fewer CV folds

### Cluster Won't Start
- Check quota limits in Azure
- Try a different region
- Use smaller node type

### Slow Performance
- Enable autoscaling
- Increase worker count
- Use ML Runtime (includes optimized libraries)

---

## 📈 Expected Results

After running the notebook:
- **F1 Score**: > 0.65
- **Accuracy**: ~0.85-0.90
- **Runtime**: 10-15 minutes
- **Output**: `/dbfs/FileStore/output/s4845110.infs4203`

---

## 🧹 Clean Up (When Done)

### Stop Cluster
- Compute → [your cluster] → **Terminate**

### Delete Workspace (if no longer needed)
```bash
az databricks workspace delete \
    --resource-group Data_Mining_Project \
    --name Project_Deployment \
    --yes
```

### Delete Resource Group (removes everything)
```bash
az group delete --name Data_Mining_Project --yes
```

---

## 📞 Support

### Databricks Documentation
- https://docs.databricks.com/

### Azure Support
- https://portal.azure.com → Support

### Check Logs
- In notebook: View cell outputs
- Cluster logs: Compute → [cluster] → Event Log

---

## ✅ Checklist

- [ ] Workspace accessed
- [ ] Data files uploaded to DBFS
- [ ] Notebook imported
- [ ] Cluster created and started
- [ ] Notebook attached to cluster
- [ ] Pipeline executed successfully
- [ ] Results downloaded
- [ ] Cluster terminated (to save costs)

---

**Good luck with your ML pipeline! 🚀**
