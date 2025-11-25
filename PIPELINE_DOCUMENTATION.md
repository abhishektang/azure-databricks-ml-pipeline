# Complete ETL + ML Pipeline Documentation

## 📊 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. ETL PIPELINE (Extract → Transform → Load)              │
│     ├── Extract: Load raw CSV files from DBFS              │
│     ├── Transform:                                          │
│     │   ├── Data profiling & quality assessment            │
│     │   ├── Missing value imputation (KNN + Mode)          │
│     │   ├── Outlier detection & removal (Z-score)          │
│     │   ├── Feature encoding (Label Encoding)              │
│     │   └── Feature normalization (MinMax Scaling)         │
│     └── Load: Save processed data to Parquet               │
│                                                             │
│  2. ML PIPELINE (Train → Predict → Evaluate)               │
│     ├── Load processed data from ETL                       │
│     ├── Class balancing (Custom resampling)                │
│     ├── Model training (Voting Classifier)                 │
│     │   ├── Random Forest                                  │
│     │   ├── K-Nearest Neighbors                            │
│     │   ├── Decision Tree                                  │
│     │   └── Naive Bayes                                    │
│     ├── Threshold optimization (F1 maximization)           │
│     ├── Cross-validation (5-fold stratified)               │
│     └── Generate predictions & output file                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Files Created

### Notebooks
1. **`etl_pipeline`** - ETL Pipeline notebook
2. **`ml_classification_pipeline`** - ML Pipeline notebook

### Scripts
1. **`run_complete_pipeline.sh`** - Runs both pipelines in sequence
2. **`monitor_job.sh`** - Monitors individual job execution

### Data Flow
```
Raw Data (DBFS)
    ↓
/FileStore/tables/train.csv
/FileStore/tables/test_data.csv
    ↓
ETL Pipeline
    ↓
/FileStore/processed/X_train.parquet
/FileStore/processed/y_train.parquet
/FileStore/processed/X_test.parquet
/FileStore/processed/metadata.csv
    ↓
ML Pipeline
    ↓
/FileStore/output/s4845110.infs4203
```

## 🚀 Execution Options

### Option 1: Run Complete Pipeline (ETL → ML)
```bash
cd /Users/abhishektanguturi/Projects/env/Assignment/INFS7203
./run_complete_pipeline.sh
```

This will:
1. ✅ Run ETL pipeline first
2. ✅ Wait for ETL completion
3. ✅ Run ML pipeline with processed data
4. ✅ Download final results automatically

### Option 2: Run Pipelines Individually

**A. Run ETL Pipeline Only:**
```bash
/Users/abhishektanguturi/Projects/env/bin/databricks runs submit --json '{
  "run_name": "ETL Pipeline",
  "existing_cluster_id": "1124-132744-47atqg0y",
  "notebook_task": {
    "notebook_path": "/Users/t.abhishek45699@yahoo.com/etl_pipeline",
    "source": "WORKSPACE"
  }
}'
```

**B. Run ML Pipeline Only:**
```bash
/Users/abhishektanguturi/Projects/env/bin/databricks runs submit --json '{
  "run_name": "ML Pipeline",
  "existing_cluster_id": "1124-132744-47atqg0y",
  "notebook_task": {
    "notebook_path": "/Users/t.abhishektanguturi/Projects/env/Assignment/INFS7203/ml_classification_pipeline",
    "source": "WORKSPACE"
  }
}'
```

### Option 3: Manual Execution in Databricks UI
1. Open: https://adb-1889721579194859.19.azuredatabricks.net
2. Navigate to **Workspace** → **Users** → **t.abhishek45699@yahoo.com**
3. Open **etl_pipeline** → Attach cluster → Run all
4. After completion, open **ml_classification_pipeline** → Run all

## 📊 ETL Pipeline Details

### Extract Phase
- Loads raw CSV files using Spark
- Performs initial data profiling
- Identifies column types (numerical, categorical, target)

### Transform Phase

#### 1. Missing Value Imputation
- **Numerical features**: Custom KNN imputation (k=5)
  - Uses Euclidean distance to find nearest neighbors
  - Imputes based on mean of k-nearest complete rows
- **Categorical features**: Mode imputation
  - Replaces missing with most frequent value

#### 2. Outlier Detection & Removal
- **Method**: Z-score statistical approach
- **Threshold**: 3 standard deviations
- **Applied to**: Numerical features only
- **Effect**: Removes ~5-10% of training data
- **Test data**: No outlier removal (only imputation)

#### 3. Feature Encoding
- **Categorical features**: Label Encoding
  - Converts categories to numerical values
  - Handles unseen categories in test data

#### 4. Feature Normalization
- **Numerical features**: MinMax Scaling
  - Scales all values to [0, 1] range
  - Uses same scaler for train and test

### Load Phase
- Saves processed data as Parquet files (efficient, compressed)
- Generates metadata file with processing statistics
- Data validation checks before saving

### Outputs
- `/FileStore/processed/X_train.parquet` - Training features
- `/FileStore/processed/y_train.parquet` - Training target
- `/FileStore/processed/X_test.parquet` - Test features
- `/FileStore/processed/metadata.csv` - Processing metadata

## 🤖 ML Pipeline Details

### Model Architecture
**Weighted Voting Classifier** combining 4 algorithms:

1. **Random Forest** (weight=3)
   - 200 estimators
   - Max depth: 12
   - Balanced class weights

2. **K-Nearest Neighbors** (weight=2)
   - 11 neighbors
   - Distance-weighted
   - Euclidean metric

3. **Decision Tree** (weight=2)
   - Max depth: 12
   - Balanced class weights
   - Gini criterion

4. **Naive Bayes** (weight=1)
   - Gaussian distribution
   - Variance smoothing: 1e-8

### Training Process
1. Custom resampling for class balance
2. Soft voting (probability-based)
3. Threshold optimization for F1 score
4. 5-fold stratified cross-validation

### Performance Targets
- **F1 Score**: > 0.65
- **Accuracy**: ~0.85-0.90
- **Cross-validation**: Stratified 5-fold

## ⏱️ Execution Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| ETL Pipeline | 5-8 min | Data processing & transformation |
| ML Pipeline | 10-15 min | Model training & prediction |
| **Total** | **15-23 min** | Complete end-to-end pipeline |

## 📥 Output Format

Final file: `/FileStore/output/s4845110.infs4203`

```
0,
1,
0,
1,
...
[n predictions]
0.875,0.682,
```

Last row contains:
- Accuracy (3 decimal places)
- F1 Score (3 decimal places)

## 🔍 Monitoring & Debugging

### Check Pipeline Status
```bash
# List recent jobs
/Users/abhishektanguturi/Projects/env/bin/databricks runs list --limit 5

# Check specific job
/Users/abhishektanguturi/Projects/env/bin/databricks runs get --run-id <RUN_ID>
```

### View Logs
- Web UI: https://adb-1889721579194859.19.azuredatabricks.net
- Navigate to: **Workflows** → **Job runs**
- Click on run ID to view detailed logs

### Common Issues

**Issue**: ETL fails with memory error  
**Solution**: Reduce batch size or increase cluster size

**Issue**: ML pipeline shows low F1 score  
**Solution**: Adjust resampling ratio or ensemble weights

**Issue**: Cluster terminated during execution  
**Solution**: Increase auto-termination time or check quota limits

## 💡 Best Practices

1. **Always run ETL first** before ML pipeline
2. **Monitor both pipelines** using the monitoring scripts
3. **Check data quality** after ETL completion
4. **Validate results** before submitting
5. **Terminate cluster** after completion to save costs

## 🎯 Success Criteria

✅ ETL Pipeline produces:
- Zero missing values
- No infinite values
- Properly encoded features
- Normalized numerical values

✅ ML Pipeline produces:
- F1 Score > 0.65
- Proper output format
- All test samples predicted

## 📞 Quick Commands

```bash
# Run complete pipeline
./run_complete_pipeline.sh

# Monitor current job
./monitor_job.sh

# List all notebooks
/Users/abhishektanguturi/Projects/env/bin/databricks workspace ls /Users/t.abhishek45699@yahoo.com/

# Download results
/Users/abhishektanguturi/Projects/env/bin/databricks fs cp dbfs:/FileStore/output/s4845110.infs4203 ./s4845110.infs4203

# Check cluster status
/Users/abhishektanguturi/Projects/env/bin/databricks clusters get --cluster-id 1124-132744-47atqg0y
```

## 🎓 Learning Outcomes

This pipeline demonstrates:
- ✅ Professional ETL design patterns
- ✅ Data quality and validation
- ✅ Feature engineering techniques
- ✅ Ensemble machine learning
- ✅ Cloud-based ML workflows
- ✅ Automated pipeline orchestration
