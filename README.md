# Azure Databricks ML Classification Pipeline

A complete ETL and Machine Learning pipeline for binary classification, deployed on Azure Databricks with ensemble methods achieving **F1 Score: 0.904**.

## 🎯 Project Overview

This project implements an end-to-end data science pipeline combining:
- **ETL Pipeline**: Data extraction, cleaning, feature engineering, and preprocessing
- **ML Pipeline**: Ensemble classification using Random Forest, KNN, Decision Tree, and Naive Bayes
- **Cloud Deployment**: Fully automated deployment on Azure Databricks

## 📊 Results

- **F1 Score**: 0.904
- **Accuracy**: 0.814
- **Cross-Validation**: 5-fold stratified

## 🏗️ Architecture

```
Raw Data → ETL Pipeline → Processed Data → ML Pipeline → Predictions
             ↓                               ↓
         DBFS Storage                    Model Training
         (Parquet)                       (Ensemble)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Azure account with Databricks workspace
- Azure CLI
- Databricks CLI

### Local Setup

```bash
# Clone repository
git clone https://github.com/yourusername/azure-databricks-ml-pipeline.git
cd azure-databricks-ml-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Azure Databricks Deployment

1. **Install Azure CLI and authenticate**:
```bash
# Install Azure CLI (macOS)
brew install azure-cli

# Login to Azure
az login
```

2. **Deploy Databricks workspace**:
```bash
chmod +x deploy_to_databricks.sh
./deploy_to_databricks.sh
```

3. **Configure Databricks CLI**:
```bash
# Get your Personal Access Token from Databricks workspace
# Settings → User Settings → Access Tokens

databricks configure --token
# Enter workspace URL and token
```

4. **Upload data and notebooks**:
```bash
# Upload training and test data
databricks fs cp train.csv dbfs:/FileStore/tables/train.csv
databricks fs cp test_data.csv dbfs:/FileStore/tables/test_data.csv

# Upload notebooks
databricks workspace import etl_pipeline_notebook.py /Users/your.email@domain.com/etl_pipeline --format SOURCE --language PYTHON
databricks workspace import databricks_notebook.py /Users/your.email@domain.com/ml_classification_pipeline --format SOURCE --language PYTHON
```

5. **Run the complete pipeline**:
```bash
python run_pipeline.py
```

## 📁 Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── Data Files
│   ├── train.csv                      # Training data
│   └── test_data.csv                  # Test data
│
├── ETL Pipeline
│   └── etl_pipeline_notebook.py       # Data preprocessing notebook
│
├── ML Pipeline
│   ├── databricks_notebook.py         # Main ML classification notebook
│   └── main.py                        # Local version of ML pipeline
│
├── Deployment Scripts
│   ├── deploy_to_databricks.sh        # Azure Databricks workspace creation
│   ├── setup_databricks.sh            # CLI configuration script
│   ├── setup_databricks.py            # Python setup script
│   ├── run_pipeline.py                # Complete pipeline orchestration
│   └── run_pipeline.sh                # Shell pipeline runner
│
├── Monitoring
│   └── monitor_job.sh                 # Job monitoring script
│
└── Documentation
    ├── DATABRICKS_SETUP.md            # Databricks setup guide
    ├── QUICKSTART.md                  # Quick start guide
    ├── PIPELINE_DOCUMENTATION.md      # Complete pipeline documentation
    └── NORMALIZATION_GUIDE.md         # Data normalization guide
```

## 🔧 Pipeline Components

### ETL Pipeline (`etl_pipeline_notebook.py`)

**Extract**:
- Loads raw CSV data from DBFS
- Data profiling and quality assessment

**Transform**:
- Custom KNN imputation for numerical features
- Mode imputation for categorical features
- Z-score outlier detection and removal (threshold=3)
- Label encoding for categorical features
- MinMax normalization for numerical features

**Load**:
- Saves processed data as Parquet files
- Generates metadata for downstream processing

**Output**:
- `X_train.parquet` - Processed training features
- `y_train.parquet` - Training labels
- `X_test.parquet` - Processed test features
- `metadata.csv` - Processing metadata

### ML Classification Pipeline (`databricks_notebook.py`)

**Models**:
1. **Random Forest** (weight=3)
   - n_estimators=200, max_depth=12
   - class_weight='balanced'
   
2. **K-Nearest Neighbors** (weight=2)
   - n_neighbors=11, weights='distance'
   
3. **Decision Tree** (weight=2)
   - max_depth=12, class_weight='balanced'
   
4. **Naive Bayes** (weight=1)
   - GaussianNB with var_smoothing=1e-8

**Techniques**:
- Weighted soft voting ensemble
- Custom resampling for class imbalance (target ratio: 0.35)
- Threshold optimization for F1 maximization
- 5-fold stratified cross-validation

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| F1 Score | 0.904 |
| Accuracy | 0.814 |
| Optimal Threshold | 0.45 |

## 🛠️ Technologies Used

- **Languages**: Python 3.12
- **Cloud Platform**: Microsoft Azure
- **Big Data**: Azure Databricks, Apache Spark
- **ML Libraries**: scikit-learn, pandas, numpy, scipy
- **Deployment**: Azure CLI, Databricks CLI
- **Version Control**: Git, GitHub

## 📝 Key Features

✅ **Separation of Concerns**: ETL and ML pipelines are completely separated  
✅ **Scalable**: Built on Apache Spark for handling large datasets  
✅ **Reproducible**: Fixed random seeds and comprehensive documentation  
✅ **Automated**: One-command deployment and execution  
✅ **Production-Ready**: Error handling, logging, and monitoring  
✅ **Best Practices**: Clean code, modular design, proper documentation  

## 🔍 Data Processing Highlights

- **Missing Values**: Custom KNN imputation preserves data relationships
- **Outliers**: Z-score method removes extreme values (3σ threshold)
- **Encoding**: Label encoding for categorical features
- **Normalization**: MinMax scaling to [0, 1] range
- **Class Imbalance**: Custom resampling with configurable target ratio

## 📊 Monitoring & Logs

Monitor job execution in real-time:
```bash
./monitor_job.sh <run_id>
```

View logs in Databricks workspace:
- Navigate to: Workflows → Runs → Select Run ID
- Check stdout/stderr for detailed execution logs

## 🤝 Contributing

This is an academic project. For questions or suggestions, please open an issue.

## 📄 License

This project is for educational purposes as part of INFS7203 coursework.

## 👤 Author

**Abhishek Tanguturi**  
Student ID: 48451109  
Email: t.abhishek45699@yahoo.com

## 🙏 Acknowledgments

- University of Queensland - INFS7203 Data Mining
- Azure Databricks Documentation
- scikit-learn Community

---

**Note**: This pipeline was designed for a binary classification task with imbalanced classes. The architecture can be adapted for multi-class problems with minimal modifications.
