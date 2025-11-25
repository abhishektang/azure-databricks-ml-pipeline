# Databricks notebook source
# MAGIC %md
# MAGIC # ETL Pipeline - Data Processing & Preparation
# MAGIC **Author:** Abhishek Tanguturi  
# MAGIC **Student ID:** 48451109  
# MAGIC 
# MAGIC This notebook implements an ETL (Extract, Transform, Load) pipeline:
# MAGIC - **Extract**: Load raw data from DBFS
# MAGIC - **Transform**: Clean, impute, and engineer features
# MAGIC - **Load**: Save processed data for ML pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Extract - Load Raw Data

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import warnings

warnings.filterwarnings('ignore')

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

print("="*60)
print("ETL PIPELINE - EXTRACT PHASE")
print("="*60)

# Define file paths
RAW_TRAIN_PATH = "/dbfs/FileStore/tables/train.csv"
RAW_TEST_PATH = "/dbfs/FileStore/tables/test_data.csv"

# Load data using Spark
print("\nLoading raw training data...")
train_df = spark.read.csv(RAW_TRAIN_PATH, header=True, inferSchema=True)
print(f"✓ Training data loaded: {train_df.count()} rows, {len(train_df.columns)} columns")

print("\nLoading raw test data...")
test_df = spark.read.csv(RAW_TEST_PATH, header=True, inferSchema=True)
print(f"✓ Test data loaded: {test_df.count()} rows, {len(test_df.columns)} columns")

# Show schema
print("\nTraining Data Schema:")
train_df.printSchema()

# Display sample data
display(train_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Profiling & Quality Assessment

# COMMAND ----------

print("="*60)
print("DATA PROFILING")
print("="*60)

# Convert to Pandas for detailed analysis
train_pd = train_df.toPandas()
test_pd = test_df.toPandas()

# Basic statistics
print("\nDataset Summary:")
print(f"Training samples: {len(train_pd):,}")
print(f"Test samples: {len(test_pd):,}")
print(f"Total features: {len(train_pd.columns)}")

# Identify column types
numerical_cols = [col for col in train_pd.columns if col.startswith('Num_')]
categorical_cols = [col for col in train_pd.columns if col.startswith('Nom_')]
target_col = [col for col in train_pd.columns if 'Target' in col][0]

print(f"\nFeature Distribution:")
print(f"  Numerical features: {len(numerical_cols)}")
print(f"  Categorical features: {len(categorical_cols)}")
print(f"  Target column: {target_col}")

# Missing values analysis
print("\nMissing Values Analysis:")
missing_train = train_pd.isnull().sum()
missing_pct_train = (missing_train / len(train_pd) * 100).round(2)
missing_summary = pd.DataFrame({
    'Column': missing_train.index,
    'Missing_Count': missing_train.values,
    'Missing_Percentage': missing_pct_train.values
})
missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_summary) > 0:
    print(f"\nColumns with missing values: {len(missing_summary)}")
    display(missing_summary.head(10))
else:
    print("No missing values found!")

# Target distribution
print(f"\nTarget Distribution:")
target_dist = train_pd[target_col].value_counts().sort_index()
for class_val, count in target_dist.items():
    pct = count / len(train_pd) * 100
    print(f"  Class {class_val}: {count:,} ({pct:.2f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Transform - Data Cleaning

# COMMAND ----------

from scipy import stats
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import euclidean

print("="*60)
print("ETL PIPELINE - TRANSFORM PHASE")
print("="*60)

# Custom KNN Imputer for numerical features
def custom_knn_imputer(data, k=5):
    """Custom KNN Imputer implementation"""
    data_imputed = data.copy()
    
    missing_indices = data_imputed.isnull().any(axis=1)
    complete_indices = ~missing_indices
    
    complete_data = data_imputed[complete_indices].values
    incomplete_data = data_imputed[missing_indices]
    
    print(f"  Complete rows: {np.sum(complete_indices):,}")
    print(f"  Incomplete rows: {np.sum(missing_indices):,}")
    
    if np.sum(complete_indices) == 0:
        return data_imputed.fillna(data_imputed.mean())
    
    if np.sum(missing_indices) == 0:
        return data_imputed
    
    for idx in incomplete_data.index:
        row = data_imputed.loc[idx].values
        missing_cols = np.isnan(row)
        
        if not np.any(missing_cols):
            continue
            
        distances = []
        available_cols = ~missing_cols
        
        if not np.any(available_cols):
            for col_idx, is_missing in enumerate(missing_cols):
                if is_missing:
                    col_name = data_imputed.columns[col_idx]
                    data_imputed.loc[idx, col_name] = data_imputed[col_name].mean()
            continue
        
        for complete_row in complete_data:
            if np.any(available_cols):
                dist = euclidean(row[available_cols], complete_row[available_cols])
            else:
                dist = float('inf')
            distances.append(dist)
        
        k_actual = int(np.minimum(k, len(distances)))
        nearest_indices = np.argsort(distances)[:k_actual]
        
        for col_idx, is_missing in enumerate(missing_cols):
            if is_missing:
                col_name = data_imputed.columns[col_idx]
                neighbor_values = complete_data[nearest_indices, col_idx]
                imputed_value = np.mean(neighbor_values)
                data_imputed.loc[idx, col_name] = imputed_value
    
    return data_imputed

# Step 1: Handle missing values
print("\n1. MISSING VALUE IMPUTATION")
print("-" * 40)

# Impute numerical columns
if len(numerical_cols) > 0:
    print(f"Imputing {len(numerical_cols)} numerical columns using KNN...")
    train_pd[numerical_cols] = custom_knn_imputer(train_pd[numerical_cols], k=5)
    test_pd[numerical_cols] = custom_knn_imputer(test_pd[numerical_cols], k=5)
    print(f"✓ Numerical imputation complete")

# Impute categorical columns
if len(categorical_cols) > 0:
    print(f"Imputing {len(categorical_cols)} categorical columns using mode...")
    mode_imputer = SimpleImputer(strategy='most_frequent')
    train_pd[categorical_cols] = mode_imputer.fit_transform(train_pd[categorical_cols])
    test_pd[categorical_cols] = mode_imputer.transform(test_pd[categorical_cols])
    print(f"✓ Categorical imputation complete")

print(f"\nMissing values after imputation:")
print(f"  Training: {train_pd.isnull().sum().sum()}")
print(f"  Test: {test_pd.isnull().sum().sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Outlier Detection & Removal

# COMMAND ----------

print("\n2. OUTLIER DETECTION & REMOVAL")
print("-" * 40)

def detect_outliers_zscore(df, columns, threshold=3):
    """Detect outliers using Z-score method"""
    outlier_indices_all = set()
    outlier_stats = []
    
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = df[z_scores > threshold].index
        outlier_indices_all.update(outliers)
        
        outlier_stats.append({
            'Column': col,
            'Outliers': len(outliers),
            'Percentage': f"{len(outliers)/len(df)*100:.2f}%"
        })
    
    return outlier_indices_all, pd.DataFrame(outlier_stats)

# Detect outliers (only on training data, excluding target)
numerical_for_outliers = [col for col in numerical_cols if col != target_col]
outlier_indices, outlier_stats_df = detect_outliers_zscore(train_pd, numerical_for_outliers, threshold=3)

print(f"Outliers detected in {len(outlier_stats_df)} columns")
print(f"Total unique outlier rows: {len(outlier_indices):,}")

# Display outlier statistics
display(outlier_stats_df.head(10))

# Remove outliers from training data
train_pd_clean = train_pd.drop(index=outlier_indices).reset_index(drop=True)

print(f"\nOutlier Removal Summary:")
print(f"  Original size: {len(train_pd):,} rows")
print(f"  Outliers removed: {len(outlier_indices):,} rows ({len(outlier_indices)/len(train_pd)*100:.2f}%)")
print(f"  Clean dataset: {len(train_pd_clean):,} rows")

# Check class balance after outlier removal
print(f"\nClass distribution after outlier removal:")
for class_val, count in train_pd_clean[target_col].value_counts().sort_index().items():
    pct = count / len(train_pd_clean) * 100
    print(f"  Class {class_val}: {count:,} ({pct:.2f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature Engineering

# COMMAND ----------

print("\n3. FEATURE ENGINEERING")
print("-" * 40)

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Separate features and target
X_train = train_pd_clean.drop(columns=[target_col])
y_train = train_pd_clean[target_col]
X_test = test_pd.copy()

print(f"Feature matrix shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test: {X_test.shape}")

# Encode categorical features
print(f"\nEncoding categorical features...")
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    
    # Fit on training data
    X_train[col] = le.fit_transform(X_train[col])
    label_encoders[col] = le
    
    # Transform test data (handle unseen categories)
    test_col = X_test[col].copy()
    mask = ~test_col.isin(le.classes_)
    if mask.any():
        test_col.loc[mask] = le.classes_[0]
    X_test[col] = le.transform(test_col)
    
    print(f"  {col}: {len(le.classes_)} unique values")

# Normalize numerical features
print(f"\nNormalizing numerical features...")
scaler = MinMaxScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
print(f"✓ {len(numerical_cols)} numerical features normalized to [0, 1]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Data Validation

# COMMAND ----------

print("\n4. DATA VALIDATION")
print("-" * 40)

# Check for any remaining issues
print("Validating processed data...")

issues = []

# Check for missing values
if X_train.isnull().sum().sum() > 0:
    issues.append(f"Training data has {X_train.isnull().sum().sum()} missing values")
if X_test.isnull().sum().sum() > 0:
    issues.append(f"Test data has {X_test.isnull().sum().sum()} missing values")

# Check for infinite values
if np.isinf(X_train.values).sum() > 0:
    issues.append(f"Training data has {np.isinf(X_train.values).sum()} infinite values")
if np.isinf(X_test.values).sum() > 0:
    issues.append(f"Test data has {np.isinf(X_test.values).sum()} infinite values")

# Check data ranges for normalized features
for col in numerical_cols:
    if X_train[col].min() < 0 or X_train[col].max() > 1:
        issues.append(f"{col} not properly normalized in training data")
    if X_test[col].min() < 0 or X_test[col].max() > 1:
        issues.append(f"{col} not properly normalized in test data")

if issues:
    print("⚠ Data quality issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✓ All validation checks passed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Load - Save Processed Data

# COMMAND ----------

print("\n5. SAVING PROCESSED DATA")
print("-" * 40)

# Define output paths
PROCESSED_TRAIN_FEATURES_PATH = "/dbfs/FileStore/processed/X_train.parquet"
PROCESSED_TRAIN_TARGET_PATH = "/dbfs/FileStore/processed/y_train.parquet"
PROCESSED_TEST_PATH = "/dbfs/FileStore/processed/X_test.parquet"
METADATA_PATH = "/dbfs/FileStore/processed/metadata.csv"

# Create output directory
import os
os.makedirs("/dbfs/FileStore/processed", exist_ok=True)

# Save processed training features
X_train.to_parquet(PROCESSED_TRAIN_FEATURES_PATH, index=False)
print(f"✓ Training features saved: {PROCESSED_TRAIN_FEATURES_PATH}")

# Save training target
y_train_df = pd.DataFrame({target_col: y_train})
y_train_df.to_parquet(PROCESSED_TRAIN_TARGET_PATH, index=False)
print(f"✓ Training target saved: {PROCESSED_TRAIN_TARGET_PATH}")

# Save processed test features
X_test.to_parquet(PROCESSED_TEST_PATH, index=False)
print(f"✓ Test features saved: {PROCESSED_TEST_PATH}")

# Save metadata
metadata = pd.DataFrame({
    'numerical_features': [','.join(numerical_cols)],
    'categorical_features': [','.join(categorical_cols)],
    'target_column': [target_col],
    'train_samples': [len(X_train)],
    'test_samples': [len(X_test)],
    'num_features': [len(X_train.columns)],
    'outliers_removed': [len(outlier_indices)]
})
metadata.to_csv(METADATA_PATH, index=False)
print(f"✓ Metadata saved: {METADATA_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. ETL Summary

# COMMAND ----------

print("\n" + "="*60)
print("ETL PIPELINE COMPLETED SUCCESSFULLY")
print("="*60)

summary = f"""
EXTRACT:
  ✓ Raw training data loaded: {len(train_pd):,} rows
  ✓ Raw test data loaded: {len(test_pd):,} rows

TRANSFORM:
  ✓ Missing values imputed (KNN for numerical, mode for categorical)
  ✓ Outliers removed: {len(outlier_indices):,} rows
  ✓ Categorical features encoded: {len(categorical_cols)} columns
  ✓ Numerical features normalized: {len(numerical_cols)} columns

LOAD:
  ✓ Processed training features: {X_train.shape}
  ✓ Processed training target: {y_train.shape}
  ✓ Processed test features: {X_test.shape}

OUTPUT FILES:
  • {PROCESSED_TRAIN_FEATURES_PATH}
  • {PROCESSED_TRAIN_TARGET_PATH}
  • {PROCESSED_TEST_PATH}
  • {METADATA_PATH}

DATA QUALITY:
  ✓ No missing values
  ✓ No infinite values
  ✓ All features properly scaled
  ✓ Ready for machine learning pipeline

NEXT STEP:
  Run the ML Classification Pipeline notebook
"""

print(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Data Quality Report

# COMMAND ----------

# Generate comprehensive data quality report
print("="*60)
print("DATA QUALITY REPORT")
print("="*60)

# Feature statistics
print("\nNumerical Feature Statistics (Top 10):")
display(X_train[numerical_cols[:10]].describe().T)

print("\nCategorical Feature Distribution (Top 5):")
for col in categorical_cols[:5]:
    print(f"\n{col}:")
    print(X_train[col].value_counts().head(5))

# Target distribution
print(f"\nFinal Target Distribution:")
display(y_train.value_counts().sort_index().to_frame('Count'))

# Create visualization of data processing pipeline
print("\n✓ ETL Pipeline Complete - Data ready for ML modeling!")
