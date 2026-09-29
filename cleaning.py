"""
cleaning.py - Data Cleaning and Preprocessing Pipeline
=====================================================

This module provides a modular, end-to-end data cleaning pipeline for the
Employee HR dataset. It handles:
  1. Missing values imputation (mean/median for numerical, mode for categorical)
  2. Duplicate row detection and removal
  3. Inconsistent categorical and text value standardization
  4. Data type casting (dates to datetime, currency strings to float)
  5. Invalid numerical anomaly handling (negative age/salary, impossible bounds)
  6. Saving the cleaned dataset to CSV

Author: Data Science Intern
Task: Task 01 - Data Cleaning and Preprocessing
"""

import os
import sys
import pandas as pd
import numpy as np


def get_project_paths():
    """
    Dynamically locate the data paths regardless of whether the script
    is executed from project root, src folder, or parent directory.
    
    Returns:
        tuple: (raw_data_path, cleaned_data_path)
    """
    # Current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    raw_path = os.path.join(project_root, "data", "raw_dataset.csv")
    cleaned_path = os.path.join(project_root, "data", "cleaned_dataset.csv")
    
    return raw_path, cleaned_path


def load_dataset(filepath):
    """
    Load the raw CSV dataset into a Pandas DataFrame.
    
    Parameters:
        filepath (str): Path to raw CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"[INFO] Successfully loaded dataset from: {filepath}")
    print(f"[INFO] Raw dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df):
    """
    Identify and impute missing values using appropriate statistical techniques.
    
    Techniques used:
      - Full_Name: Fill with 'Unknown Employee' (names cannot be calculated)
      - Department, City, Gender: Fill with column mode (most frequent value)
      - Age, Experience_Years: Fill with column median (robust against outliers)
      
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    df = df.copy()
    initial_nulls = df.isnull().sum().sum()
    print(f"\n[STEP 1] Handling missing values (Initial total nulls: {initial_nulls})...")
    
    # 1. Names - Replace missing with 'Unknown Employee'
    if 'Full_Name' in df.columns:
        df['Full_Name'] = df['Full_Name'].fillna("Unknown Employee")
        
    # 2. Categorical columns - Impute using Mode (most common category)
    # Explanation: Mode is the most representative category in categorical distributions.
    categorical_cols = ['Department', 'Gender', 'City']
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            col_mode = df[col].mode()[0]
            df[col] = df[col].fillna(col_mode)
            print(f"  - Imputed missing '{col}' with mode: '{col_mode}'")
            
    # 3. Numerical columns - Impute using Median
    # Explanation: Median is resistant to extreme outliers (unlike mean which gets pulled).
    numerical_cols = ['Age', 'Experience_Years']
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            col_median = df[col].median()
            df[col] = df[col].fillna(col_median)
            print(f"  - Imputed missing '{col}' with median: {col_median}")
            
    return df


def remove_duplicates(df):
    """
    Identify and remove exact duplicate records from the dataset.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    df = df.copy()
    duplicate_count = df.duplicated().sum()
    print(f"\n[STEP 2] Removing duplicates...")
    print(f"  - Duplicate rows found: {duplicate_count}")
    
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  - Successfully dropped {duplicate_count} duplicate rows.")
    else:
        print("  - No duplicate rows to drop.")
        
    return df


def standardize_categorical_values(df):
    """
    Clean text strings and standardize inconsistent categorical values:
      - Strips leading and trailing whitespaces
      - Standardizes casing (Title Case)
      - Maps alternate city names (e.g. 'Bangalore' -> 'Bengaluru')
      - Maps department abbreviations (e.g. 'HR' -> 'Human Resources')
      - Maps gender abbreviations (e.g. 'M' -> 'Male', 'F' -> 'Female')
      
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with standardized categorical values.
    """
    df = df.copy()
    print("\n[STEP 3] Standardizing text and categorical values...")
    
    # 1. Standardize Full_Name
    if 'Full_Name' in df.columns:
        df['Full_Name'] = df['Full_Name'].astype(str).str.strip().str.title()
        
    # 2. Standardize Department
    if 'Department' in df.columns:
        df['Department'] = df['Department'].astype(str).str.strip().str.title()
        # Map abbreviations to standard names
        dept_mapping = {
            'Hr': 'Human Resources',
            'Human Resources': 'Human Resources',
            'Engineering': 'Engineering',
            'Marketing': 'Marketing',
            'Sales': 'Sales',
            'Finance': 'Finance'
        }
        df['Department'] = df['Department'].replace(dept_mapping)
        print(f"  - Standardized 'Department' unique values: {sorted(df['Department'].unique())}")

    # 3. Standardize City
    if 'City' in df.columns:
        df['City'] = df['City'].astype(str).str.strip().str.title()
        # Map aliases (Bangalore -> Bengaluru)
        city_mapping = {
            'Bangalore': 'Bengaluru'
        }
        df['City'] = df['City'].replace(city_mapping)
        print(f"  - Standardized 'City' unique values: {sorted(df['City'].unique())}")
        
    # 4. Standardize Gender
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).str.strip().str.title()
        gender_mapping = {
            'M': 'Male',
            'F': 'Female'
        }
        df['Gender'] = df['Gender'].replace(gender_mapping)
        print(f"  - Standardized 'Gender' unique values: {sorted(df['Gender'].unique())}")

    return df


def correct_data_types(df):
    """
    Convert columns to their correct, analysis-ready data types:
      - 'Joining_Date': string -> datetime64[ns]
      - 'Salary': string with currency symbols ($ and ,) -> float64
      - 'Age' & 'Experience_Years': float/float -> int64
      
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with corrected data types.
    """
    df = df.copy()
    print("\n[STEP 4] Correcting data types...")
    
    # 1. Convert Joining_Date to datetime
    if 'Joining_Date' in df.columns:
        df['Joining_Date'] = pd.to_datetime(df['Joining_Date'], format='mixed')
        print(f"  - 'Joining_Date' converted to: {df['Joining_Date'].dtype}")
        
    # 2. Clean and convert Salary from string ($65,000) to numeric float
    if 'Salary' in df.columns:
        # Convert to string, strip whitespace, remove '$' and ','
        cleaned_salary = (
            df['Salary']
            .astype(str)
            .str.strip()
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
        )
        df['Salary'] = pd.to_numeric(cleaned_salary, errors='coerce')
        # If any salary was missing (NaN after coercion), impute with median salary
        median_salary = df.loc[df['Salary'] > 0, 'Salary'].median()
        df['Salary'] = df['Salary'].fillna(median_salary)
        print(f"  - 'Salary' converted to: {df['Salary'].dtype} (Missing values filled with median: ${median_salary:,.2f})")
        
    # 3. Cast Age and Experience to integer
    if 'Age' in df.columns:
        df['Age'] = df['Age'].round().astype(int)
    if 'Experience_Years' in df.columns:
        df['Experience_Years'] = df['Experience_Years'].round().astype(int)
        
    return df


def handle_invalid_numerical_values(df):
    """
    Detect and correct physically or logically impossible numerical values:
      - Negative Age (e.g. -25) or unrealistic Age (> 65) -> replaced with median age
      - Negative or zero Salary (e.g. -$5,000, $0) -> replaced with median positive salary
      - Negative Experience (< 0) -> replaced with 0
      - Experience exceeding Age - 18 -> capped to (Age - 18)
      
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with invalid numerical values rectified.
    """
    df = df.copy()
    print("\n[STEP 5] Handling invalid numerical anomalies...")
    
    # 1. Handle Age anomalies (valid working age: 18 to 65)
    if 'Age' in df.columns:
        invalid_age_mask = (df['Age'] < 18) | (df['Age'] > 65)
        num_invalid_ages = invalid_age_mask.sum()
        if num_invalid_ages > 0:
            valid_age_median = int(df.loc[~invalid_age_mask, 'Age'].median())
            df.loc[invalid_age_mask, 'Age'] = valid_age_median
            print(f"  - Detected {num_invalid_ages} invalid Age records (<18 or >65). Replaced with median: {valid_age_median}")
            
    # 2. Handle Salary anomalies (valid salary must be > 0)
    if 'Salary' in df.columns:
        invalid_salary_mask = df['Salary'] <= 0
        num_invalid_salaries = invalid_salary_mask.sum()
        if num_invalid_salaries > 0:
            valid_sal_median = df.loc[~invalid_salary_mask, 'Salary'].median()
            df.loc[invalid_salary_mask, 'Salary'] = valid_sal_median
            print(f"  - Detected {num_invalid_salaries} non-positive Salary records (<= 0). Replaced with median: ${valid_sal_median:,.2f}")

    # 3. Handle Experience anomalies (experience cannot be negative or logically impossible)
    if 'Experience_Years' in df.columns:
        # Negative experience corrected to 0
        neg_exp_mask = df['Experience_Years'] < 0
        num_neg_exp = neg_exp_mask.sum()
        if num_neg_exp > 0:
            df.loc[neg_exp_mask, 'Experience_Years'] = 0
            print(f"  - Detected {num_neg_exp} negative Experience records. Corrected to 0.")
            
        # Experience cannot exceed Age - 18 (person couldn't work before age 18)
        max_possible_exp = (df['Age'] - 18).clip(lower=0)
        exceed_mask = df['Experience_Years'] > max_possible_exp
        num_exceed = exceed_mask.sum()
        if num_exceed > 0:
            df.loc[exceed_mask, 'Experience_Years'] = max_possible_exp[exceed_mask]
            print(f"  - Detected {num_exceed} Experience records exceeding Age - 18. Capped to max logical experience.")
            
    return df


def clean_pipeline(raw_filepath=None, cleaned_filepath=None):
    """
    Execute the entire data cleaning pipeline end-to-end and save results.
    
    Parameters:
        raw_filepath (str, optional): Custom path to raw dataset CSV.
        cleaned_filepath (str, optional): Custom path to output cleaned dataset CSV.
        
    Returns:
        tuple: (raw_df, cleaned_df, summary_dict)
    """
    default_raw, default_cleaned = get_project_paths()
    raw_path = raw_filepath if raw_filepath else default_raw
    cleaned_path = cleaned_filepath if cleaned_filepath else default_cleaned
    
    print("=" * 70)
    print("      DATA CLEANING AND PREPROCESSING PIPELINE STARTING")
    print("=" * 70)
    
    # 1. Load Data
    raw_df = load_dataset(raw_path)
    original_rows = len(raw_df)
    original_nulls = raw_df.isnull().sum().sum()
    original_duplicates = raw_df.duplicated().sum()
    
    # 2. Handle Missing Values
    df_clean = handle_missing_values(raw_df)
    
    # 3. Remove Duplicates
    df_clean = remove_duplicates(df_clean)
    
    # 4. Standardize Text & Categorical Values
    df_clean = standardize_categorical_values(df_clean)
    
    # 5. Correct Data Types
    df_clean = correct_data_types(df_clean)
    
    # 6. Handle Invalid Numerical Values
    df_clean = handle_invalid_numerical_values(df_clean)
    
    # 7. Ensure output directory exists and save cleaned CSV
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    df_clean.to_csv(cleaned_path, index=False)
    print(f"\n[SUCCESS] Cleaned dataset saved to: {cleaned_path}")
    
    final_rows = len(df_clean)
    duplicates_removed = original_duplicates
    final_nulls = df_clean.isnull().sum().sum()
    
    summary = {
        "original_rows": original_rows,
        "duplicates_removed": duplicates_removed,
        "missing_values_handled": original_nulls,
        "columns_processed": df_clean.shape[1],
        "final_rows": final_rows,
        "final_nulls": final_nulls,
        "status": "Clean, Validated & Ready for Modeling"
    }
    
    print("\n" + "=" * 70)
    print("                   DATA CLEANING SUMMARY")
    print("=" * 70)
    print(f"  * Number of original rows:         {summary['original_rows']}")
    print(f"  * Number of duplicate rows removed:{summary['duplicates_removed']}")
    print(f"  * Number of missing values handled:{summary['missing_values_handled']}")
    print(f"  * Number of columns processed:     {summary['columns_processed']}")
    print(f"  * Number of final rows:            {summary['final_rows']}")
    print(f"  * Remaining null values:           {summary['final_nulls']}")
    print(f"  * Final dataset status:            {summary['status']}")
    print("=" * 70)
    
    return raw_df, df_clean, summary


if __name__ == "__main__":
    clean_pipeline()
