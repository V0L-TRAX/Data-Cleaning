# Data Cleaning and Preprocessing Pipeline

## 📌 Project Title
**End-to-End Data Cleaning and Preprocessing Pipeline in Python**

---

## 🎯 Project Objective
The primary objective of this project is to build a robust, beginner-friendly, and production-grade data cleaning and preprocessing pipeline using Python. The project demonstrates how raw, noisy, real-world data containing common quality issues is systematically diagnosed, transformed, standardized, and validated into high-quality, analysis-ready data for downstream analytics and machine learning.

---

## ⚠️ Problem Statement
Real-world datasets rarely come clean. They are frequently plagued by data entry errors, system merge duplications, missing information, inconsistent string casing, trailing spaces, incorrect data types, and logically impossible values. Feeding such dirty data into analytical dashboards or machine learning models leads to distorted metrics, biased predictions, and runtime errors ("garbage in, garbage out"). 

This project tackles these exact data quality challenges on an **Employee HR & Payroll Dataset** through a structured, step-by-step pipeline.

---

## 📊 Dataset Description
The dataset simulates employee records across various departments within a tech enterprise (`raw_dataset.csv`), containing **420 records** and **9 features**.

### Schema & Attributes:
| Column Name | Raw Data Type | Cleaned Data Type | Description | Injected Quality Issues |
| :--- | :--- | :--- | :--- | :--- |
| **`Employee_ID`** | `object` (string) | `object` (string) | Unique identifier for employees (e.g., EMP001) | Baseline identifier; duplicated across duplicate rows |
| **`Full_Name`** | `object` (string) | `object` (string) | Employee's full name | Missing values (`NaN`), extra leading/trailing whitespace |
| **`Department`** | `object` (string) | `object` (string) | Assigned department | Inconsistent casing (`marketing`, `SALES`), abbreviations (`HR` vs `Human Resources`), missing values |
| **`Age`** | `float64` | `int64` | Employee's age | Missing values, negative numbers (`-25`), impossible values (`150`) |
| **`Gender`** | `object` (string) | `object` (string) | Employee's gender | Abbreviations (`M`, `F`), mixed casing (`male`, `Female`), missing values |
| **`City`** | `object` (string) | `object` (string) | Office location city | Inconsistent casing (`delhi`, `MUMBAI`), aliases (`Bangalore` vs `Bengaluru`), whitespaces, missing values |
| **`Joining_Date`** | `object` (string) | `datetime64[ns]` | Date of joining company | Mixed date string formats (`YYYY-MM-DD`, `YYYY/MM/DD`, `DD-MM-YYYY`, `YYYY.MM.DD`) |
| **`Salary`** | `object` (string) | `float64` | Annual salary in USD | Formatted with `$` and `,`, missing values, negative salaries (`-$5,000`), zero values |
| **`Experience_Years`** | `float64` | `int64` | Total years of experience | Missing values, negative values (`-2`), values exceeding `(Age - 18)` |

---

## 🛠️ Technologies Used
* **Python (3.10+)**: Core programming language.
* **Pandas**: High-performance data manipulation, DataFrame structures, and statistical querying.
* **NumPy**: Mathematical operations, array manipulations, and missing value indicators (`np.nan`).
* **Jupyter Notebook**: Interactive, step-by-step development and analytical reporting environment.

---

## 🔄 Data Cleaning Steps
The cleaning process is implemented in a strict, logical sequence across both the Jupyter Notebook (`notebooks/data_cleaning.ipynb`) and the automated Python script (`src/cleaning.py`):

1. **Exploratory Data Inspection**:
   * Inspect first and last 5 records (`df.head()`, `df.tail()`).
   * Measure dimensions (`df.shape`) and examine column names (`df.columns`).
   * Diagnose data types and null counts using `df.info()`.
   * Compute initial descriptive statistics with `df.describe()` to uncover range anomalies.

2. **Missing Values Identification & Statistical Imputation**:
   * Quantify missing values per column via `df.isnull().sum()`.
   * **Full_Name**: Imputed with `'Unknown Employee'` (identities cannot be averaged).
   * **Categorical Variables (`Department`, `City`, `Gender`)**: Imputed using **Mode** (the most frequent category, preserving probability distribution).
   * **Numerical Variables (`Age`, `Experience_Years`)**: Imputed using **Median** (the 50th percentile is robust against outliers, unlike the mean).

3. **Duplicate Record Detection & Removal**:
   * Detect duplicate rows using `df.duplicated()`.
   * Drop identical records with `df.drop_duplicates()`, resetting row indices.

4. **Categorical Standardization & Text Sanitization**:
   * Strip leading and trailing whitespace using `.str.strip()`.
   * Normalize text casing to Title Case using `.str.title()`.
   * Resolve aliases and abbreviations using `.replace()`:
     * City: `"Bangalore"` ➔ `"Bengaluru"`
     * Department: `"Hr"` ➔ `"Human Resources"`
     * Gender: `"M"` ➔ `"Male"`, `"F"` ➔ `"Female"`

5. **Data Type Correction**:
   * `Joining_Date`: Converted from raw string to `datetime64[ns]` with `pd.to_datetime(..., format='mixed')`.
   * `Salary`: Stripped `$` and `,` symbols, cast to `float64`, and imputed remaining nulls with median salary.
   * `Age` & `Experience_Years`: Cast from float to `int64`.

6. **Invalid Numerical Value Handling**:
   * **Age**: Replaced values `< 18` or `> 65` with the median valid age.
   * **Salary**: Replaced non-positive values (`<= 0`) with the median positive salary.
   * **Experience**: Replaced negative values with `0`, and capped values exceeding `(Age - 18)`.

7. **Post-Cleaning Validation & Output Export**:
   * Re-verify zero nulls and zero duplicates using assertion tests.
   * Export final clean dataset to `data/cleaned_dataset.csv`.

---

## ⚖️ Before vs. After Cleaning Comparison

| Metric | Raw Dataset (Before) | Cleaned Dataset (After) | Rationale / Transformation |
| :--- | :--- | :--- | :--- |
| **Total Rows** | 420 | 400 | 20 duplicate records eliminated |
| **Total Columns** | 9 | 9 | Schema preserved |
| **Duplicate Records** | 20 | 0 | 100% duplicate-free |
| **Total Missing Values** | 133 | 0 | All missing data imputed using statistical best practices |
| **`Joining_Date` Type** | `object` (string) | `datetime64[ns]` | Converted to datetime for time-series / tenure analysis |
| **`Salary` Type** | `object` (e.g. `"$110,000"`) | `float64` | Stripped `$` and `,` to enable mathematical calculations |
| **`Age` Minimum / Maximum** | -25 / 150 | 22 / 60 | Domain errors replaced with median valid age |
| **`Salary` Minimum / Maximum** | -$5,000 / $150,000 | $35,000 / $150,000 | Negative and zero salaries replaced with median salary |
| **Unique Departments** | 14 (dirty) | 5 (clean) | Casing, whitespace, and "HR" abbreviations standardized |
| **Unique Cities** | 17 (dirty) | 6 (clean) | Casing, spaces, and "Bangalore" alias standardized |
| **Unique Genders** | 9 (dirty) | 3 (clean) | "M"/"F", casing, and whitespace standardized |

---

## 📁 Project Structure

```text
Data-Cleaning-Task/
│
├── data/
│   ├── raw_dataset.csv             # Raw input dataset with intentional quality issues (420 rows)
│   └── cleaned_dataset.csv         # Final preprocessed, analysis-ready dataset (400 rows)
│
├── notebooks/
│   └── data_cleaning.ipynb         # Interactive Jupyter Notebook with step-by-step explanations
│
├── src/
│   └── cleaning.py                 # Modular, executable Python script for automated pipeline
│
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # List of required Python packages
└── .gitignore                      # Git configuration to ignore cache and temporary files
```

---

## ⚙️ How to Install Dependencies

1. Clone or download this repository.
2. Open your terminal or PowerShell and navigate to the project directory:
   ```bash
   cd Data-Cleaning-Task
   ```
3. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### Option 1: Run the Interactive Jupyter Notebook
Launch Jupyter Notebook from the project folder:
```bash
jupyter notebook
```
Navigate to `notebooks/data_cleaning.ipynb` and click **Kernel > Restart & Run All**.

### Option 2: Run the Automated Python Script
Execute the modular cleaning script directly from the terminal:
```bash
python src/cleaning.py
```

---

## 📋 Expected Output

When running `src/cleaning.py`, you will see a structured terminal log:
```text
======================================================================
      DATA CLEANING AND PREPROCESSING PIPELINE STARTING
======================================================================
[INFO] Successfully loaded dataset from: .../data/raw_dataset.csv
[INFO] Raw dataset shape: 420 rows, 9 columns

[STEP 1] Handling missing values (Initial total nulls: 133)...
  - Imputed missing 'Department' with mode: 'Marketing'
  - Imputed missing 'Gender' with mode: 'Male'
  - Imputed missing 'City' with mode: 'Delhi'
  - Imputed missing 'Age' with median: 43.0
  - Imputed missing 'Experience_Years' with median: 18.0

[STEP 2] Removing duplicates...
  - Duplicate rows found: 20
  - Successfully dropped 20 duplicate rows.

[STEP 3] Standardizing text and categorical values...
  - Standardized 'Department' unique values: ['Engineering', 'Finance', 'Human Resources', 'Marketing', 'Sales']
  - Standardized 'City' unique values: ['Bengaluru', 'Chennai', 'Delhi', 'Hyderabad', 'Mumbai', 'Pune']
  - Standardized 'Gender' unique values: ['Female', 'Male', 'Other']

[STEP 4] Correcting data types...
  - 'Joining_Date' converted to: datetime64[ns]
  - 'Salary' converted to: float64 (Missing values filled with median: $94,000.00)

[STEP 5] Handling invalid numerical anomalies...
  - Detected 4 invalid Age records (<18 or >65). Replaced with median: 43
  - Detected 3 non-positive Salary records (<= 0). Replaced with median: $94,000.00
  - Detected 2 negative Experience records. Corrected to 0.
  - Detected 12 Experience records exceeding Age - 18. Capped to max logical experience.

[SUCCESS] Cleaned dataset saved to: .../data/cleaned_dataset.csv

======================================================================
                   DATA CLEANING SUMMARY
======================================================================
  * Number of original rows:          420
  * Number of duplicate rows removed: 20
  * Number of missing values handled: 133
  * Number of columns processed:      9
  * Number of final rows:             400
  * Remaining null values:            0
  * Final dataset status:             Clean, Validated & Ready for Modeling
======================================================================
```

---

## 💡 Key Learnings
1. **Garbage In, Garbage Out**: Data cleaning is often 70-80% of a data scientist's time; models are only as good as the data fed into them.
2. **Mean vs. Median**: Mean is heavily distorted by extreme values or negative anomalies; median is a robust measure of central tendency for imputation.
3. **Categorical Normalization**: Casing, whitespace, and synonym differences silently break groupings (`groupby`, `value_counts`), making string normalization essential.
4. **Data Type Integrity**: Storing numbers or dates as objects blocks arithmetic, time-delta calculations, and model ingestion.
5. **Domain Validation**: Data cleaning is not just about removing `NaN`s; it involves domain-specific logic (e.g., verifying that `Experience <= Age - 18` and `Age >= 18`).
