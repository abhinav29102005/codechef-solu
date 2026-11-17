# Module 2: Data & Preprocessing

## Chapters & Topics
1. **Understanding Data and Data Cleaning (`SDMELR03` / `YTQAI24`-`YTQAI37`)**
   - Detecting and handling missing data (`dropna`, `fillna`, median/mean/mode imputation)
   - Outlier detection via IQR ($Q_1 - 1.5 \times IQR$, $Q_3 + 1.5 \times IQR$) and Z-score ($|Z| > 3$)
   - Data types: Numerical (discrete/continuous), Categorical (nominal/ordinal)

2. **Project: Clean and Explore Employee Dataset (`SDMELR04` / `YTQAI38`)**
   - Script: `employee_performance_analysis.py`
   - Statistical hypothesis testing: Two-sample independent t-test (`stats.ttest_ind`)
   - Categorical association: Chi-square test of independence (`stats.chi2_contingency`)
   - Correlation: Pearson correlation coefficient and p-value (`stats.pearsonr`)
   - Estimation: 95% Confidence Interval for mean performance (`stats.t.interval`)
   - Trend fitting: Ordinary Least Squares linear regression (`stats.linregress`)

3. **Data Preparation and Splitting (`SDMELR05` / `YTQAI39`-`YTQAI50`)**
   - Categorical Encoding: One-Hot Encoding (`pd.get_dummies`), Label Encoding (`LabelEncoder`)
   - Feature Scaling: Standardization ($z = \frac{x - \mu}{\sigma}$ via `StandardScaler`), Min-Max Normalization via `MinMaxScaler`
   - Data Splitting: Train/Test splitting (`train_test_split`, `test_size=0.2, random_state=42`), stratified sampling

4. **Project: Data Preparation and Splitting (`SDMELR06` / `YTQAI51`)**
   - Full end-to-end data transformation pipeline

5. **Assignment 2 (`THAPARML2` / `THRML01`-`THRML05`) & Practice Problems (`TPRMLPR01`)**
   - Status: 🟢 100% Solved
