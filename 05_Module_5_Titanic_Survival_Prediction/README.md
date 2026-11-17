# Module 5: Project : Titanic Survival Prediction

## Overview
- **Problem Code**: `POWERUR03`
- **Objective**: Predict passenger survival on the Titanic dataset, exceeding the 74.83% benchmark.
- **Result**: 🟢 **10 / 10 Tests Accepted (100% Pass)**

## Implementation Highlights (`solution_titanic.py`)
- Missing value imputation (Age via median grouped by Pclass/Sex, Embarked via mode, Fare via median)
- Feature Engineering: `FamilySize = SibSp + Parch + 1`, `IsAlone`, Title extraction (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`)
- Evaluated Models:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier (Best ensemble performance > 82% validation accuracy)
