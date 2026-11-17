# Module 3: Regression (Supervised Learning)

## Chapters & Problems
1. **Basics of Regression & Evaluation Metrics (`SDMELR07`, `SDMELR08`)**
   - Cost Functions: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE)
   - Goodness of Fit: $R^2$ Score ($1 - \frac{SS_{res}}{SS_{tot}}$)

2. **Linear, Multiple & Polynomial Regression (`KJSHDBC01`-`03`)**
   - Closed-form OLS solutions vs Gradient Descent
   - Polynomial Feature Expansion (`PolynomialFeatures(degree=2)`)

3. **Regularization Techniques (`KJSHDBC04`-`06`, `08`)**
   - Ridge ($L_2$ Penalty: $\lambda \sum \beta_j^2$)
   - Lasso ($L_1$ Penalty: $\lambda \sum |\beta_j|$ with sparse feature selection)
   - Cross-validation with TimeSeriesSplit / KFold

4. **Projects & Assignments**
   - `DJFWQ03_car_price_prediction.py`: Assignment 3 Automobile Regression before/after PCA
   - `house_price_prediction_pipeline.py`: Linear Regression Pipeline with full EDA
   - `crop_yield_anomaly_regression.py`: Assignment 4 Temporal Crop Yield Anomaly Modeling
   - Status: 🟢 100% Solved
