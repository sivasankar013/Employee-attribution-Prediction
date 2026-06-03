# Employee Attrition Prediction Report

## 1. Executive Summary

This project analyzes the IBM HR employee attrition dataset and builds a baseline machine learning model to predict whether an employee will leave the company.

The notebook uses a straightforward preprocessing pipeline, trains a `RandomForestClassifier`, and reports an accuracy of about `0.816` on the test split. While this is a reasonable first baseline, the problem is imbalanced, so accuracy alone is not enough to judge model quality. A deeper evaluation shows that the model is much better at identifying employees who stay than employees who leave.

## 2. Dataset Overview

- Dataset file: `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- Number of rows: `1470`
- Number of columns: `35`
- Target column: `Attrition`
- Missing values: none detected

### Target Distribution

- `No`: `1233` employees
- `Yes`: `237` employees
- Attrition rate: about `16.1%`

This is an imbalanced classification problem, which makes metrics such as recall, precision, F1-score, and ROC-AUC especially important.

## 3. Notebook Workflow

The notebook follows this general flow:

1. Load the CSV into a pandas DataFrame.
2. Convert binary categorical fields into numeric values:
   - `Attrition`
   - `Gender`
   - `Over18`
   - `OverTime`
3. One-hot encode these categorical columns:
   - `BusinessTravel`
   - `Department`
   - `EducationField`
   - `JobRole`
   - `MaritalStatus`
4. Convert any boolean columns to integers.
5. Drop identifier or constant columns:
   - `EmployeeNumber`
   - `EmployeeCount`
   - `Over18`
   - `StandardHours`
6. Visualize distributions with histograms.
7. Train a `RandomForestClassifier`.
8. Evaluate the model using accuracy.
9. Plot feature importances.

## 4. Data Preparation Assessment

The preprocessing choices are mostly reasonable for a baseline model:

- Binary columns are converted correctly into numeric labels.
- One-hot encoding is appropriate for categorical features with more than two classes.
- Constant and identifier-like columns are removed.

However, there are a few caveats:

- The notebook uses a manual series of transformations instead of a reusable preprocessing pipeline.
- `train_test_split` is used without `random_state`, so results are not reproducible.
- `train_test_split` is also used without `stratify=y`, which is important because the target is imbalanced.
- The line `df = df.map(lambda x: 1 if x is True else 0 if x is False else x)` is unusual and could be fragile if the DataFrame contains mixed dtypes or unexpected values.

## 5. Model Training

The notebook trains a `RandomForestClassifier` using the default parameters and fits it on an 80/20 train-test split.

### Reported Result in Notebook

- Test accuracy: approximately `0.8163`

This is a decent baseline, but it needs to be interpreted carefully because the majority class alone already provides a strong naive benchmark.

## 6. Re-Evaluation Notes

To better understand model quality, the same style of pipeline was re-evaluated with a fixed random seed and stratified split.

### Observed Metrics

- Accuracy: `0.8435`
- ROC-AUC: `0.7704`
- Majority-class baseline accuracy: `0.8401`

### Confusion Matrix

- True negatives: `244`
- False positives: `3`
- False negatives: `43`
- True positives: `4`

### Classification Report

- Precision for attrition class (`1`): `0.571`
- Recall for attrition class (`1`): `0.085`
- F1-score for attrition class (`1`): `0.148`

### Interpretation

The model performs well at identifying employees who do not leave, but it misses most employees who do leave. That makes it weak for practical attrition prevention use cases, where detecting at-risk employees is the main goal.

## 7. Feature Importance Insights

The random forest ranked the following features as the most important:

- `MonthlyIncome`
- `Age`
- `TotalWorkingYears`
- `DailyRate`
- `HourlyRate`
- `MonthlyRate`
- `DistanceFromHome`
- `OverTime`
- `YearsWithCurrManager`
- `YearsAtCompany`

### Business Interpretation

These features suggest that attrition is associated with:

- Compensation level
- Career stage and tenure
- Workload or travel-related factors
- Time spent in current role or with current manager
- Work-life pressure signals such as overtime

This is consistent with common HR attrition patterns.

## 8. Strengths of the Notebook

- Uses a real-world dataset with meaningful business value.
- Applies practical preprocessing steps.
- Includes a machine learning model and feature importance analysis.
- Produces an initial predictive baseline quickly.

## 9. Weaknesses and Risks

- Accuracy is treated as the main metric despite class imbalance.
- No cross-validation is used.
- No hyperparameter tuning is performed.
- No threshold tuning is performed for the attrition class.
- No calibration or probability analysis is included.
- No comparison against simpler models such as logistic regression or decision trees.
- Feature importance from random forests can be unstable when features are correlated.

## 10. Recommended Improvements

If the goal is to turn this into a stronger attrition prediction project, the next steps should be:

1. Use a reproducible train/test split with `random_state`.
2. Use `stratify=y` in the split.
3. Evaluate with:
   - precision
   - recall
   - F1-score
   - ROC-AUC
   - confusion matrix
4. Try class balancing approaches:
   - `class_weight='balanced'`
   - oversampling methods such as SMOTE
   - undersampling if appropriate
5. Compare multiple models:
   - logistic regression
   - random forest
   - gradient boosting
   - XGBoost or LightGBM if available
6. Tune hyperparameters with cross-validation.
7. Investigate feature engineering, especially:
   - tenure-related ratios
   - salary-to-role comparisons
   - interaction between overtime and workload
8. Add explainability using permutation importance or SHAP.

## 11. Final Conclusion

This notebook is a solid baseline attrition prediction project, but it is not yet a production-quality classifier. The current workflow demonstrates that the data contains useful signals, especially around income, age, tenure, and overtime, but the model is not strong at identifying the minority attrition class.

In short:

- The data is clean and usable.
- The modeling approach is simple and valid as a first pass.
- The evaluation needs to be expanded.
- Better recall for attrition detection should be the main objective for future work.

