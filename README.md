# Churn Prediction using Logistic Regression

This project implements a **Logistic Regression** model using Python and scikit-learn to predict whether a customer will **churn** (leave the service) based on various features. It includes data preprocessing, model training, evaluation, and interpretation of model coefficients.

---

## Task Objective

- Load and preprocess a binary classification dataset (churn data).
- Train a logistic regression model.
- Interpret model coefficients and odds ratios.
- Evaluate model performance using:
  - Accuracy
  - Precision
  - Recall
  - ROC Curve

---

## Demo

https://github.com/user-attachments/assets/e7bbe38a-1d3e-4b70-89ee-3c71ad944a06

## Tools & Libraries

- Python
- pandas
- scikit-learn
- matplotlib

---

##  Dataset

The dataset includes features like:

- State
- Account length
- International plan
- Voice mail plan
- Call durations and charges (day, evening, night, international)
- Customer service calls
- Target column: `Churn` (Yes/No)

---

## How it Works

### 1. **Preprocessing**
- Handle missing values
- Encode categorical features (`LabelEncoder`, `get_dummies`)
- Normalize features using `StandardScaler`

### 2. **Model Training**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```
### 3. **Model Evaluation**

- Accuracy, Precision, Recall
- ROC Curve with AUC score

### 4. **Model Interpretation**
```python
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Odds Ratio': np.exp(model.coef_[0])
})
```
## Example Results
```yaml
Accuracy:      0.84
Precision:     0.43
Recall:        0.24
ROC AUC Score: 0.78
```
These metrics indicate that the model performs reasonably well in distinguishing churn vs. non-churn but could improve on recall.

## Visualization
ROC Curve is plotted using matplotlib:
```python

plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--')
```
## Future Improvements
- Try other models: Random Forest, XGBoost
- Use class balancing techniques (SMOTE or class_weight)
- GridSearchCV for hyperparameter tuning
