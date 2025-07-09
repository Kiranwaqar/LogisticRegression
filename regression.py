import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            roc_curve,roc_auc_score)

# Load the dataset
df = pd.read_csv('churn-bigml-80 - churn-bigml-80.csv')

# Handle missing data
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical variables
categorical_cols = ['International plan', 'Voice mail plan','Churn']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode 'State'
df=pd.get_dummies(df, columns=['State'], drop_first=True)

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Normalize numerical features
scaler=StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model coefficients and odd ratio
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Odds Ratio' : np.exp(model.coef_[0])
})
print("\nModel Coefficients and Odds Ratios:\n")
print(coefficients.sort_values(by="Odds Ratio",ascending=False))

# Evaluate the model 
print("Evaluation Metrics:")
print(f"Accuracy : {accuracy_score(y_test,y_pred):.2f}")
print(f"Precision : {precision_score(y_test,y_pred):.2f}")
print(f"Recall : {recall_score(y_test,y_pred):.2f}")

# ROC Curve
y_probs = model.predict_proba(X_test)[:,1]
fpr,tpr,thresholds = roc_curve(y_test,y_probs)
roc_auc = roc_auc_score(y_test,y_probs)

plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,color='darkorange',label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],linestyle='--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid(True)
plt.show()