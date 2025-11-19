# ==========================================================
# 🧠 CUSTOMER CHURN PREDICTION — ADVANCED VERSION (TUNED)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

#  LOAD & CLEAN DATA

df = pd.read_csv("Customer Churn.csv")

# Drop CustomerID if present
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode Target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})


#   EDA 

#numerical and Categotrical Column
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

print("\nNumerical Columns:", num_cols)
print("\nCategorical Columns:", cat_cols)

#  Churn Distribution
plt.figure(figsize=(5,4))
df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#0077B6','#DC3545'])
plt.title('Churn Distribution')
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Numeric Features Only)')
plt.show()

# --- Tenure vs Churn ---
# plt.figure(figsize=(6,4))
# sns.histplot(data=df, x='tenure', hue='Churn', bins=30, kde=True)
# plt.title('Tenure vs Churn')
# plt.show()

# --- Contract Type vs Churn ---
# plt.figure(figsize=(5,3))
# sns.barplot(data=df, x='Contract', y='Churn', palette='magma')
# plt.title('Contract Type vs Churn')
# plt.show()

# --- Payment Method vs Churn ---
# plt.figure(figsize=(7,3))
# sns.barplot(data=df, x='PaymentMethod', y='Churn', palette='viridis')
# plt.title('Payment Method vs Churn')
# plt.xticks(rotation=45)
# plt.show()



#  FEATURE ENGINEERING


for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
df['IsLongTermCustomer'] = np.where(df['tenure'] > 24, 1, 0)
print("✅New Feature created!")

#  TRAIN-TEST SPLIT

X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


#  BASE MODELS & Multiple Models + Boosting

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    results.append((name, acc, roc))
    print(f"\n📊 {name}")
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")


# HYPERPARAMETER TUNING — BOOSTING 


#  XGBoost
xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='roc_auc', n_jobs=1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_

print("\n🔍 Best XGBoost Params:", xgb_grid.best_params_)

#  LightGBM
lgb = LGBMClassifier(random_state=42)
lgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [-1, 4, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100]
}

lgb_grid = GridSearchCV(lgb, lgb_params, cv=3, scoring='roc_auc', n_jobs=1)
lgb_grid.fit(X_train, y_train)
best_lgb = lgb_grid.best_estimator_

print("\n🔍 Best LightGBM Params:", lgb_grid.best_params_)

# Evaluate tuned models
for name, model in {"XGBoost (Tuned)": best_xgb, "LightGBM (Tuned)": best_lgb}.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    results.append((name, acc, roc))
    print(f"\n⚙️ {name}")
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")


#  selecting best model 

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "ROC_AUC"]).sort_values(by="ROC_AUC", ascending=False)
print("\n🚀MODEL PERFORMANCE ")
print(results_df)

best_model_name = results_df.iloc[0, 0]
best_model = None
if "XGBoost" in best_model_name:
    best_model = best_xgb
elif "LightGBM" in best_model_name:
    best_model = best_lgb
else:
    best_model = models[best_model_name]

print(f"\n🏆 Best Model Selected: {best_model_name}")


#  Evaluation
y_pred_final = best_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nClassification Report:\n", classification_report(y_test, y_pred_final))
print(f"✅ Final Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"✅ Final ROC-AUC: {roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1]):.4f}")


#  Saving the best model
model_filename = f"best_churn_model_{best_model_name.replace(' ', '_')}.pkl"
joblib.dump(best_model, model_filename)
joblib.dump(list(X.columns), "feature_columns.pkl")

joblib.dump(scaler, "scaler.pkl")

print(f"\n💾 Saved model as: {model_filename}")
print("💾 Saved scaler as: scaler.pkl")
print("\n🎉 All Steps Completed Successfully!")
