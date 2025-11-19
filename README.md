

# 📊 Customer Churn Prediction

## An End-to-End Machine Learning Project with Streamlit Dashboard

This project predicts whether a customer is likely to **churn (leave the service)** using machine learning techniques. It includes data preprocessing, feature engineering, model training, evaluation, and a fully interactive **Streamlit web application** for real-time churn prediction.



## 🚀 Project Overview

Customer churn is a major challenge for service-based industries. This project analyzes customer behavior, subscription details, service usage, and billing patterns to predict churn probability.
It helps businesses identify at-risk customers and take preventive actions.



## 🧠 Key Features

* Complete ML pipeline: Cleaning, encoding, scaling, training, and evaluation.
 
* Feature Engineering:
* Average charges per month
* Long-term customer indicator
* Label encoding of categorical attributes
* 
* Model Comparison:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Gradient Boosting
  * AdaBoost
  * XGBoost (Tuned)
  * LightGBM (Tuned)
* Model selection based on accuracy & ROC-AUC score.
* Interactive Streamlit Dashboard:

  * Input customer attributes
  * Predict churn probability
  * Visual insights & KPIs
  * Pie charts, box plots, and dataset exploration

---

## 📂 Project Structure

```
├── app.py                         # Streamlit dashboard
├── Customer.py                    # ML pipeline + model training + tuning
├── Customer Churn.csv             # Dataset
├── best_churn_model_*.pkl         # Saved ML model
├── scaler.pkl                     # StandardScaler file
├── feature_columns.pkl            # Feature list used for prediction
└── README.md                      # Project documentation


## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost, LightGBM
* Matplotlib, Seaborn
* Streamlit
* Joblib

---

## 📈 Model Performance

The best-performing model (XGBoost or LightGBM, depending on tuning results) achieves:

* High Accuracy
* Strong ROC-AUC score
* Stable prediction performance across test data

Detailed metrics and confusion matrix are displayed in the notebook/output during training.



## 🌐 Streamlit Dashboard Features

✔ Predicts if a customer will churn
✔ Displays prediction probability
✔ KPI cards (Churn rate, Avg. tenure, Avg. charges)
✔ Visual insights (pie charts, box plots, correlations)
✔ Explore dataset (head preview)



## ▶️ How to Run the Project

### 1. Install Dependencies


pip install -r requirements.txt


### 2. Train the Model

Run:


python Customer.py


This will generate:

* `best_churn_model_*.pkl`
* `scaler.pkl`
* `feature_columns.pkl`

### 3. Launch Streamlit App


streamlit run app.py


## 📬 Output

The app predicts whether a customer will churn or stay, along with:

* Probability score
* Visualizations
* Customer insights
* Exploratory analytics


## ⭐ Future Improvements

* Add SHAP interpretability
* Deploy on cloud (Streamlit Cloud / Render)
* Add more visual analytics
* Improve UI with custom CSS



## 🧑‍💻 Author

Aditi Kalmegh
B.Tech – Artificial Intelligence & Data Science
YCCE, Nagpur


