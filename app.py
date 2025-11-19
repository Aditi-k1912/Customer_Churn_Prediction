
# 📊 CUSTOMER CHURN DASHBOARD (IMPROVED UI/UX)

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import altair as alt


#  LOAD MODEL, SCALER & FEATURE COLUMNS



MODEL_FILE = "best_churn_model_Logistic_Regression.pkl"
SCALER_FILE = "scaler.pkl"
FEATURE_FILE = "feature_columns.pkl"

@st.cache_resource
def load_model_assets():
    """Loads and caches the model, scaler, and feature columns."""
    if not os.path.exists(MODEL_FILE):
        st.error(f"❌ Model file not found! Expected: `{MODEL_FILE}`. Please run your training script first.")
        st.stop()
    if not os.path.exists(SCALER_FILE):
        st.error("❌ Scaler file not found! (scaler.pkl). Please run your training script.")
        st.stop()
    if not os.path.exists(FEATURE_FILE):
        st.error("❌ Feature columns file not found! (feature_columns.pkl). Please run your training script.")
        st.stop()

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_columns = joblib.load(FEATURE_FILE)
    return model, scaler, feature_columns

model, scaler, feature_columns = load_model_assets()

@st.cache_data
def load_csv(csv_path):
    """Loads and caches the raw data from CSV."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

df = load_csv("Customer Churn.csv")

#  PAGE CONFIGURATION + STYLING
st.set_page_config(page_title="Customer Churn Prediction",
                   layout="wide",
                   page_icon="👋")
st.markdown("""
    <style>
        /* --- Main App Styling --- */
        /* This now handles the main app background, effectively the default Streamlit grey */
        .main {
            background-color: #f0f2f6; 
        }
        
        /* --- Sidebar Styling --- */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 2px solid #f0f2f6;
            color: #111111; /* Default dark text for sidebar */
        }
        [data-testid="stSidebar"] h1 {
            color: #0072B5;
            font-weight: 700;
        }

        /* --- FIX FOR EXPANDERS --- */
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
             color: #111111 !important;
             background-color: transparent !important;
        }
        [data-testid="stSidebar"] label {
             color: #111111 !important;
        }
        
        /* --- Main Content Area - REMOVED BACKGROUND IMAGE --- */
        /* Removed the [data-testid="stAppViewContainer"] > section rule that set the background image. */
        /* The .main rule now effectively sets the overall app background. */

        /* --- Reset h1, h2 styling for default background --- */
        h1, h2 {
            color: unset; /* Revert to default text color */
            text-shadow: none; /* Remove any lingering shadow */
        }
        
        /* --- Content Blocks (Tabs, Metrics) --- */
        /* Note: These will still have a translucent white background over the main app background */
        [data-testid="stTabs"], .metric-box, .prediction-card {
            background-color: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #111 !important; /* Force dark text inside */
        }
        
        /* --- Metric Boxes in Visuals Tab --- */
        .metric-box {
            text-align: center;
            transition: all 0.3s ease;
            color: #111; /* Force dark text */
        }
        .metric-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }
        .metric-box h3 {
            color: #4a4a4a; /* Dimmed title color */
            font-size: 1.1rem;
            font-weight: 600;
        }
        .metric-box h2 {
            color: #0072B5; /* Primary color for value */
            font-size: 2.5rem;
            font-weight: 700;
            margin-top: -10px;
        }

        /* --- Prediction Card Styling --- */
        .prediction-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px;
            color: #111; /* Force dark text */
        }
        .prediction-card .result-text {
            font-size: 1.5rem;
            font-weight: 600;
            text-align: center;
            color: #111; /* Force dark text */
        }
        .prediction-card .proba-text {
            font-size: 1.1rem;
            color: #555;
            text-align: center;
        }
        .prediction-card .stProgress {
            width: 80%;
            margin-top: 15px;
        }

        /* --- Tab Styling --- */
        [data-testid="stTabs"] [data-testid="stMarkdownContainer"] p {
            font-size: 1.1rem;
            color: #333; /* Dark text for paragraphs inside tabs */
        }
        [data-testid="stTabs"] h2, [data-testid="stTabs"] h4 {
            color: #111; /* Force dark text for headers */
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            font-size: 1.1rem;
            font-weight: 600;
            background-color: transparent;
            color: #333; /* Dark text for tab labels */
        }
        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
            background-color: rgba(255, 255, 255, 0.5);
            border-radius: 8px 8px 0 0;
            color: #0072B5; /* Highlight color for selected tab */
        }
        
        /* --- Fix for st.info box --- */
        [data-testid="stInfo"] {
            color: #0c5460; /* Dark text for info box */
            background-color: rgba(209, 236, 241, 0.85); /* Light blue bg */
            border: 1px solid rgba(190, 229, 235, 0.5);
            border-radius: 12px;
            backdrop-filter: blur(5px);
        }

        /* --- Button Styling --- */
        .stButton>button {
            background-color: #0072B5;
            color: white;
            border-radius: 8px;
            font-weight: 600;
            padding: 12px 20px;
            width: 100%;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #005A8C;
            box-shadow: 0 4px 10px rgba(0,114,181,0.3);
        }

    </style>
""", unsafe_allow_html=True)



#  SIDEBAR INPUT FORM
with st.sidebar:
    st.markdown("<h1><span style='color: #0072B5;'>Client</span> Details</h1>", unsafe_allow_html=True)
    st.write("Enter the customer's information to predict churn probability.")

    st.markdown("---")

    # Grouping inputs for better organization
    with st.expander("**Account Information**", expanded=True):
        tenure = st.slider("Tenure (months)", 0, 72, 24)
        contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    with st.expander("**Charges**"):
        monthly_charges = st.number_input("Monthly Charges ($)", 10.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)

    with st.expander("**Services & Demographics**"):
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    st.markdown("---")
    predict_button = st.button("🔍 Predict Churn", key="predict")



st.title("👋 Customer Churn Prediction")
st.write("This dashboard predicts customer churn and provides key insights into your dataset.")


tab1, tab2 = st.tabs(["🔮 Prediction Result", "📊 Visual Insights"])

# 🔮 PREDICTION TAB

with tab1:
    if predict_button:
      
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
        payment_map = {
            "Electronic check": 0, "Mailed check": 1,
            "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
        }
        yes_no = {"Yes": 1, "No": 0}

        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'Contract': [contract_map[contract_type]],
            'InternetService': [internet_map[internet_service]],
            'PaymentMethod': [payment_map[payment_method]],
            'SeniorCitizen': [senior_citizen],
            'Partner': [yes_no[partner]],
            'Dependents': [yes_no[dependents]],
            'AvgChargesPerMonth': [total_charges / (tenure + 1e-6)], # Avoid division by zero
            'IsLongTermCustomer': [1 if tenure > 24 else 0]
        })

  
        input_aligned = input_data.reindex(columns=feature_columns, fill_value=0)
        X_scaled = scaler.transform(input_aligned)

        
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1] # Probability of Churn (class 1)

        st.markdown("## 🔮 Prediction Result")
        
        if prediction == 1:
            result_text = "<span style='color: #D32F2F;'>Likely to CHURN</span>"
            proba_text = f"**{probability*100:.2f}%** probability of churning"
            progress_color = "#D32F2F"
        else:
            result_text = "<span style='color: #388E3C;'>Likely to STAY</span>"
            proba_text = f"**{(1-probability)*100:.2f}%** probability of staying"
            probability = 1 - probability # Invert for 'stay' progress bar
            progress_color = "#388E3C"
        
        st.markdown(f"""
            <div class="prediction-card">
                <div class="result-text">{result_text}</div>
                <div class="proba-text">{proba_text}</div>
                <progress value="{probability}" max="1" style="width: 80%; height: 20px; accent-color: {progress_color};"></progress>
            </div>
        """, unsafe_allow_html=True)

    else:
        st.info("ℹ️ Please fill in the customer details in the sidebar and click 'Predict Churn'.")


#  VISUAL INSIGHTS TAB

with tab2:
    if df is not None:
       
        df_display = df.copy()
        df_display['Churn Status'] = df_display['Churn'].map({0: 'Stayed', 1: 'Churned'})
        
        st.markdown("## 📊 Key Performance Indicators")
        
       
        col1, col2, col3 = st.columns(3)
        with col1:
            churn_rate = df['Churn'].mean() * 100
            st.markdown(f"<div class='metric-box'><h3>Overall Churn Rate</h3><h2>{churn_rate:.2f}%</h2></div>", unsafe_allow_html=True)
        with col2:
            avg_tenure = df['tenure'].mean()
            st.markdown(f"<div class='metric-box'><h3>Avg. Tenure</h3><h2>{avg_tenure:.1f} <span style='font-size: 1.5rem; color: #555;'>months</span></h2></div>", unsafe_allow_html=True)
        with col3:
            avg_charge = df['MonthlyCharges'].mean()
            st.markdown(f"<div class='metric-box'><h3>Avg. Monthly Charge</h3><h2>${avg_charge:.2f}</h2></div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 📈 Customer Demographics")
        
        colA, colB = st.columns(2)
        
        
        with colA:
            st.markdown("#### Churn Distribution")
            # This logic is fine as it creates its own 'Churn' text column
            chart_data = df['Churn'].value_counts().reset_index()
            chart_data.columns = ['Churn', 'count']
            chart_data['Churn'] = chart_data['Churn'].map({0: 'Stayed', 1: 'Churned'})

            base = alt.Chart(chart_data).encode(
                theta=alt.Theta("count:Q", stack=True)
            ).properties(
                title='Overall Churn vs. Stay'
            )
            
            pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(
                color=alt.Color("Churn:N", scale=alt.Scale(domain=['Churned', 'Stayed'], range=['#D32F2F', '#388E3C'])),
                order=alt.Order("count:Q", sort="descending"),
                tooltip=["Churn", "count", alt.Tooltip("count", format=".1%")]
            )
            
            text = base.mark_text(radius=140).encode(
                text=alt.Text("count:Q", format=".1%"),
                order=alt.Order("count:Q", sort="descending"),
                color=alt.value("black")  # Set text color
            )

            st.altair_chart(pie + text, use_container_width=True)

        
        with colB:
            st.markdown("#### Monthly Charges vs. Churn")
            
            chart = alt.Chart(df_display).mark_boxplot(extent='min-max').encode(
                # Use 'Churn Status' for the X-axis
                x=alt.X('Churn Status:N', axis=alt.Axis(title="Customer Status")),
                y=alt.Y('MonthlyCharges:Q', title='Monthly Charges ($)'),
                color=alt.Color('Churn Status:N',
                                scale=alt.Scale(domain=['Stayed', 'Churned'], range=['#388E3C', '#D32F2F']),
                                legend=alt.Legend(title="Status")
                               ),
                tooltip=[
                    alt.Tooltip('Churn Status:N', title='Status'),
                    alt.Tooltip('min(MonthlyCharges):Q', title='Min Charges', format="$.2f"),
                    alt.Tooltip('max(MonthlyCharges):Q', title='Max Charges', format="$.2f"),
                    alt.Tooltip('median(MonthlyCharges):Q', title='Median Charges', format="$.2f")
                ]
            ).properties(
                title='Impact of Monthly Charges on Churn'
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 🔍 Explore Data")
        st.dataframe(df_display.head(10), use_container_width=True)

    else:
        st.info("ℹ️ `Customer Churn.csv` not found. Please upload the file to see the visual insights.")