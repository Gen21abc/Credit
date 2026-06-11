import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Artifacts
@st.cache_resource
def load_artifacts():
    model = pickle.load(open('credit_risk_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    feature_names = pickle.load(open('feature_names.pkl', 'rb'))
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

st.title("Credit Lending Risk Prediction System")
st.write("Predict loan applicant risk tier using machine learning.")

# Form for user inputs
with st.form("prediction_form"):
    st.header("1. Personal Info")
    col1, col2, col3 = st.columns(3)
    # Using sensible defaults and ranges
    age = col1.number_input("Age", min_value=18, max_value=100, value=30)
    gender = col2.selectbox("Gender", options=["M", "F", "Other"])
    marital_status = col3.selectbox("Marital Status", options=["Single", "Married"])
    education = col1.selectbox("Education", options=["12TH", "GRADUATE", "POST-GRADUATE", "OTHERS"])
    income = col2.number_input("Net Monthly Income", min_value=0, value=50000)
    emp_tenure = col3.number_input("Employment Tenure (Months)", min_value=0, value=24)

    st.header("2. Loan History")
    col4, col5, col6 = st.columns(3)
    total_tl = col4.number_input("Total Trade Lines (TLs)", min_value=0, value=5)
    active_tl = col5.number_input("Active TLs", min_value=0, value=2)
    missed_pmnt = col6.number_input("Missed Payments", min_value=0, value=0)
    age_oldest_tl = col4.number_input("Age Oldest TL (Months)", min_value=0, value=48)
    pl_count = col5.number_input("Personal Loan (PL) Count", min_value=0, value=1)
    cc_count = col6.number_input("Credit Card (CC) Count", min_value=0, value=1)
    hl_flag = col4.selectbox("Home Loan (HL) Flag", [0, 1])
    gl_flag = col5.selectbox("Gold Loan (GL) Flag", [0, 1])

    st.header("3. Risk Signals")
    col7, col8, col9 = st.columns(3)
    delq_count = col7.number_input("Delinquency Count", min_value=0, value=0)
    max_delq_level = col8.number_input("Max Delinquency Level", min_value=0, value=0)
    time_since_pmnt = col9.number_input("Time Since Payment (Days)", min_value=0, value=15)
    enquiries = col7.number_input("Total Enquiries", min_value=0, value=1)
    last_prod_enq = col8.selectbox("Last Product Enquired", options=["PL", "CC", "HL", "AL", "Others"])
    credit_score = col9.number_input("Credit Score", min_value=300, max_value=900, value=750)

    submitted = st.form_submit_button("Predict Risk Tier")

if submitted:
    # 1. Gather all inputs into a dictionary
    # (Note: In a real-world scenario, you map these direct UI inputs EXACTLY 
    # to the `feature_names` generated during feature selection)
    input_data = {
        'AGE': age,
        'NETMONTHLYINCOME': income,
        'Time_With_Curr_Empr': emp_tenure,
        'Total_TL': total_tl,
        'Tot_Active_TL': active_tl,
        'Tot_Missed_Pmnt': missed_pmnt,
        'Age_Oldest_TL': age_oldest_tl,
        'PL_TL': pl_count,
        'CC_TL': cc_count,
        'Home_TL': hl_flag,
        'Gold_TL': gl_flag,
        'num_dbt': delq_count, # Approximating field name mapped to delinquent
        'max_delinquency_level': max_delq_level,
        'time_since_recent_payment': time_since_pmnt,
        'tot_enq': enquiries,
        'Credit_Score': credit_score,
        
        # Categorical maps (mock encoding to match training preprocessing)
        'GENDER': 0 if gender == 'F' else 1,
        'MARITALSTATUS': 1 if marital_status == 'Married' else 0,
        'EDUCATION': 1 if education == 'GRADUATE' else 0,
        'last_prod_enq2': 1,
        'first_prod_enq2': 1
    }
    
    # 2. Build the exact feature row needed by the model
    # Fill missing columns from feature_names with a safe default (e.g., 0)
    row = pd.DataFrame(columns=feature_names)
    row.loc[0] = [input_data.get(col, 0) for col in feature_names]

    # 3. Scale inputs
    row_scaled = scaler.transform(row)

    # 4. Predict
    prediction = model.predict(row_scaled)[0]
    probabilities = model.predict_proba(row_scaled)[0]

    # Target was encoded as P1->0, P2->1, P3->2, P4->3
    tier_map = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}
    label_map = {0: "Approve", 1: "Moderate", 2: "High Risk", 3: "Reject"}
    color_map = {0: "green", 1: "blue", 2: "orange", 3: "red"}

    pred_tier = tier_map[prediction]
    pred_label = label_map[prediction]
    pred_color = color_map[prediction]

    # 5. Display Outputs
    st.markdown("---")
    st.subheader("Prediction Results")
    
    st.markdown(
        f"<h2 style='text-align: center; color: {pred_color};'>"
        f"Tier: {pred_tier} ({pred_label})"
        f"</h2>", 
        unsafe_allow_html=True
    )
    
    st.write("**Class Probabilities:**")
    prob_df = pd.DataFrame(
        [probabilities], 
        columns=["P1 (Approve)", "P2 (Moderate)", "P3 (High Risk)", "P4 (Reject)"]
    )
    st.dataframe(prob_df.style.format("{:.2%}"))