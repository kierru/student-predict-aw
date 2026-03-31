"""
Student Dropout Prediction System
Jaya Jaya Institut - Early Warning System for Student Retention

This application predicts student dropout risk using XGBoost machine learning model
and provides actionable recommendations for academic support and intervention.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Student Dropout Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

MODEL_PATH = 'model/dropout_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
ENCODERS_PATH = 'model/encoders.pkl'
FEATURE_NAMES_PATH = 'model/feature_names.pkl'

MARITAL_STATUS_MAP = {
    1: "Single", 
    2: "Married", 
    3: "Widower", 
    4: "Divorced", 
    5: "Facto Union", 
    6: "Legally Separated"
}

GENDER_MAP = {0: "Female", 1: "Male"}

DAYTIME_MAP = {0: "Evening", 1: "Daytime"}

APPLICATION_MODE_MAP = {
    1: "1st phase - General contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - Special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - Special contingent (Madeira Island)",
    17: "2nd phase - General contingent",
    18: "3rd phase - General contingent",
    26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)"
}

COURSE_MAP = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)"
}

PREVIOUS_QUAL_MAP = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year of schooling - not completed",
    10: "11th year of schooling - not completed",
    12: "Other - 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year of schooling - not completed",
    19: "Basic education 3rd cycle (9th/10th/11th year)",
    38: "Basic education 2nd cycle (6th/7th/8th year)",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)"
}

NATIONALITY_MAP = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldova (Republic of)",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian"
}

QUALIFICATION_MAP = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year of schooling - not completed",
    10: "11th year of schooling - not completed",
    12: "Other - 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year of schooling - not completed",
    19: "Basic education 3rd cycle",
    38: "Basic education 2nd cycle",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)",
    44: "Other qualification"
}

OCCUPATION_MAP = {
    0: "Student",
    1: "Representatives of Legislative & Executive Bodies, Directors, Managers",
    2: "Specialists in Intellectual and Scientific Activities",
    3: "Intermediate Level Technicians and Professions",
    4: "Administrative staff",
    5: "Personal Services, Security and Safety Workers, Sellers",
    6: "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
    7: "Skilled Workers in Industry, Construction and Craftsmen",
    8: "Installation and Machine Operators, Assembly Workers",
    9: "Unskilled Workers",
    10: "Armed Forces Professions",
    90: "Other Situation",
    99: "Blank",
    101: "Armed Forces Officers",
    102: "Armed Forces Sergeants",
    103: "Other Armed Forces personnel",
    112: "Directors of Administrative and Commercial Services",
    114: "Hotel, Catering, Trade and Other Services Directors",
    121: "Specialists in Physical Sciences, Mathematics, Engineering",
    122: "Health professionals",
    123: "Teachers",
    124: "Specialists in Finance, Accounting, Administration, Public and Commercial Relations",
    131: "Intermediate Level Science and Engineering Technicians",
    132: "Intermediate Level Health Technicians and Professionals",
    134: "Intermediate Level Technicians from Legal, Social, Sports, Cultural Services",
    135: "Information and Communication Technology Technicians",
    141: "Office Workers, Secretaries, Data Processing Operators",
    143: "Data, Accounting, Statistical, Financial Services Operators",
    144: "Other Administrative Support Staff",
    151: "Personal Service Workers",
    152: "Sellers",
    153: "Personal Care Workers",
    154: "Protection and Security Services Personnel",
    161: "Market-oriented Farmers and Skilled Agricultural Workers",
    163: "Subsistence Farmers, Livestock Keepers, Fishermen, Hunters, Gatherers",
    171: "Skilled Construction Workers (except Electricians)",
    172: "Skilled Workers in Metallurgy and Metalworking",
    174: "Skilled Workers in Electricity and Electronics",
    175: "Workers in Food Processing, Woodworking, Clothing and Other Industries",
    181: "Fixed Plant and Machine Operators",
    182: "Assembly Workers",
    183: "Vehicle Drivers and Mobile Equipment Operators",
    192: "Unskilled Workers in Agriculture, Animal Production, Fisheries, Forestry",
    193: "Unskilled Workers in Extractive Industry, Construction, Manufacturing, Transport",
    194: "Meal Preparation Assistants",
    195: "Street Vendors (except food) and Street Service Providers"
}

APPLICATION_ORDER_MAP = {
    0: "1st Choice",
    1: "2nd Choice",
    2: "3rd Choice",
    3: "4th Choice",
    4: "5th Choice",
    5: "6th Choice",
    6: "7th Choice",
    7: "8th Choice",
    8: "9th Choice",
    9: "Last Choice"
}

# ============================================================================
# TEST SCENARIOS
# ============================================================================

SCENARIO_1 = {
    "name": "Scenario 1",
    "marital_status": 3,
    "application_mode": 39,
    "application_order": 5,
    "course": 9119,
    "daytime": 0,
    "previous_qual": 14,
    "previous_qual_grade": 70.0,
    "nacionality": 1,
    "mothers_qual": 6,
    "fathers_qual": 1,
    "mothers_occ": 9,
    "fathers_occ": 9,
    "admission_grade": 95.0,
    "displaced": 1,
    "international": 0,
    "gender": 0,
    "age": 28,
    "debtor": 1,
    "tuition_updated": 0,
    "scholarship": 0,
    "special_needs": 0,
    "sem1_enrolled": 6,
    "sem1_credited": 1,
    "sem1_evaluations": 3,
    "sem1_approved": 1,
    "sem1_grade": 8.0,
    "sem1_without_eval": 3,
    "sem2_enrolled": 4,
    "sem2_credited": 0,
    "sem2_evaluations": 2,
    "sem2_approved": 0,
    "sem2_grade": 5.5,
    "sem2_without_eval": 4,
    "unemployment_rate": 14.5,
    "inflation_rate": 2.8,
    "gdp": -1.5,
}

SCENARIO_2 = {
    "name": "Scenario 2",
    "marital_status": 2,
    "application_mode": 1,
    "application_order": 8,
    "course": 9147,
    "daytime": 0,
    "previous_qual": 15,
    "previous_qual_grade": 50.0,
    "nacionality": 25,
    "mothers_qual": 10,
    "fathers_qual": 9,
    "mothers_occ": 9,
    "fathers_occ": 9,
    "admission_grade": 80.0,
    "displaced": 1,
    "international": 1,
    "gender": 1,
    "age": 35,
    "debtor": 1,
    "tuition_updated": 0,
    "scholarship": 0,
    "special_needs": 1,
    "sem1_enrolled": 6,
    "sem1_credited": 0,
    "sem1_evaluations": 1,
    "sem1_approved": 0,
    "sem1_grade": 2.0,
    "sem1_without_eval": 5,
    "sem2_enrolled": 2,
    "sem2_credited": 0,
    "sem2_evaluations": 0,
    "sem2_approved": 0,
    "sem2_grade": 0.0,
    "sem2_without_eval": 2,
    "unemployment_rate": 15.8,
    "inflation_rate": 3.5,
    "gdp": -3.0,
}

SCENARIO_3 = {
    "name": "Scenario 3",
    "marital_status": 1,
    "application_mode": 1,
    "application_order": 0,
    "course": 9119,
    "daytime": 1,
    "previous_qual": 1,
    "previous_qual_grade": 130.0,
    "nacionality": 1,
    "mothers_qual": 2,
    "fathers_qual": 3,
    "mothers_occ": 123,
    "fathers_occ": 123,
    "admission_grade": 135.0,
    "displaced": 0,
    "international": 0,
    "gender": 1,
    "age": 21,
    "debtor": 0,
    "tuition_updated": 1,
    "scholarship": 1,
    "special_needs": 0,
    "sem1_enrolled": 6,
    "sem1_credited": 5,
    "sem1_evaluations": 6,
    "sem1_approved": 5,
    "sem1_grade": 14.5,
    "sem1_without_eval": 0,
    "sem2_enrolled": 6,
    "sem2_credited": 6,
    "sem2_evaluations": 6,
    "sem2_approved": 6,
    "sem2_grade": 15.0,
    "sem2_without_eval": 0,
    "unemployment_rate": 9.5,
    "inflation_rate": 1.5,
    "gdp": 1.5,
}

SCENARIO_4 = {
    "name": "Scenario 4",
    "marital_status": 1,
    "application_mode": 1,
    "application_order": 0,
    "course": 9500,
    "daytime": 1,
    "previous_qual": 1,
    "previous_qual_grade": 155.0,
    "nacionality": 1,
    "mothers_qual": 4,
    "fathers_qual": 4,
    "mothers_occ": 121,
    "fathers_occ": 121,
    "admission_grade": 160.0,
    "displaced": 0,
    "international": 0,
    "gender": 0,
    "age": 20,
    "debtor": 0,
    "tuition_updated": 1,
    "scholarship": 1,
    "special_needs": 0,
    "sem1_enrolled": 6,
    "sem1_credited": 6,
    "sem1_evaluations": 6,
    "sem1_approved": 6,
    "sem1_grade": 16.5,
    "sem1_without_eval": 0,
    "sem2_enrolled": 6,
    "sem2_credited": 6,
    "sem2_evaluations": 6,
    "sem2_approved": 6,
    "sem2_grade": 17.2,
    "sem2_without_eval": 0,
    "unemployment_rate": 8.5,
    "inflation_rate": 1.2,
    "gdp": 2.2,
}

ALL_SCENARIOS = [SCENARIO_1, SCENARIO_2, SCENARIO_3, SCENARIO_4]

DROPOUT_THRESHOLD = 0.5

# ============================================================================
# MODEL LOADING & CACHING
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """
    Load and cache ML model artifacts from disk.
    
    Returns:
        tuple: (model, scaler, encoders, feature_names) or (None, None, None, None) on error
    """
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        return model, scaler, encoders, feature_names
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {e}")
        return None, None, None, None

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_prediction_features(data_dict):
    """
    Engineer features from raw student data for model prediction.
    
    Computes aggregated features including:
    - Total units across semesters
    - Average grades and trend
    - Approval rates
    - Academic consistency
    
    Args:
        data_dict (dict): Raw student input data
        
    Returns:
        dict: Enhanced data dictionary with engineered features
    """
    try:
        # Aggregate semester metrics
        data_dict['Total_units_credited'] = (
            data_dict['Curricular_units_1st_sem_credited'] + 
            data_dict['Curricular_units_2nd_sem_credited']
        )
        data_dict['Total_units_enrolled'] = (
            data_dict['Curricular_units_1st_sem_enrolled'] + 
            data_dict['Curricular_units_2nd_sem_enrolled']
        )
        data_dict['Total_units_evaluated'] = (
            data_dict['Curricular_units_1st_sem_evaluations'] + 
            data_dict['Curricular_units_2nd_sem_evaluations']
        )
        data_dict['Total_units_approved'] = (
            data_dict['Curricular_units_1st_sem_approved'] + 
            data_dict['Curricular_units_2nd_sem_approved']
        )
        data_dict['Total_units_without_eval'] = (
            data_dict['Curricular_units_1st_sem_without_evaluations'] + 
            data_dict['Curricular_units_2nd_sem_without_evaluations']
        )
        
        # Grade metrics
        sem1_grade = data_dict['Curricular_units_1st_sem_grade']
        sem2_grade = data_dict['Curricular_units_2nd_sem_grade']
        
        data_dict['Avg_grade'] = (sem1_grade + sem2_grade) / 2
        data_dict['Grade_trend'] = sem2_grade - sem1_grade
        
        # Approval rates (division by zero protection)
        sem1_enrolled = max(data_dict['Curricular_units_1st_sem_enrolled'], 1)
        sem1_approved = data_dict['Curricular_units_1st_sem_approved']
        data_dict['Sem1_approval_rate'] = sem1_approved / sem1_enrolled
        
        sem2_enrolled = max(data_dict['Curricular_units_2nd_sem_enrolled'], 1)
        sem2_approved = data_dict['Curricular_units_2nd_sem_approved']
        data_dict['Sem2_approval_rate'] = sem2_approved / sem2_enrolled
        
        total_enrolled = max(data_dict['Total_units_enrolled'], 1)
        data_dict['Overall_approval_rate'] = data_dict['Total_units_approved'] / total_enrolled
        
        # Academic consistency
        data_dict['Academic_consistency'] = (
            1 - abs(data_dict['Sem1_approval_rate'] - data_dict['Sem2_approval_rate'])
        )
        
        return data_dict
        
    except KeyError as e:
        st.error(f"❌ Missing required field in student data: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error in feature engineering: {e}")
        return None

# ============================================================================
# PREDICTION & OUTPUT FUNCTIONS
# ============================================================================

def make_prediction(model, scaler, encoders, feature_names, input_data):
    """
    Generate dropout prediction for student input data.
    
    Args:
        model: Trained XGBoost classifier
        scaler: StandardScaler for feature normalization
        encoders: Label encoders for categorical variables
        feature_names: List of feature names in training order
        input_data: Raw student input dictionary
        
    Returns:
        tuple: (prediction, probability) or (None, None) on error
    """
    try:
        # Create features
        input_with_features = create_prediction_features(input_data.copy())
        if input_with_features is None:
            return None, None
        
        # Create DataFrame and filter features
        input_df = pd.DataFrame([input_with_features])
        input_df_filtered = input_df[feature_names]
        
        # Encode categorical variables
        input_df_encoded = input_df_filtered.copy()
        for col in encoders.keys():
            if col in input_df_encoded.columns:
                try:
                    input_df_encoded[col] = encoders[col].transform(input_df_encoded[col].astype(str))
                except Exception:
                    pass
        
        # Scale and predict
        scaled_input = scaler.transform(input_df_encoded)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0]
        
        return prediction, probability
        
    except ValueError as e:
        st.error(f"❌ Feature value error: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None


def display_prediction_results(prediction, probability):
    """
    Display prediction results with visualizations and recommendations.
    
    Args:
        prediction: Binary prediction (0 or 1)
        probability: Probability array [not_dropout_prob, dropout_prob]
    """
    st.divider()
    st.subheader("📊 Prediction Results")
    
    # Status display
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error(f"⚠️ **HIGH RISK** - Dropout Probability: **{probability[1]*100:.1f}%**")
        else:
            st.success(f"✅ **LOW RISK** - Safe Probability: **{probability[0]*100:.1f}%**")
    
    # Probability gauge
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability[1]*100,
            title={'text': "Dropout Risk %"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if probability[1] > DROPOUT_THRESHOLD else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics
    st.write("**Probability Breakdown:**")
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Dropout Risk", f"{probability[1]*100:.2f}%")
    metric_col2.metric("Success Probability", f"{probability[0]*100:.2f}%")
    
    # Recommendations
    st.divider()
    st.subheader("💡 Recommended Actions")
    
    if prediction == 1:
        st.warning("**This student is at high risk of dropout.**")
        with st.expander("📋 View Intervention Recommendations", expanded=True):
            st.markdown("""
            **Immediate Actions:**
            - 📞 Schedule 1:1 academic advising session
            - 💰 Assess financial support options and scholarships
            - 📚 Arrange peer tutoring or study group enrollment
            - ⏰ Explore flexible course scheduling options
            - 👥 Connect with student support services
            
            **Follow-up:**
            - Regular bi-weekly check-ins with academic staff
            - Monitor assignment submissions and grades
            - Provide mental health resources if needed
            - Consider course load adjustment
            """)
    else:
        st.success("**This student is predicted to succeed!**")
        with st.expander("📋 View Support Recommendations", expanded=True):
            st.markdown("""
            **Continued Support:**
            - ✨ Maintain regular academic engagement
            - 🎯 Encourage participation in extracurricular activities
            - 🚀 Explore career development opportunities
            - 👥 Foster strong study group connections
            - 🏆 Provide recognition and positive reinforcement
            """)


def load_scenario(scenario_dict):
    """Load a scenario's values into session state."""
    st.session_state.update(scenario_dict)


def get_index(options_list, value, default=0):
    """Safely get index of value in options list, return default if not found."""
    try:
        return list(options_list).index(value)
    except ValueError:
        return default


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function - Student Dropout Prediction System"""
    
    # Initialize session state with default values if not already set
    if "marital_status" not in st.session_state:
        st.session_state.marital_status = 1
    if "gender" not in st.session_state:
        st.session_state.gender = 0
    if "age" not in st.session_state:
        st.session_state.age = 20
    if "nacionality" not in st.session_state:
        st.session_state.nacionality = 1
    if "displaced" not in st.session_state:
        st.session_state.displaced = 0
    if "application_mode" not in st.session_state:
        st.session_state.application_mode = 1
    if "application_order" not in st.session_state:
        st.session_state.application_order = 0
    if "course" not in st.session_state:
        st.session_state.course = 9500
    if "daytime" not in st.session_state:
        st.session_state.daytime = 1
    if "previous_qual" not in st.session_state:
        st.session_state.previous_qual = 1
    if "previous_qual_grade" not in st.session_state:
        st.session_state.previous_qual_grade = 155.0
    if "mothers_qual" not in st.session_state:
        st.session_state.mothers_qual = 4
    if "fathers_qual" not in st.session_state:
        st.session_state.fathers_qual = 4
    if "mothers_occ" not in st.session_state:
        st.session_state.mothers_occ = 121
    if "fathers_occ" not in st.session_state:
        st.session_state.fathers_occ = 121
    if "admission_grade" not in st.session_state:
        st.session_state.admission_grade = 160.0
    if "international" not in st.session_state:
        st.session_state.international = 0
    if "debtor" not in st.session_state:
        st.session_state.debtor = 0
    if "tuition_updated" not in st.session_state:
        st.session_state.tuition_updated = 1
    if "scholarship" not in st.session_state:
        st.session_state.scholarship = 1
    if "special_needs" not in st.session_state:
        st.session_state.special_needs = 0
    if "sem1_enrolled" not in st.session_state:
        st.session_state.sem1_enrolled = 6
    if "sem1_credited" not in st.session_state:
        st.session_state.sem1_credited = 6
    if "sem1_evaluations" not in st.session_state:
        st.session_state.sem1_evaluations = 6
    if "sem1_approved" not in st.session_state:
        st.session_state.sem1_approved = 6
    if "sem1_grade" not in st.session_state:
        st.session_state.sem1_grade = 16.5
    if "sem1_without_eval" not in st.session_state:
        st.session_state.sem1_without_eval = 0
    if "sem2_enrolled" not in st.session_state:
        st.session_state.sem2_enrolled = 6
    if "sem2_credited" not in st.session_state:
        st.session_state.sem2_credited = 6
    if "sem2_evaluations" not in st.session_state:
        st.session_state.sem2_evaluations = 6
    if "sem2_approved" not in st.session_state:
        st.session_state.sem2_approved = 6
    if "sem2_grade" not in st.session_state:
        st.session_state.sem2_grade = 17.2
    if "sem2_without_eval" not in st.session_state:
        st.session_state.sem2_without_eval = 0
    if "unemployment_rate" not in st.session_state:
        st.session_state.unemployment_rate = 8.5
    if "inflation_rate" not in st.session_state:
        st.session_state.inflation_rate = 1.2
    if "gdp" not in st.session_state:
        st.session_state.gdp = 2.2
    
    # Header Section
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #1f77b4;'>🎓 Student Dropout Prediction System</h1>
        <p style='color: #666; font-size: 16px;'><b>Jaya Jaya Institut</b> - Early Warning System for Student Retention</p>
        <hr style='margin: 10px 0;'>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, encoders, feature_names = load_model_artifacts()
    if model is None:
        st.error("⚠️ Unable to load model. Please contact support.")
        return
    
    st.info("""
    📝 **How to use:** Fill in the student information below and click 'Predict Dropout Risk' 
    to receive a prediction and personalized recommendations.
    """)
    
    # SCENARIO SELECTOR
    st.markdown("### 🔄 Quick Scenario Selector")
    cols = st.columns(4)
    with cols[0]:
        if st.button("🔴 Scenario 1\n(High Risk)", use_container_width=True):
            load_scenario(SCENARIO_1)
            st.rerun()
    with cols[1]:
        if st.button("🔴🔴 Scenario 2\n(Very High Risk)", use_container_width=True):
            load_scenario(SCENARIO_2)
            st.rerun()
    with cols[2]:
        if st.button("🟢 Scenario 3\n(Low Risk)", use_container_width=True):
            load_scenario(SCENARIO_3)
            st.rerun()
    with cols[3]:
        if st.button("🟢🟢 Scenario 4\n(Very Low Risk)", use_container_width=True):
            load_scenario(SCENARIO_4)
            st.rerun()
    
    st.divider()
    
    # INPUT FORM
    with st.form("student_prediction_form", clear_on_submit=False):
        
        # ====== DEMOGRAPHIC INFORMATION ======
        st.markdown("### 📋 Demographic Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            marital_status = st.selectbox(
                "Marital Status",
                options=list(MARITAL_STATUS_MAP.keys()),
                format_func=lambda x: MARITAL_STATUS_MAP[x],
                help="Student's marital status",
                index=get_index(MARITAL_STATUS_MAP.keys(), st.session_state.marital_status)
            )
            st.session_state.marital_status = marital_status
            
            gender = st.selectbox(
                "Gender",
                options=[0, 1],
                format_func=lambda x: GENDER_MAP[x],
                help="Student's gender",
                index=st.session_state.gender
            )
            st.session_state.gender = gender
            
            age = st.slider(
                "Age at Enrollment (years)",
                min_value=18,
                max_value=80,
                value=st.session_state.age,
                step=1,
                help="Age when student enrolled"
            )
            st.session_state.age = age
        
        with col2:
            nacionality = st.selectbox(
                "Nationality",
                options=list(NATIONALITY_MAP.keys()),
                format_func=lambda x: NATIONALITY_MAP[x],
                help="Student's country of origin",
                index=get_index(NATIONALITY_MAP.keys(), st.session_state.nacionality)
            )
            st.session_state.nacionality = nacionality
            
            displaced = st.selectbox(
                "Displaced Student",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Whether student is displaced",
                index=st.session_state.displaced
            )
            st.session_state.displaced = displaced
            
            international = st.selectbox(
                "International Student",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Whether student is from another country",
                index=st.session_state.international
            )
            st.session_state.international = international
        
        with col3:
            previous_qual = st.selectbox(
                "Previous Qualification",
                options=list(PREVIOUS_QUAL_MAP.keys()),
                format_func=lambda x: PREVIOUS_QUAL_MAP[x],
                help="Type and level of previous qualification",
                index=get_index(PREVIOUS_QUAL_MAP.keys(), st.session_state.previous_qual)
            )
            st.session_state.previous_qual = previous_qual
            
            previous_qual_grade = st.slider(
                "Previous Qualification Grade",
                min_value=0.0,
                max_value=200.0,
                value=st.session_state.previous_qual_grade,
                step=1.0,
                help="Grade from previous qualification (0-200)"
            )
            st.session_state.previous_qual_grade = previous_qual_grade
            
            admission_grade = st.slider(
                "Admission Grade",
                min_value=0.0,
                max_value=200.0,
                value=st.session_state.admission_grade,
                step=1.0,
                help="Grade used for admission (0-200)"
            )
            st.session_state.admission_grade = admission_grade
        
        # ====== ACADEMIC INFORMATION ======
        st.markdown("### 📚 Academic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            course = st.selectbox(
                "Course/Program",
                options=list(COURSE_MAP.keys()),
                format_func=lambda x: COURSE_MAP[x],
                help="Academic course or program",
                index=get_index(COURSE_MAP.keys(), st.session_state.course)
            )
            st.session_state.course = course
            
            daytime = st.selectbox(
                "Attendance Type",
                options=[0, 1],
                format_func=lambda x: DAYTIME_MAP[x],
                help="Whether classes are daytime or evening",
                index=st.session_state.daytime
            )
            st.session_state.daytime = daytime
            
            application_mode = st.selectbox(
                "Application Mode",
                options=list(APPLICATION_MODE_MAP.keys()),
                format_func=lambda x: APPLICATION_MODE_MAP[x],
                help="How student applied to the program",
                index=get_index(APPLICATION_MODE_MAP.keys(), st.session_state.application_mode)
            )
            st.session_state.application_mode = application_mode
            
            application_order = st.selectbox(
                "Application Order",
                options=list(APPLICATION_ORDER_MAP.keys()),
                format_func=lambda x: APPLICATION_ORDER_MAP[x],
                help="Priority order of application (0=first choice, 9=last choice)",
                index=get_index(APPLICATION_ORDER_MAP.keys(), st.session_state.application_order)
            )
            st.session_state.application_order = application_order
        
        with col2:
            mothers_qual = st.selectbox(
                "Mother's Qualification",
                options=list(QUALIFICATION_MAP.keys()),
                format_func=lambda x: QUALIFICATION_MAP[x],
                help="Educational level of mother",
                index=get_index(QUALIFICATION_MAP.keys(), st.session_state.mothers_qual)
            )
            st.session_state.mothers_qual = mothers_qual
            
            fathers_qual = st.selectbox(
                "Father's Qualification",
                options=list(QUALIFICATION_MAP.keys()),
                format_func=lambda x: QUALIFICATION_MAP[x],
                help="Educational level of father",
                index=get_index(QUALIFICATION_MAP.keys(), st.session_state.fathers_qual)
            )
            st.session_state.fathers_qual = fathers_qual
            
            mothers_occ = st.selectbox(
                "Mother's Occupation",
                options=list(OCCUPATION_MAP.keys()),
                format_func=lambda x: OCCUPATION_MAP[x],
                help="Mother's occupation classification",
                index=get_index(OCCUPATION_MAP.keys(), st.session_state.mothers_occ)
            )
            st.session_state.mothers_occ = mothers_occ
            
            fathers_occ = st.selectbox(
                "Father's Occupation",
                options=list(OCCUPATION_MAP.keys()),
                format_func=lambda x: OCCUPATION_MAP[x],
                help="Father's occupation classification",
                index=get_index(OCCUPATION_MAP.keys(), st.session_state.fathers_occ)
            )
            st.session_state.fathers_occ = fathers_occ
        
        # ====== 1ST SEMESTER PERFORMANCE ======
        st.markdown("### 📊 1st Semester Academic Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Units**")
            sem1_enrolled = st.slider(
                "Sem 1 - Units Enrolled",
                min_value=0,
                max_value=20,
                value=st.session_state.sem1_enrolled,
                step=1,
                help="Number of units enrolled in first semester"
            )
            st.session_state.sem1_enrolled = sem1_enrolled
            
            sem1_credited = st.slider(
                "Sem 1 - Units Credited",
                min_value=0,
                max_value=20,
                value=st.session_state.sem1_credited,
                step=1,
                help="Number of units credited in first semester"
            )
            st.session_state.sem1_credited = sem1_credited
            
            sem1_evaluations = st.slider(
                "Sem 1 - Evaluations Completed",
                min_value=0,
                max_value=33,
                value=st.session_state.sem1_evaluations,
                step=1,
                help="Number of evaluations completed in first semester"
            )
            st.session_state.sem1_evaluations = sem1_evaluations
        
        with col2:
            st.write("**Performance**")
            sem1_approved = st.slider(
                "Sem 1 - Units Approved",
                min_value=0,
                max_value=20,
                value=st.session_state.sem1_approved,
                step=1,
                help="Number of units approved in first semester"
            )
            st.session_state.sem1_approved = sem1_approved
            
            sem1_grade = st.slider(
                "Sem 1 - Average Grade",
                min_value=0.0,
                max_value=20.0,
                value=st.session_state.sem1_grade,
                step=0.5,
                help="Average grade in first semester (0-20)"
            )
            st.session_state.sem1_grade = sem1_grade
            
            sem1_without_eval = st.slider(
                "Sem 1 - Units Without Evaluations",
                min_value=0,
                max_value=12,
                value=st.session_state.sem1_without_eval,
                step=1,
                help="Number of units without evaluations in first semester"
            )
            st.session_state.sem1_without_eval = sem1_without_eval
        
        # ====== 2ND SEMESTER PERFORMANCE ======
        st.markdown("### 📊 2nd Semester Academic Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Units**")
            sem2_enrolled = st.slider(
                "Sem 2 - Units Enrolled",
                min_value=0,
                max_value=23,
                value=st.session_state.sem2_enrolled,
                step=1,
                help="Number of units enrolled in second semester"
            )
            st.session_state.sem2_enrolled = sem2_enrolled
            
            sem2_credited = st.slider(
                "Sem 2 - Units Credited",
                min_value=0,
                max_value=20,
                value=st.session_state.sem2_credited,
                step=1,
                help="Number of units credited in second semester"
            )
            st.session_state.sem2_credited = sem2_credited
            
            sem2_evaluations = st.slider(
                "Sem 2 - Evaluations Completed",
                min_value=0,
                max_value=33,
                value=st.session_state.sem2_evaluations,
                step=1,
                help="Number of evaluations completed in second semester"
            )
            st.session_state.sem2_evaluations = sem2_evaluations
        
        with col2:
            st.write("**Performance**")
            sem2_approved = st.slider(
                "Sem 2 - Units Approved",
                min_value=0,
                max_value=20,
                value=st.session_state.sem2_approved,
                step=1,
                help="Number of units approved in second semester"
            )
            st.session_state.sem2_approved = sem2_approved
            
            sem2_grade = st.slider(
                "Sem 2 - Average Grade",
                min_value=0.0,
                max_value=20.0,
                value=st.session_state.sem2_grade,
                step=0.5,
                help="Average grade in second semester (0-20)"
            )
            st.session_state.sem2_grade = sem2_grade
            
            sem2_without_eval = st.slider(
                "Sem 2 - Units Without Evaluations",
                min_value=0,
                max_value=12,
                value=st.session_state.sem2_without_eval,
                step=1,
                help="Number of units without evaluations in second semester"
            )
            st.session_state.sem2_without_eval = sem2_without_eval
        
        # ====== SOCIOECONOMIC FACTORS ======
        st.markdown("### 💰 Socioeconomic Factors")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Financial Status**")
            debtor = st.selectbox(
                "Student is Debtor",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                index=st.session_state.debtor,
                help="Whether student owes money"
            )
            st.session_state.debtor = debtor
            
            tuition_updated = st.selectbox(
                "Tuition Fees Up to Date",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                index=st.session_state.tuition_updated,
                help="Whether tuition fees are paid"
            )
            st.session_state.tuition_updated = tuition_updated
            
            scholarship = st.selectbox(
                "Scholarship Holder",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                index=st.session_state.scholarship,
                help="Whether student has scholarship support"
            )
            st.session_state.scholarship = scholarship
        
        with col2:
            st.write("**Support & Needs**")
            special_needs = st.selectbox(
                "Educational Special Needs",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                index=st.session_state.special_needs,
                help="Whether student has special educational needs"
            )
            st.session_state.special_needs = special_needs
            
            unemployment_rate = st.slider(
                "Unemployment Rate (%)",
                min_value=7.6,
                max_value=16.2,
                value=st.session_state.unemployment_rate,
                step=0.1,
                help="National unemployment rate (%)"
            )
            st.session_state.unemployment_rate = unemployment_rate
            
            inflation_rate = st.slider(
                "Inflation Rate (%)",
                min_value=-0.8,
                max_value=3.7,
                value=st.session_state.inflation_rate,
                step=0.1,
                help="National inflation rate (%)"
            )
            st.session_state.inflation_rate = inflation_rate
        
        with col3:
            st.write("**Economic Indicators**")
            gdp = st.slider(
                "GDP Growth (%)",
                min_value=-4.06,
                max_value=3.51,
                value=st.session_state.gdp,
                step=0.1,
                help="GDP growth (%)"
            )
            st.session_state.gdp = gdp
        
        # SUBMIT BUTTON
        st.divider()
        submit_btn = st.form_submit_button(
            "🎯 Predict Dropout Risk",
            use_container_width=True,
            type="primary"
        )
    
    # PREDICTION & RESULTS
    if submit_btn:
        with st.spinner("⏳ Analyzing student data and generating prediction..."):
            
            # Prepare input data
            input_data = {
                'Marital_status': marital_status,
                'Application_mode': application_mode,
                'Application_order': application_order,
                'Course': course,
                'Daytime_evening_attendance': daytime,
                'Previous_qualification': previous_qual,
                'Previous_qualification_grade': previous_qual_grade,
                'Nacionality': nacionality,
                'Mothers_qualification': mothers_qual,
                'Fathers_qualification': fathers_qual,
                'Mothers_occupation': mothers_occ,
                'Fathers_occupation': fathers_occ,
                'Admission_grade': admission_grade,
                'Displaced': displaced,
                'Educational_special_needs': special_needs,
                'Debtor': debtor,
                'Tuition_fees_up_to_date': tuition_updated,
                'Gender': gender,
                'Scholarship_holder': scholarship,
                'Age_at_enrollment': age,
                'International': international,
                'Curricular_units_1st_sem_credited': sem1_credited,
                'Curricular_units_1st_sem_enrolled': sem1_enrolled,
                'Curricular_units_1st_sem_evaluations': sem1_evaluations,
                'Curricular_units_1st_sem_approved': sem1_approved,
                'Curricular_units_1st_sem_grade': sem1_grade,
                'Curricular_units_1st_sem_without_evaluations': sem1_without_eval,
                'Curricular_units_2nd_sem_credited': sem2_credited,
                'Curricular_units_2nd_sem_enrolled': sem2_enrolled,
                'Curricular_units_2nd_sem_evaluations': sem2_evaluations,
                'Curricular_units_2nd_sem_approved': sem2_approved,
                'Curricular_units_2nd_sem_grade': sem2_grade,
                'Curricular_units_2nd_sem_without_evaluations': sem2_without_eval,
                'Unemployment_rate': unemployment_rate,
                'Inflation_rate': inflation_rate,
                'GDP': gdp,
            }
            
            # Make prediction
            prediction, probability = make_prediction(
                model, scaler, encoders, feature_names, input_data
            )
            
            # Display results
            if prediction is not None:
                display_prediction_results(prediction, probability)
            else:
                st.error("❌ Prediction failed. Please check your input and try again.")
    
    # FOOTER
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px; padding: 20px;'>
        <hr style='margin: 10px 0;'>
        <p><b>Student Dropout Prediction System v1.0</b></p>
        <p>Powered by Gradient Boosting Machine Learning • Accuracy: 93.66% • F1-Score: 91.93%</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
