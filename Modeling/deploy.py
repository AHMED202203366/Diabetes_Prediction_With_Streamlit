import pickle
import streamlit as st
import pandas as pd
from PIL import Image

# Configure page settings
st.set_page_config(
    page_title="Diabetes Risk Assessment System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load assets
@st.cache_resource
def load_assets():
    try:
        with open("diabetes_model.pkl", 'rb') as f:
            assets = pickle.load(f)
        healthy_img = Image.open(r"Images\No_Dia.png")
        diabetes_img = Image.open(r"Images\Diabetes.png")
        return {**assets, 'healthy_img': healthy_img, 'diabetes_img': diabetes_img}
    except Exception as e:
        st.error(f"Error loading assets: {str(e)}")
        st.stop()

assets = load_assets()

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .header { 
        color: #2c3e50;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .form-container {
        background-color: white;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .stSelectbox, .stNumberInput, .stRadio, .stSlider {
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: white;
        padding: 2rem !important;
    }
    .sidebar-result {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .high-risk { border-left: 4px solid #e74c3c; }
    .low-risk { border-left: 4px solid #2ecc71; }
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e0e0e0;
        color: #7f8c8d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="header"><h1>Diabetes Risk Assessment</h1></div>', unsafe_allow_html=True)
st.markdown("""
<p style="color: #7f8c8d; font-size: 1.1rem;">
A clinical decision support tool for diabetes risk evaluation based on patient health metrics.
</p>
""", unsafe_allow_html=True)

# Initialize session state
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

def prepare_input(age, gender, hypertension, heart_disease, bmi, hba1c, glucose, smoking_history):
    """Convert Streamlit inputs to model-ready format"""
    features = {
        'age': age,
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glucose,
        'gender_Female': 0,
        'gender_Male': 0,
        'gender_Other': 0,
        'smoking_history_current': 0,
        'smoking_history_former': 0,
        'smoking_history_never': 0,
        'smoking_history_not known': 0
    }
    
    # Set gender
    features[f'gender_{gender}'] = 1
    
    # Set smoking history (handle spaces in 'not known')
    smoking_key = smoking_history.replace(" ", "_")
    features[f'smoking_history_{smoking_key}'] = 1
    
    # Return as DataFrame with correct column order
    return pd.DataFrame([features], columns=assets['feature_names'])

# Input form
with st.container():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    with st.form("patient_form"):
        st.markdown("### Patient Demographic Information")
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender*",
                options=["Male", "Female", "Other"],
                help="Patient's self-identified gender"
            )
            age = st.number_input(
                "Age (years)*",
                min_value=0,
                max_value=120,
                value=25,
                step=1,
                help="Patient's age in years"
            )
            
        with col2:
            bmi = st.number_input(
                "BMI (kg/m²)*",
                min_value=10.0,
                max_value=50.0,
                value=22.0,
                step=0.1,
                format="%.1f",
                help="Body Mass Index"
            )
            smoking_history = st.selectbox(
                "Smoking History*",
                options=["never", "current", "former", "not known"],
                help="Patient's smoking status"
            )
        
        st.markdown("### Medical History")
        col3, col4 = st.columns(2)
        with col3:
            hypertension = st.radio(
                "Hypertension*",
                options=["Yes", "No"],
                index=1,
                horizontal=True,
                help="Diagnosed with high blood pressure"
            )
        with col4:
            heart_disease = st.radio(
                "Heart Disease*",
                options=["Yes", "No"],
                index=1,
                horizontal=True,
                help="History of cardiovascular disease"
            )
        
        st.markdown("### Laboratory Results")
        col5, col6 = st.columns(2)
        with col5:
            hba1c = st.slider(
                "HbA1c Level (%)*",
                min_value=3.0,
                max_value=15.0,
                value=5.0,
                step=0.1,
                help="Glycated hemoglobin level"
            )
        with col6:
            glucose = st.slider(
                "Blood Glucose Level (mg/dL)*",
                min_value=50.0,
                max_value=300.0,
                value=90.0,
                step=1.0,
                help="Fasting blood glucose level"
            )
        
        st.markdown("<small>* Required fields</small>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Calculate Diabetes Risk", type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Process form submission
if submitted:
    st.session_state.submitted = True
    try:
        # Prepare input in correct format
        input_df = prepare_input(
            age=age,
            gender=gender,
            hypertension=hypertension,
            heart_disease=heart_disease,
            bmi=bmi,
            hba1c=hba1c,
            glucose=glucose,
            smoking_history=smoking_history
        )
        
        # Scale the input
        scaled_input = assets['scaler'].transform(input_df)
        
        # Make prediction
        prediction = assets['model'].predict(scaled_input)
        probability = assets['model'].predict_proba(scaled_input)[0][1]
        
        # Store results
        st.session_state.prediction = prediction
        st.session_state.probability = probability

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Display results in sidebar if submitted
if st.session_state.get('submitted', False):
    with st.sidebar:
        st.markdown('<div class="header"><h2>Assessment Results</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.prediction[0] == 1:
            with st.container():
                st.markdown('<div class="sidebar-result high-risk">', unsafe_allow_html=True)
                st.error(f"High Risk of Diabetes ({st.session_state.probability:.1%} probability)")
                st.image(assets['diabetes_img'], width=250)
                st.markdown("""
                **Clinical Recommendations:**
                - Immediate referral to endocrinology
                - Fasting plasma glucose test
                - Oral glucose tolerance test
                - Lifestyle intervention program
                - Consider pharmacological therapy
                """)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown('<div class="sidebar-result low-risk">', unsafe_allow_html=True)
                st.success(f"Low Risk of Diabetes ({st.session_state.probability:.1%} probability)")
                st.image(assets['healthy_img'], width=250)
                st.markdown("""
                **Preventive Guidance:**
                - Annual diabetes screening
                - Maintain BMI < 25 kg/m²
                - 150 minutes weekly exercise
                - Mediterranean diet recommended
                - Smoking cessation if applicable
                """)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed analysis
        with st.expander("Detailed Risk Analysis", expanded=True):
            st.markdown(f"""
            **Risk Score Breakdown:**
            - Probability of Diabetes: {st.session_state.probability:.1%}
            - Probability of No Diabetes: {1-st.session_state.probability:.1%}
            """)
            st.progress(st.session_state.probability)

# Footer
st.markdown("""
<div class="footer">
    <p><strong>Disclaimer:</strong> This tool provides risk assessment only and does not constitute medical advice. 
    Clinical correlation and professional judgment are required for diagnosis.</p>
    <p>© 2023 Clinical Decision Support System | v1.0.0</p>
</div>
""", unsafe_allow_html=True)