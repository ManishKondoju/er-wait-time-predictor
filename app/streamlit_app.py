import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, time
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure page
st.set_page_config(
    page_title="ER Wait Time Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {padding-top: 2rem;}
    .stButton>button {width: 100%; background-color: #FF4B4B; color: white;}
    .stMetric {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and preprocessors"""
    try:
        model = joblib.load('models/model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
        return model, scaler, feature_cols
    except FileNotFoundError:
        st.error("⚠️ Model not found! Please run 'python train.py' first to train the model.")
        st.stop()

def prepare_input_data(complaint, age, triage_level, arrival_mode, occupancy, timestamp, feature_cols):
    """Prepare input data for prediction"""
    
    # Create base dictionary
    input_dict = {
        'hour': timestamp.hour,
        'day_of_week': timestamp.weekday(),
        'month': timestamp.month,
        'age': age,
        'triage_level': triage_level,
        'current_occupancy': occupancy,
        'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
        'is_night': 1 if timestamp.hour < 6 or timestamp.hour >= 22 else 0,
        'is_evening_rush': 1 if 18 <= timestamp.hour <= 21 else 0,
        'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
        'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
        'dow_sin': np.sin(2 * np.pi * timestamp.weekday() / 7),
        'dow_cos': np.cos(2 * np.pi * timestamp.weekday() / 7)
    }
    
    # Create DataFrame
    df = pd.DataFrame([input_dict])
    
    # Add one-hot encoded columns
    for col in feature_cols:
        if col not in df.columns:
            if col == f'complaint_{complaint}':
                df[col] = 1
            elif col == f'arrival_{arrival_mode}':
                df[col] = 1
            else:
                df[col] = 0
    
    # Ensure correct column order
    df = df[feature_cols]
    
    return df

def main():
    # Title and description
    st.title("🏥 Emergency Room Wait Time Predictor")
    st.markdown("### Get an instant estimate of ER wait times based on current conditions")
    
    # Load model
    model, scaler, feature_cols = load_model()
    
    # Create layout
    col1, col2 = st.columns([1, 2])
    
    # Input section
    with col1:
        st.markdown("### 📋 Patient Information")
        
        with st.form("prediction_form"):
            complaint = st.selectbox(
                "Chief Complaint",
                options=['Chest Pain', 'Abdominal Pain', 'Breathing Difficulty', 
                        'Fracture/Injury', 'Fever', 'Headache', 'Minor Cut/Wound', 'Other'],
                help="Select the main reason for your ER visit"
            )
            
            age = st.number_input(
                "Patient Age",
                min_value=1,
                max_value=120,
                value=35,
                help="Enter the patient's age"
            )
            
            triage_level = st.select_slider(
                "Triage Level",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: f"Level {x} - {'Critical' if x==1 else 'Urgent' if x==2 else 'Less Urgent' if x==3 else 'Non-Urgent' if x==4 else 'Routine'}",
                help="1 = Most Urgent, 5 = Least Urgent"
            )
            
            arrival_mode = st.radio(
                "Arrival Mode",
                options=['Walk-in', 'Ambulance'],
                horizontal=True
            )
            
            occupancy = st.slider(
                "Current ER Occupancy (%)",
                min_value=0,
                max_value=100,
                value=70,
                step=5,
                help="Estimated current occupancy of the ER"
            )
            
            st.markdown("#### ⏰ Time Settings")
            
            use_current = st.checkbox("Use Current Time", value=True)
            
            if use_current:
                timestamp = datetime.now()
                st.info(f"Current time: {timestamp.strftime('%I:%M %p, %A, %B %d')}")
            else:
                col_date, col_time = st.columns(2)
                with col_date:
                    selected_date = st.date_input("Date", value=datetime.now().date())
                with col_time:
                    selected_time = st.time_input("Time", value=datetime.now().time())
                timestamp = datetime.combine(selected_date, selected_time)
            
            submitted = st.form_submit_button("🔮 Predict Wait Time", type="primary")
    
    # Results section
    with col2:
        if submitted:
            # Prepare input
            input_df = prepare_input_data(
                complaint, age, triage_level, arrival_mode,
                occupancy, timestamp, feature_cols
            )
            
            # Make prediction
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            # Display results
            st.markdown("### 📊 Prediction Results")
            
            # Metrics row
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="Estimated Wait Time",
                    value=f"{int(prediction)} minutes",
                    delta=None
                )
            
            with metric_col2:
                hours = prediction / 60
                st.metric(
                    label="In Hours",
                    value=f"{hours:.1f} hours",
                    delta=None
                )
            
            with metric_col3:
                if prediction < 60:
                    status = "🟢 Short Wait"
                elif prediction < 120:
                    status = "🟡 Moderate Wait"
                else:
                    status = "🔴 Long Wait"
                st.metric(
                    label="Status",
                    value=status,
                    delta=None
                )
            
            # Gauge chart
            st.markdown("### Wait Time Visualization")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                title={'text': "Predicted Wait Time (minutes)"},
                delta={'reference': 90, 'relative': True},
                gauge={
                    'axis': {'range': [None, 300], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgreen"},
                        {'range': [60, 120], 'color': "yellow"},
                        {'range': [120, 180], 'color': "orange"},
                        {'range': [180, 300], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 240
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            if prediction < 60:
                st.success("""
                ✅ **Short wait expected!** This is a good time to visit the ER.
                - Expect to be seen within an hour
                - Bring any relevant medical records
                - Have your insurance information ready
                """)
            elif prediction < 120:
                st.warning("""
                ⚠️ **Moderate wait expected.** Consider these options:
                - Call ahead to confirm current wait times
                - Consider urgent care for non-emergency issues
                - Bring something to keep yourself occupied
                """)
            else:
                st.error("""
                🚨 **Long wait expected.** Please consider:
                - For non-emergencies, try urgent care or schedule with your primary care
                - If you must visit ER, bring essentials for a long wait
                - Consider visiting during off-peak hours (early morning)
                """)
        
        else:
            # Show general information when no prediction
            st.markdown("### 📈 Typical ER Wait Time Patterns")
            
            # Create sample data for visualization
            hours = list(range(24))
            typical_wait = [
                45, 40, 35, 30, 35, 40,      # 12 AM - 6 AM (night)
                55, 70, 85, 95, 100, 105,    # 6 AM - 12 PM (morning)
                110, 105, 100, 95, 100, 110, # 12 PM - 6 PM (afternoon)
                125, 135, 130, 115, 85, 65   # 6 PM - 12 AM (evening)
            ]
            
            df_viz = pd.DataFrame({
                'Hour': hours,
                'Average Wait (min)': typical_wait
            })
            
            fig = px.area(
                df_viz,
                x='Hour',
                y='Average Wait (min)',
                title='Average ER Wait Times Throughout the Day',
                labels={'Hour': 'Hour of Day (24-hour format)'},
                color_discrete_sequence=['#FF6B6B']
            )
            
            # FIXED: Using update_xaxes instead of update_xaxis
            fig.update_xaxes(
                tickmode='array',
                tickvals=[0, 6, 12, 18, 23],
                ticktext=['12 AM', '6 AM', '12 PM', '6 PM', '11 PM']
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Information boxes
            st.markdown("### ℹ️ How It Works")
            
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.info("""
                **Factors that INCREASE wait time:**
                - Evening hours (6 PM - 11 PM)
                - Weekends
                - High ER occupancy
                - Lower triage priority
                - Flu season (Dec-Feb)
                """)
            
            with info_col2:
                st.info("""
                **Factors that DECREASE wait time:**
                - Early morning (3 AM - 8 AM)
                - Ambulance arrival
                - High triage priority (1-2)
                - Low ER occupancy
                - Weekday visits
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>⚠️ <b>Disclaimer:</b> This is a predictive model for educational purposes only. 
        Actual wait times may vary significantly. In case of a medical emergency, always call 911 immediately.</p>
        <p>Made with ❤️ using Python and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()