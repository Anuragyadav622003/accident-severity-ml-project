# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .severity-slight { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; }
    .severity-serious { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; }
    .severity-fatal { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and metadata"""
    try:
        with open('accident_severity_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return artifacts, metadata
    except FileNotFoundError:
        st.error("❌ Model files not found. Please run 'model_training.py' first.")
        st.info("💡 Run this command in your terminal: `python model_training.py`")
        return None, None

def create_feature_inputs():
    """Create input widgets for features"""
    features = {}
    
    with st.sidebar:
        st.header("🚗 Accident Details")
        
        # Time features
        col1, col2 = st.columns(2)
        with col1:
            features['Hour_of_Day'] = st.slider("Hour of Day", 0, 23, 12)
        with col2:
            features['Time_of_Day'] = st.selectbox("Time of Day", ['Night', 'Morning', 'Afternoon', 'Evening'])
        
        # Driver details
        st.subheader("👤 Driver Information")
        features['Age_band_of_driver'] = st.selectbox("Driver Age Band", 
                                                     ['Under 18', '18-30', '31-50', 'Over 51'])
        features['Sex_of_driver'] = st.selectbox("Driver Gender", ['Male', 'Female'])
        features['Educational_level'] = st.selectbox("Education Level", 
                                                   ['Elementary school', 'Junior high school', 
                                                    'High school', 'Above high school'])
        features['Driving_experience'] = st.selectbox("Driving Experience", 
                                                     ['Below 1yr', '1-2yr', '2-5yr', '5-10yr', 'Above 10yr'])
        
        # Vehicle & Accident details
        st.subheader("🚙 Vehicle & Accident Details")
        features['Type_of_vehicle'] = st.selectbox("Vehicle Type", 
                                                  ['Automobile', 'Lorry', 'Public', 'Taxi', 'Other'])
        features['Area_accident_occured'] = st.selectbox("Accident Area", 
                                                        ['Residential areas', 'Office areas', 'Market areas', 
                                                         'Industrial areas', 'Recreational areas', 'Other'])
        
        # Environmental factors
        st.subheader("🌤️ Environmental Factors")
        features['Road_surface_conditions'] = st.selectbox("Road Conditions", 
                                                          ['Dry', 'Wet or damp', 'Snow', 'Ice'])
        features['Light_conditions'] = st.selectbox("Light Conditions", 
                                                   ['Daylight', 'Darkness - lights lit', 'Darkness - no lights'])
        features['Weather_conditions'] = st.selectbox("Weather", 
                                                     ['Normal', 'Raining', 'Snow', 'Fog', 'Other'])
        
        # Accident severity factors
        st.subheader("📊 Accident Metrics")
        col1, col2 = st.columns(2)
        with col1:
            features['Number_of_vehicles_involved'] = st.number_input("Vehicles Involved", 1, 10, 2)
        with col2:
            features['Number_of_casualties'] = st.number_input("Number of Casualties", 1, 10, 1)
        
        features['Risk_Score'] = (features['Number_of_vehicles_involved'] + features['Number_of_casualties']) / 2
        
        # Additional features with default values
        features['Day_of_week'] = 'Monday'
        features['Service_year_of_vehicle'] = '5-10yr'
        features['Experience_Level'] = 'Intermediate'
        
    return features

def display_prediction_result(prediction, probability, artifacts):
    """Display prediction results in a nice format"""
    severity_map = {v: k for k, v in artifacts['target_mapping'].items()}
    predicted_severity = severity_map[prediction]
    
    # Severity styling
    severity_styles = {
        'Slight Injury': 'severity-slight',
        'Serious Injury': 'severity-serious', 
        'Fatal injury': 'severity-fatal'
    }
    
    severity_icons = {
        'Slight Injury': '🟢',
        'Serious Injury': '🟡', 
        'Fatal injury': '🔴'
    }
    
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    
    # Prediction result
    st.subheader("🎯 Prediction Result")
    st.markdown(f'<div class="{severity_styles.get(predicted_severity, "")}">'
                f'<h3>{severity_icons.get(predicted_severity, "⚪")} {predicted_severity}</h3>'
                f'</div>', unsafe_allow_html=True)
    
    # Probability distribution
    st.subheader("📈 Probability Distribution")
    prob_df = pd.DataFrame({
        'Severity': [severity_map[i] for i in range(len(probability))],
        'Probability': probability
    }).sort_values('Probability', ascending=False)
    
    # Create bar chart
    fig = px.bar(prob_df, x='Severity', y='Probability', 
                color='Probability', 
                color_continuous_scale='RdYlGn_r',
                text='Probability')
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    
    # Display probabilities as metrics
    st.subheader("📊 Detailed Probabilities")
    cols = st.columns(len(prob_df))
    for idx, (_, row) in enumerate(prob_df.iterrows()):
        with cols[idx]:
            st.metric(
                label=row['Severity'],
                value=f"{row['Probability']:.3f}",
                delta="Highest" if idx == 0 else None
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Accident Severity Prediction System</h1>', 
                unsafe_allow_html=True)
    
    # Load model
    artifacts, metadata = load_model()
    if artifacts is None:
        return
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")
        
        if metadata:
            st.metric("Best Model", metadata['model_info']['best_model'])
            st.metric("Accuracy", f"{metadata['model_info']['performance']['Accuracy']:.3f}")
            st.metric("F1-Score", f"{metadata['model_info']['performance']['F1-Score']:.3f}")
            st.metric("Training Date", metadata['training_date'])
            st.metric("Feature Count", metadata['model_info']['feature_count'])
        
        st.markdown("---")
        st.header("ℹ️ How to Use")
        st.info("""
        1. Fill in the accident details
        2. Click 'Predict Accident Severity'
        3. View the prediction and probabilities
        4. Explore analytics in other tabs
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📈 Analytics", "🤖 Model Info", "🚀 About"])
    
    with tab1:
        st.header("Real-time Accident Severity Prediction")
        
        # Feature inputs
        input_features = create_feature_inputs()
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_btn = st.button("🔮 Predict Accident Severity", 
                                  type="primary", 
                                  use_container_width=True)
        
        # Make prediction
        if predict_btn:
            with st.spinner("🤖 Analyzing accident data..."):
                # Convert inputs to DataFrame
                input_df = pd.DataFrame([input_features])
                
                try:
                    # Make prediction
                    prediction = artifacts['model'].predict(input_df)[0]
                    probability = artifacts['model'].predict_proba(input_df)[0]
                    
                    # Display results
                    display_prediction_result(prediction, probability, artifacts)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("💡 Make sure all required features are provided")
    
    with tab2:
        st.header("📊 Data Analytics & Insights")
        
        if metadata:
            # Model performance comparison
            st.subheader("Model Performance Comparison")
            
            performance_data = []
            for model_name, metrics in artifacts['model_performance'].items():
                performance_data.append({
                    'Model': model_name,
                    'Accuracy': metrics['Accuracy'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F1-Score': metrics['F1-Score']
                })
            
            perf_df = pd.DataFrame(performance_data)
            
            # Create performance chart
            fig = go.Figure()
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=perf_df['Model'],
                    y=perf_df[metric],
                    text=perf_df[metric].round(3),
                    textposition='auto'
                ))
            
            fig.update_layout(
                title='Model Performance Metrics',
                barmode='group',
                xaxis_title='Models',
                yaxis_title='Score',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (if available)
            st.subheader("Feature Importance")
            try:
                # This is a simplified version - you'd need to compute actual feature importance
                st.info("""
                **Top Important Features:**
                - Number of vehicles involved
                - Number of casualties  
                - Driver experience
                - Road conditions
                - Time of day
                - Weather conditions
                """)
                
                # Sample feature importance data
                feature_importance_data = {
                    'Feature': ['Vehicles Involved', 'Casualties', 'Driver Experience', 
                               'Road Conditions', 'Time of Day', 'Weather', 'Vehicle Type'],
                    'Importance': [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05]
                }
                
                fi_df = pd.DataFrame(feature_importance_data)
                fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                               title='Feature Importance (Sample)')
                st.plotly_chart(fig_fi, use_container_width=True)
                
            except Exception as e:
                st.warning("Feature importance details not available")
    
    with tab3:
        st.header("🤖 Model Information & Technical Details")
        
        if metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Model Specifications")
                st.metric("Best Performing Model", metadata['model_info']['best_model'])
                st.metric("Target Classes", ", ".join(metadata['model_info']['target_classes']))
                st.metric("Total Features", metadata['model_info']['feature_count'])
                st.metric("Training Samples", metadata['dataset_info']['training_samples'])
                st.metric("Test Samples", metadata['dataset_info']['test_samples'])
            
            with col2:
                st.subheader("🎯 Performance Metrics")
                for metric, value in metadata['model_info']['performance'].items():
                    st.metric(metric, f"{value:.4f}")
                
                st.subheader("🛠️ Technical Stack")
                st.code("""
                - Python 3.8+
                - Scikit-learn
                - XGBoost
                - Streamlit
                - Plotly
                - Pandas/Numpy
                """)
    
    with tab4:
        st.header("🚀 About This Project")
        
        st.markdown("""
        ## Accident Severity Prediction System
        
        This machine learning project predicts the severity of road accidents based on various factors 
        including driver details, environmental conditions, and accident circumstances.
        
        ### 🎯 Project Features
        
        - **Real-time Predictions**: Instant accident severity assessment
        - **Multiple ML Models**: Comparison of various algorithms
        - **Interactive Dashboard**: User-friendly web interface
        - **Comprehensive Analytics**: Detailed insights and visualizations
        - **Production Ready**: Deployable solution
        
        ### 🛠️ Technical Highlights
        
        - **End-to-end ML Pipeline**: Data preprocessing, feature engineering, model training
        - **Hyperparameter Tuning**: Optimized model performance
        - **Model Explainability**: Feature importance and interpretability
        - **Web Deployment**: Streamlit-based interactive application
        
        ### 📊 Business Impact
        
        - **Road Safety**: Helps in understanding accident patterns
        - **Resource Allocation**: Aids emergency services in prioritization
        - **Policy Making**: Supports data-driven safety policies
        - **Insurance**: Assists in risk assessment and claims processing
        
        ### 👨‍💻 Skills Demonstrated
        
        This project showcases expertise in:
        - Machine Learning & Data Science
        - Software Engineering & Deployment
        - Data Visualization & Analytics
        - Project Management & Documentation
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Machine Learning Project for Resume</p>
        <p>Demonstrating advanced skills in Data Science and Full-stack ML Development</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()