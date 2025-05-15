import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import time
import plotly.graph_objects as go
from models import ModelHandler

# Add parent directory to path to import custom modules
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Nuclear Accident Prediction and Classification",
    page_icon="☢️",
    layout="wide"
)

# Initialize session state
if 'model_handler' not in st.session_state:
    st.session_state.model_handler = ModelHandler()
    
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
    
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
    
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

def load_models():
    """Load and initialize models"""
    with st.spinner("Loading models..."):
        success = st.session_state.model_handler.load_models()
        if success:
            st.session_state.models_loaded = True
            st.success("Models loaded successfully!")
        else:
            st.error("Error loading models. Please check the model files.")
            st.session_state.models_loaded = False

def process_uploaded_file(uploaded_file):
    """Process the uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.current_data = df
        st.session_state.current_index = 0
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def create_parameter_plot(df, current_idx, window_size=50):
    """Create a real-time plot of reactor parameters"""
    start_idx = max(0, current_idx - window_size)
    end_idx = current_idx + 1
    
    fig = go.Figure()
    
    # Plot key parameters
    key_params = ['Primary pressure', 'Primary temperature', 'Secondary pressure', 'Secondary temperature']
    for param in key_params:
        if param in df.columns:
            fig.add_trace(go.Scatter(
                x=df['TIME'].iloc[start_idx:end_idx],
                y=df[param].iloc[start_idx:end_idx],
                name=param
            ))
    
    fig.update_layout(
        title="Real-time Reactor Parameters",
        xaxis_title="Time (s)",
        yaxis_title="Parameter Value",
        height=400
    )
    
    return fig

def main():
    st.title("Nuclear Accident Prediction and Classification System")
    
    st.write("""
    This system analyzes nuclear reactor data to:
    1. Predict potential reactor scrams in the next 180 seconds
    2. Classify the type of accident if a scram is predicted
    """)
    
    # Load models button
    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load the models first!")
        if st.button("Load Models"):
            load_models()
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = process_uploaded_file(uploaded_file)
        
        if df is not None:
            st.success("File uploaded successfully!")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Create two columns for the analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Real-time Analysis")
                
                # Only show start button if models are loaded
                if st.session_state.models_loaded:
                    start_analysis = st.button("Start Analysis")
                else:
                    st.error("Please load the models before starting analysis!")
                    start_analysis = False
                
                if start_analysis:
                    # Process data
                    features = st.session_state.model_handler.preprocess_data(df)
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create placeholder for parameter plot
                    plot_placeholder = st.empty()
                    
                    # Create placeholders for predictions
                    scram_prob_text = st.empty()
                    accident_type_text = st.empty()
                    
                    # Analyze data in real-time
                    while st.session_state.current_index < len(df) - st.session_state.model_handler.sequence_length:
                        # Update progress
                        progress = st.session_state.current_index / (len(df) - st.session_state.model_handler.sequence_length)
                        progress_bar.progress(progress)
                        
                        # Get current sequence
                        current_sequence = features[st.session_state.current_index:
                                                 st.session_state.current_index + st.session_state.model_handler.sequence_length]
                        
                        # Make predictions
                        scram_prob = st.session_state.model_handler.predict_scram(current_sequence)
                        
                        # Update parameter plot
                        plot_placeholder.plotly_chart(
                            create_parameter_plot(df, st.session_state.current_index),
                            use_container_width=True
                        )
                        
                        # Update prediction text
                        scram_prob_text.write(f"Probability of Scram: {scram_prob:.2%}")
                        
                        if scram_prob > 0.5:
                            # Classify accident type
                            accident_probs = st.session_state.model_handler.classify_accident(current_sequence)
                            accident_type = np.argmax(accident_probs)
                            accident_type_text.write(f"Predicted Accident Type: {accident_type}")
                            
                            # Stop analysis when scram is detected
                            status_text.warning("⚠️ Scram detected! Analysis stopped.")
                            break
                        
                        # Update status
                        status_text.text(f"Analyzing time step {st.session_state.current_index}")
                        
                        # Increment index
                        st.session_state.current_index += 1
                        time.sleep(0.1)  # Add small delay for visualization
                    
                    if st.session_state.current_index >= len(df) - st.session_state.model_handler.sequence_length:
                        status_text.success("✅ Analysis completed. No scram detected.")
            
            with col2:
                st.subheader("Analysis Settings")
                st.write("Model Configuration:")
                st.write(f"- Sequence Length: {st.session_state.model_handler.sequence_length} time steps")
                st.write("- Time step interval: 10 seconds")
                st.write("- Prediction horizon: 180 seconds")

if __name__ == "__main__":
    main() 