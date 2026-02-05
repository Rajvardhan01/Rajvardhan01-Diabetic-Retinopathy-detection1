import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from db_module import Database
from utils import (
    save_uploaded_file,
    preprocess_image,
    plot_prediction_confidence,
    plot_prediction_history,
    format_date,
    get_class_color
)

# Initialize database
db = Database()

# Load model
model_path = 'model/model.h5'
model = tf.keras.models.load_model(model_path)

# Load remedies data
with open('remedies.json', 'r') as file:
    remedies_data = json.load(file)

# Constants
image_height = 150
image_width = 150
class_names = ['Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Session state initialization
if 'user' not in st.session_state:
    st.session_state.user = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'  # Default to login page

def login_form():
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            user = db.authenticate_user(username, password)
            if user:
                st.session_state.user = user
                st.session_state.page = 'home'
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")

def signup_form():
    with st.form("signup_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        full_name = st.text_input("Full Name")
        submit = st.form_submit_button("Sign Up")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match!")
            else:
                try:
                    db.create_user(username, email, password, full_name)
                    st.success("Account created successfully! Please login.")
                    st.session_state.page = 'login'
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error creating account: {str(e)}")

def auth_page():
    st.title("Diabetic Retinopathy Detection")
    st.subheader("Welcome to the DR Detection System")
    
    # Display an image if available
    if os.path.exists("assets/login_image.png"):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("assets/login_image.png", width=200)
        with col2:
            tab1, tab2 = st.tabs(["Login", "Sign Up"])
            with tab1:
                login_form()
            with tab2:
                signup_form()
    else:
        # If image doesn't exist, use full width
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            login_form()
        with tab2:
            signup_form()

def home_page():
    st.title("Diabetic Retinopathy Detection")
    
    # Upload and predict section
    st.header("Upload Retinal Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                # Save and preprocess image
                image_path = save_uploaded_file(uploaded_file)
                img_array = preprocess_image(image_path)
                
                # Make prediction
                prediction = model.predict(img_array)
                predicted_class_index = np.argmax(prediction)
                predicted_class = class_names[predicted_class_index]
                confidence = float(prediction[0][predicted_class_index])
                
                # Save prediction to database
                db.save_prediction(
                    st.session_state.user['id'],
                    image_path,
                    predicted_class,
                    confidence
                )
                
                # Display results
                st.subheader("Prediction Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Predicted Class:** {predicted_class}")
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    st.markdown(f"**Remedy:** {remedies_data.get(predicted_class, 'No remedy available')}")
                
                with col2:
                    fig = plot_prediction_confidence(prediction, class_names)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add color-coded severity indicator
                color = get_class_color(predicted_class)
                st.markdown(
                    f'<div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white;">'
                    f'<h3 style="margin: 0;">Severity: {predicted_class}</h3>'
                    f'</div>',
                    unsafe_allow_html=True
                )

def history_page():
    st.title("Prediction History")
    
    # Get user's predictions
    predictions = db.get_user_predictions(st.session_state.user['id'])
    
    if not predictions:
        st.info("No prediction history available.")
        return
    
    # Display prediction history
    for pred in predictions:
        with st.expander(f"Prediction on {format_date(pred['created_at'])}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(pred['image_path'], width=200)
            
            with col2:
                st.markdown(f"**Class:** {pred['prediction_class']}")
                st.markdown(f"**Confidence:** {pred['confidence']:.2%}")
                st.markdown(f"**Date:** {format_date(pred['created_at'])}")
    
    # Plot prediction history
    st.subheader("Prediction Trend")
    fig = plot_prediction_history(predictions)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

def profile_page():
    st.title("User Profile")
    
    user = db.get_user_by_id(st.session_state.user['id'])
    
    with st.form("profile_form"):
        st.markdown(f"**Username:** {user['username']}")
        email = st.text_input("Email", value=user['email'])
        full_name = st.text_input("Full Name", value=user['full_name'] or "")
        submit = st.form_submit_button("Update Profile")
        
        if submit:
            try:
                db.update_user_profile(user['id'], full_name, email)
                st.success("Profile updated successfully!")
            except Exception as e:
                st.error(f"Error updating profile: {str(e)}")

def main():
    st.set_page_config(
        page_title="Diabetic Retinopathy Detection",
        page_icon="👁️",
        layout="wide"
    )
    
    # Sidebar navigation
    if st.session_state.user:
        with st.sidebar:
            # Show logo if available
            if os.path.exists("assets/logo.png"):
                st.image("assets/logo.png", width=200)
            
            st.title("Navigation")
            
            if st.button("Home"):
                st.session_state.page = 'home'
                st.experimental_rerun()
            
            if st.button("History"):
                st.session_state.page = 'history'
                st.experimental_rerun()
            
            if st.button("Profile"):
                st.session_state.page = 'profile'
                st.experimental_rerun()
            
            if st.button("Logout"):
                st.session_state.user = None
                st.session_state.page = 'login'
                st.experimental_rerun()
    
    # Page routing
    if st.session_state.user is None:
        auth_page()
    else:
        if st.session_state.page == 'home':
            home_page()
        elif st.session_state.page == 'history':
            history_page()
        elif st.session_state.page == 'profile':
            profile_page()

if __name__ == "__main__":
    main()

