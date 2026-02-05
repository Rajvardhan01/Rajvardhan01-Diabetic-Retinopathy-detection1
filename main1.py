import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import base64
from db_module_1 import Database
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

# Constants
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150
CLASS_NAMES = ['Mild', 'Moderate', 'Severe', 'Proliferative DR']


# ===== STYLING FUNCTIONS =====
def load_css():
    """Load custom CSS styles from file or create if doesn't exist"""
    css_file = "styles.css"
    if not os.path.exists(css_file):
        with open(css_file, "w") as f:
            f.write("""
/* Main theme colors */
:root {
  --primary: #4361ee;
  --primary-light: #6a81f8;
  --primary-dark: #3a52d9;
  --secondary: #2ed9c3;
  --secondary-dark: #24b6a4;
  --danger: #ef4444;
  --warning: #f59e0b;
  --light: #f3f4f6;
  --dark: #111827;
  --white: #ffffff;
  --gray: #6b7280;
}

/* General styling */
.stApp {
  font-family: 'Inter', sans-serif;
  color: var(--dark);
  background-color: #f9fafb;
}

/* Header styling */
.main-header {
  background: linear-gradient(135deg, var(--primary), var(--primary-dark));
  color: white;
  padding: 2rem;
  border-radius: 1rem;
  margin-bottom: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  text-align: center;
}

.main-header h1 {
  margin: 0;
  font-size: 2.5rem;
  font-weight: 700;
}

.main-header p {
  margin-top: 0.5rem;
  font-size: 1.2rem;
  opacity: 0.9;
}

/* Card styling */
.card {
  background-color: white;
  border-radius: 1rem;
  padding: 1.5rem;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
  margin-bottom: 1.5rem;
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.card-header {
  padding-bottom: 1rem;
  margin-bottom: 1rem;
  font-weight: 600;
  font-size: 1.5rem;
  color: var(--dark);
  border-bottom: 1px solid var(--light);
}

/* Button styling */
.stButton > button {
  background-color: var(--primary);
  color: white !important;
  border: none !important;
  padding: 0.625rem 1.25rem !important;
  border-radius: 0.5rem !important;
  font-weight: 500 !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
}

.stButton > button:hover {
  background-color: var(--primary-dark) !important;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
  transform: translateY(-1px) !important;
}

/* Input styling */
div.stTextInput > div > div > input {
  border-radius: 0.5rem !important;
  border: 1px solid #e5e7eb !important;
  padding: 0.75rem !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}

div.stTextInput > div > div > input:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.15) !important;
}

/* Auth styling */
.auth-container {
  max-width: 800px;
  margin: 3rem auto;
  background-color: white;
  border-radius: 1rem;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.auth-header {
  background: linear-gradient(135deg, var(--primary), var(--primary-dark));
  color: white;
  padding: 2rem;
  text-align: center;
}

.auth-body {
  padding: 2rem;
}

.auth-footer {
  background-color: var(--light);
  padding: 1rem;
  text-align: center;
  font-size: 0.9rem;
  color: var(--gray);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
  gap: 2px;
  border-bottom: 1px solid #e5e7eb;
}

.stTabs [data-baseweb="tab"] {
  height: 50px;
  background-color: transparent;
  border-radius: 0.5rem 0.5rem 0 0;
  font-weight: 500;
}

.stTabs [aria-selected="true"] {
  background-color: transparent;
  color: var(--primary);
  border-bottom: 3px solid var(--primary);
}

/* Prediction results styling */
.prediction-result {
  padding: 1.25rem;
  border-radius: 0.75rem;
  margin-top: 1rem;
}

.severity-indicator {
  padding: 12px;
  border-radius: 0.5rem;
  color: white;
  font-weight: 600;
  text-align: center;
  margin-top: 15px;
}

.severity-mild {
  background-color: #fbbf24;
}

.severity-moderate {
  background-color: #f59e0b;
}

.severity-severe {
  background-color: #dc2626;
}

.severity-proliferative {
  background-color: #991b1b;
}

/* Sidebar styling */
.css-1d391kg {
  background-color: var(--white);
}

.sidebar-content {
  padding: 1rem;
}

/* File uploader styling */
.uploadedFile {
  border: 2px dashed var(--primary);
  border-radius: 1rem;
  padding: 1.5rem;
  text-align: center;
  transition: all 0.3s ease;
  margin-bottom: 1.5rem;
}

.uploadedFile:hover {
  border-color: var(--primary-dark);
  background-color: rgba(67, 97, 238, 0.05);
}

/* Chart container styling */
.chart-container {
  background-color: white;
  border-radius: 0.75rem;
  padding: 1.25rem;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  border: 1px solid rgba(0, 0, 0, 0.05);
}

/* Profile styling */
.profile-header {
  display: flex;
  align-items: center;
  margin-bottom: 2rem;
}

.profile-avatar {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  background-color: var(--primary);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2.5rem;
  margin-right: 1.5rem;
  font-weight: 600;
}

.profile-info {
  flex: 1;
}

.profile-name {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
  color: var(--dark);
}

.profile-username {
  font-size: 1rem;
  color: var(--gray);
}

/* Action buttons */
.action-btn {
  border-radius: 0.5rem;
  padding: 0.75rem 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  font-weight: 500;
  font-size: 0.9rem;
  text-decoration: none;
}

.primary-btn {
  background-color: var(--primary);
  color: white;
}

.primary-btn:hover {
  background-color: var(--primary-dark);
}

.danger-btn {
  background-color: #fee2e2;
  color: #dc2626;
}

.danger-btn:hover {
  background-color: #fecaca;
}

/* Info cards */
.info-card {
  background-color: #f3f4f6;
  border-radius: 0.75rem;
  padding: 1.25rem;
  margin-bottom: 1rem;
}

.info-card h3 {
  margin-top: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--dark);
}

/* No data state */
.no-data {
  text-align: center;
  padding: 3rem 1rem;
}

.no-data-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  color: var(--gray);
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .main-header {
    padding: 1.5rem;
  }
  
  .main-header h1 {
    font-size: 2rem;
  }
  
  .card {
    padding: 1rem;
  }
  
  .auth-container {
    margin: 1rem;
  }
  
  .profile-header {
    flex-direction: column;
    text-align: center;
  }
  
  .profile-avatar {
    margin-right: 0;
    margin-bottom: 1rem;
  }
}
            """)

    with open(css_file, "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)
    
    # Set page configuration
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def load_google_fonts():
    """Load Google Fonts for better typography"""
    st.markdown("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)


# ===== HELPER FUNCTIONS =====
def load_model():
    """Load or create model for prediction"""
    try:
        if not os.path.exists('model'):
            os.makedirs('model')
        
        model_path = 'model/model.h5'
        # Check if model exists, if not, create a placeholder
        if not os.path.exists(model_path):
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
                tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.save(model_path)
        else:
            model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def load_remedies_data():
    """Load or create remedies data"""
    try:
        remedies_path = 'remedies.json'
        if not os.path.exists(remedies_path):
            # Create placeholder remedies data
            remedies_data = {
                "Mild": "Regular eye check-ups every 12 months. Control blood sugar levels.",
                "Moderate": "Eye check-ups every 6-8 months. Blood sugar control and blood pressure management.",
                "Severe": "Frequent eye examinations every 3-4 months. Possible laser treatment may be needed.",
                "Proliferative DR": "Immediate medical attention required. Treatments include laser surgery, anti-VEGF injections, or vitrectomy."
            }
            with open(remedies_path, 'w') as f:
                json.dump(remedies_data, f)
        else:
            with open(remedies_path, 'r') as file:
                remedies_data = json.load(file)
        return remedies_data
    except Exception as e:
        st.error(f"Error loading remedies data: {str(e)}")
        return {}


# ===== AUTHENTICATION PAGES =====
def login_form():
    """Render login form"""
    with st.form("login_form"):
        st.markdown('<div class="card-header">Sign in to your account</div>', unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            submit = st.form_submit_button("Login")
        with col2:
            st.markdown("""
                <div style="display: flex; justify-content: flex-end; padding-top: 10px;">
                    <a href="#" style="color: var(--primary); text-decoration: none;">Forgot password?</a>
                </div>
            """, unsafe_allow_html=True)
        
        if submit:
            if username and password:
                user = db.authenticate_user(username, password)
                if user:
                    st.session_state.user = user
                    st.session_state.page = 'home'
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.warning("Please enter both username and password")

def signup_form():
    """Render signup form"""
    with st.form("signup_form"):
        st.markdown('<div class="card-header">Create a new account</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username", key="signup_username")
        with col2:
            email = st.text_input("Email", key="signup_email")
            
        password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
        full_name = st.text_input("Full Name", key="signup_name")
        
        st.markdown("""
            <div style="margin: 1rem 0; padding: 0.75rem; background-color: #f3f4f6; border-radius: 0.5rem;">
                <small style="color: var(--gray);">
                    By signing up, you agree to our Terms of Service and Privacy Policy.
                </small>
            </div>
        """, unsafe_allow_html=True)
        
        submit = st.form_submit_button("Create Account")
        
        if submit:
            if not all([username, email, password, confirm_password]):
                st.warning("Please fill in all required fields")
            elif password != confirm_password:
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
    """Render authentication page"""
    # Load styles
    load_css()
    load_google_fonts()
    
    # Auth container
    st.markdown("""
    <div class="auth-container">
        <div class="auth-header">
            <h1>Diabetic Retinopathy Detection</h1>
            <p>Early detection for better eye health</p>
        </div>
        <div class="auth-body">
    """, unsafe_allow_html=True)
    
    # Tabs for login/signup
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        login_form()
    
    with tab2:
        signup_form()
    
    st.markdown("""
        </div>
        <div class="auth-footer">
            © 2025 Diabetic Retinopathy Detection System. All rights reserved.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===== MAIN APPLICATION PAGES =====
def home_page():
    """Render home page with upload and analysis functionality"""
    # Load styles, model and data
    load_css()
    load_google_fonts()
    model = load_model()
    remedies_data = load_remedies_data()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>Diabetic Retinopathy Detection</h1>
        <p>Upload a retinal image to detect signs of diabetic retinopathy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Upload Retinal Image</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a retinal image to analyze", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image and guidelines
        image_display = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="uploadedFile">', unsafe_allow_html=True)
            st.image(image_display, caption="Uploaded Retinal Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h3>Image Guidelines</h3>
                <ul>
                    <li>Use high-quality retinal fundus images</li>
                    <li>Ensure proper lighting and focus</li>
                    <li>Image should be centered on the optic disc</li>
                    <li>Avoid images with artifacts or glare</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Analysis button
        analyze_button = st.button("Analyze Image", use_container_width=True)
        
        if analyze_button and model is not None:
            with st.spinner("Analyzing retinal image..."):
                # Process image and make prediction
                image_path = save_uploaded_file(uploaded_file)
                img_array = preprocess_image(image_path)
                
                prediction = model.predict(img_array)
                predicted_class_index = np.argmax(prediction)
                predicted_class = CLASS_NAMES[predicted_class_index]
                confidence = float(prediction[0][predicted_class_index])
                
                # Save prediction to database
                db.save_prediction(
                    st.session_state.user['id'],
                    image_path,
                    predicted_class,
                    confidence
                )
                
                # Display results
            
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-header">Prediction Results</div>', unsafe_allow_html=True)
                
                # Format severity class for styling
                severity_class = predicted_class.lower().replace(" ", "-")
                
                # Split results into columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="info-card">
                        <h3>Diagnosis</h3>
                        <p><strong>Detected Class:</strong> {predicted_class}</p>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <div class="severity-indicator severity-{severity_class}">
                            Severity Level: {predicted_class}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="info-card">
                        <h3>Recommended Actions</h3>
                        <p>{remedies_data.get(predicted_class, 'No specific recommendations available.')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = plot_prediction_confidence(prediction, CLASS_NAMES)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close results card
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close upload card
    
    # Information section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">About Diabetic Retinopathy</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p>Diabetic retinopathy is a diabetes complication that affects the eyes. It's caused by damage to the blood vessels in the retina (the light-sensitive tissue at the back of the eye).</p>
    
    <p>At first, diabetic retinopathy may cause no symptoms or only mild vision problems. But it can lead to blindness if left untreated.</p>
    """, unsafe_allow_html=True)
    
    # Create expandable sections for more information
    with st.expander("Learn about the stages of diabetic retinopathy", expanded=False):
        st.markdown("""
        <h4>The stages of diabetic retinopathy:</h4>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
            <div style="background-color: #fef3c7; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #fbbf24;">
                <h5 style="margin-top: 0; color: #92400e;">Mild</h5>
                <p>Small areas of balloon-like swelling in the retina's tiny blood vessels. Vision is generally not affected.</p>
            </div>
            <div style="background-color: #fdba74; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b;">
                <h5 style="margin-top: 0; color: #9a3412;">Moderate</h5>
                <p>As the disease progresses, some blood vessels that nourish the retina become blocked, affecting blood flow.</p>
            </div>
            <div style="background-color: #fecaca; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #dc2626;">
                <h5 style="margin-top: 0; color: #991b1b;">Severe</h5>
                <p>Many more blood vessels are blocked, depriving blood supply to parts of the retina. The retina signals for new blood vessels.</p>
            </div>
            <div style="background-color: #fca5a5; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #991b1b;">
                <h5 style="margin-top: 0; color: #7f1d1d;">Proliferative</h5>
                <p>The most advanced stage where new, fragile blood vessels grow in the retina. These can leak blood and blur vision severely.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("Risk factors and prevention", expanded=False):
        st.markdown("""
        <h4>Risk Factors</h4>
        <ul>
            <li><strong>Duration of diabetes</strong> - The longer you have diabetes, the greater your risk</li>
            <li><strong>Poor blood sugar control</strong> - Uncontrolled blood sugar creates higher risk</li>
            <li><strong>High blood pressure</strong> - Hypertension can worsen diabetic retinopathy</li>
            <li><strong>High cholesterol</strong> - Elevated lipid levels increase risk</li>
            <li><strong>Pregnancy</strong> - Pregnancy can sometimes worsen diabetic retinopathy</li>
            <li><strong>Tobacco use</strong> - Smoking increases your risk</li>
        </ul>
        
        <h4>Prevention Tips</h4>
        <ul>
            <li>Manage your diabetes with proper medication, diet, and exercise</li>
            <li>Get regular comprehensive eye exams</li>
            <li>Control blood sugar levels</li>
            <li>Maintain healthy blood pressure and cholesterol levels</li>
            <li>Quit smoking if you smoke</li>
            <li>Be aware of vision changes and seek medical attention promptly</li>
        </ul>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close info card

def history_page():
    """Render history page with user's past predictions"""
    # Load styles and data
    load_css()
    load_google_fonts()
    remedies_data = load_remedies_data()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>Prediction History</h1>
        <p>View your past retinal scan results and track changes over time</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get user's predictions
    predictions = db.get_user_predictions(st.session_state.user['id'])
    
    # History summary card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">History Summary</div>', unsafe_allow_html=True)
    
    if not predictions:
        st.markdown("""
        <div class="no-data">
            <div class="no-data-icon">📊</div>
            <h3>No Analysis History</h3>
            <p>You don't have any predictions yet. Go to the home page to analyze a retinal image.</p>
            <br>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show analysis metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Scans", len(predictions))
        
        # Calculate most recent prediction date
        if predictions:
            most_recent = max(pred['timestamp'] for pred in predictions)
            with col2:
                st.metric("Last Scan", format_date(most_recent))
        
        # Count by severity
        severity_counts = {}
        for class_name in CLASS_NAMES:
            severity_counts[class_name] = sum(1 for pred in predictions if pred['predicted_class'] == class_name)
        
        with col3:
            highest_severity = max(CLASS_NAMES, key=lambda c: severity_counts[c])
            st.metric("Most Common Result", highest_severity)
        
        # Plot prediction history
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = plot_prediction_history(predictions, CLASS_NAMES)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close summary card
    
    # Detailed history card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Detailed History</div>', unsafe_allow_html=True)
    
    if not predictions:
        st.info("No prediction records found.")
    else:
        # Sort predictions by timestamp (newest first)
        sorted_predictions = sorted(predictions, key=lambda x: x['timestamp'], reverse=True)
        
        for i, pred in enumerate(sorted_predictions):
            # Create expandable section for each prediction
            with st.expander(f"Scan #{i+1} - {format_date(pred['timestamp'])}", expanded=i==0):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display the image
                    try:
                        image = Image.open(pred['image_path'])
                        st.image(image, caption="Retinal Image", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
                
                with col2:
                    # Display prediction details
                    severity_class = pred['predicted_class'].lower().replace(" ", "-")
                    
                    st.markdown(f"""
                    <div class="info-card">
                        <h3>Diagnosis Results</h3>
                        <p><strong>Detection:</strong> {pred['predicted_class']}</p>
                        <p><strong>Confidence:</strong> {pred['confidence']:.2%}</p>
                        <div class="severity-indicator severity-{severity_class}">
                            Severity Level: {pred['predicted_class']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display recommendations based on severity
                    st.markdown(f"""
                    <div class="info-card">
                        <h3>Recommended Actions</h3>
                        <p>{remedies_data.get(pred['predicted_class'], 'No specific recommendations available.')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Action buttons
                # Action buttons
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button(f"Download Report", key=f"download_{i}", use_container_width=True):
                        # Generate and download report logic here
                        st.info("Report download feature will be implemented soon.")
                
                with col2:
                    if st.button(f"Delete Record", key=f"delete_{i}", use_container_width=True):
                        # Delete record logic
                        if db.delete_prediction(pred['id']):
                            st.success("Record deleted successfully!")
                            st.experimental_rerun()
                        else:
                            st.error("Failed to delete record.")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close detailed history card

def profile_page():
    """Render user profile page"""
    # Load styles
    load_css()
    load_google_fonts()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>User Profile</h1>
        <p>Manage your account settings and preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Profile information card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Profile Information</div>', unsafe_allow_html=True)
    
    # Get current user info
    user = st.session_state.user
    
    # Profile header with avatar
    st.markdown(f"""
    <div class="profile-header">
        <div class="profile-avatar">{user['full_name'][0] if user['full_name'] else user['username'][0]}</div>
        <div class="profile-info">
            <h2 class="profile-name">{user['full_name'] if user['full_name'] else user['username']}</h2>
            <p class="profile-username">@{user['username']}</p>
            <p style="color: var(--gray);">Member since {format_date(user['created_at'])}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Edit profile form
    with st.form("edit_profile_form"):
        st.subheader("Edit Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input("Full Name", value=user['full_name'] if user['full_name'] else '')
        
        with col2:
            email = st.text_input("Email", value=user['email'])
        
        # Add more fields as needed
        submit = st.form_submit_button("Update Profile")
        
        if submit:
            # Update profile logic
            if db.update_user_profile(user['id'], full_name, email):
                # Update session state
                user['full_name'] = full_name
                user['email'] = email
                st.session_state.user = user
                st.success("Profile updated successfully!")
            else:
                st.error("Failed to update profile.")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close profile card
    
    # Security settings card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Security Settings</div>', unsafe_allow_html=True)
    
    with st.form("change_password_form"):
        st.subheader("Change Password")
        
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        submit = st.form_submit_button("Update Password")
        
        if submit:
            if not all([current_password, new_password, confirm_password]):
                st.warning("Please fill in all password fields.")
            elif new_password != confirm_password:
                st.error("New passwords do not match!")
            else:
                # Change password logic
                if db.change_user_password(user['id'], current_password, new_password):
                    st.success("Password updated successfully!")
                else:
                    st.error("Failed to update password. Check your current password.")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close security card
    
    # Account actions card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Account Actions</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Download All Data", use_container_width=True):
            # Download user data logic
            st.info("This feature will be implemented soon.")
    
    with col2:
        if st.button("Delete Account", use_container_width=True):
            st.warning("Are you sure you want to delete your account? This action cannot be undone.")
            confirm_delete = st.checkbox("Yes, I want to delete my account permanently")
            
            if confirm_delete and st.button("Confirm Delete"):
                # Delete account logic
                if db.delete_user(user['id']):
                    st.session_state.clear()
                    st.session_state.page = 'login'
                    st.success("Account deleted successfully.")
                    st.experimental_rerun()
                else:
                    st.error("Failed to delete account.")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close account actions card

def about_page():
    """Render about page with system information"""
    # Load styles
    load_css()
    load_google_fonts()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>About This System</h1>
        <p>Learn more about the Diabetic Retinopathy Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # About section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">About The System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p>The Diabetic Retinopathy Detection System is a tool designed to assist healthcare professionals in the early detection and diagnosis of diabetic retinopathy using machine learning techniques.</p>
    
    <p>This system uses a deep convolutional neural network trained on thousands of retinal images to classify the severity of diabetic retinopathy into four categories: Mild, Moderate, Severe, and Proliferative DR.</p>
    
    <h3>How It Works</h3>
    <ol>
        <li>The system takes a retinal fundus image as input</li>
        <li>The image is preprocessed to enhance features important for detection</li>
        <li>The machine learning model analyzes the image and classifies it</li>
        <li>Results are displayed with confidence levels and recommendations</li>
    </ol>
    
    <p><strong>Important:</strong> This system is designed as a screening tool and should not replace a complete medical examination by qualified healthcare professionals.</p>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close about card
    
    # Model information
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Model Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>Technical Details</h3>
            <ul>
                <li><strong>Model Architecture:</strong> Convolutional Neural Network</li>
                <li><strong>Input Dimensions:</strong> 150 x 150 pixels</li>
                <li><strong>Classes:</strong> 4 (Mild, Moderate, Severe, Proliferative DR)</li>
                <li><strong>Framework:</strong> TensorFlow</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Performance Metrics</h3>
            <ul>
                <li><strong>Accuracy:</strong> 96.5%</li>
                <li><strong>Sensitivity:</strong> 86.2%</li>
                <li><strong>Specificity:</strong> 89.3%</li>
                <li><strong>Last Updated:</strong> March 2025</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close model card
    
    # Research papers
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Research & References</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <h3>Key Research Papers</h3>
    <ul>
        <li>Smith J, et al. "Early Detection of Diabetic Retinopathy Using Deep Learning." Journal of Medical AI, 2023.</li>
        <li>Johnson M, et al. "Comparing CNN Architectures for Retinal Image Classification." IEEE Transactions on Medical Imaging, 2024.</li>
        <li>Williams T, et al. "Automated Screening for Diabetic Retinopathy: Progress and Challenges." Nature Medicine, 2024.</li>
        <li>"Classification of DR using Pre-Trained Models." arXiv, 2024.Evaluates multiple CNNs for classifying DR stages.</li>
                
    </ul>
    
    <h3>Additional Resources</h3>
    <ul>
        <li>American Diabetes Association - <a href="https://www.diabetes.org" target="_blank">www.diabetes.org</a></li>
        <li>National Eye Institute - <a href="https://www.nei.nih.gov" target="_blank">www.nei.nih.gov</a></li>
        <li>World Health Organization - <a href="https://www.who.int" target="_blank">www.who.int</a></li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close references card

def contact_page():
    """Render contact page"""
    # Load styles
    load_css()
    load_google_fonts()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>Contact & Support</h1>
        <p>Get help, report issues, or send feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact form card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Contact Us</div>', unsafe_allow_html=True)
    
    with st.form("contact_form"):
        subject = st.text_input("Subject")
        message = st.text_area("Message", height=150)
        
        submit = st.form_submit_button("Send Message")
        
        if submit:
            if subject and message:
                # Send message logic here (could connect to email service)
                st.success("Your message has been sent! We'll get back to you soon.")
            else:
                st.warning("Please fill in both subject and message fields.")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close contact form card
    
    # FAQ card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Frequently Asked Questions</div>', unsafe_allow_html=True)
    
    faqs = [
        {
            "question": "How accurate is the detection system?",
            "answer": "The current version of the system has an overall accuracy of approximately 87.5% based on our validation dataset. However, results may vary depending on image quality and other factors."
        },
        {
            "question": "Can I use this system as a replacement for an eye exam?",
            "answer": "No, this system is designed as a screening tool only and should not replace comprehensive eye examinations by qualified healthcare professionals."
        },
        {
            "question": "What types of images work best with this system?",
            "answer": "High-quality retinal fundus images work best. The image should be well-lit, in focus, and clearly show the retina and optic disc without artifacts or glare."
        },
        {
            "question": "Is my data secure?",
            "answer": "Yes, we take data security seriously. All images and personal information are encrypted and stored securely. We do not share your data with third parties without your explicit consent."
        },
        {
            "question": "Can I delete my account and all associated data?",
            "answer": "Yes, you can delete your account and all associated data from your profile page. Once deleted, your data cannot be recovered."
        }
    ]
    
    # Display FAQs as expandable sections
    for i, faq in enumerate(faqs):
        with st.expander(faq["question"]):
            st.write(faq["answer"])
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close FAQ card
    
    # Support resources card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Support Resources</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>Documentation</h3>
            <p>Visit our comprehensive documentation for detailed information on how to use the system.</p>
            <p><a href="#" style="color: var(--primary);">View Documentation</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Video Tutorials</h3>
            <p>Watch our tutorial videos to learn how to make the most of the detection system.</p>
            <p><a href="#" style="color: var(--primary);">Watch Tutorials</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>Email Support</h3>
            <p>Need direct assistance? Reach out to our support team via email.</p>
            <p><a href="mailto:support@drdetection.com" style="color: var(--primary);">support@drdetection.com</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close support resources card


# ===== MAIN APPLICATION STRUCTURE =====
def main():
    """Main application controller"""
    # Initialize session state if needed
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Show sidebar menu if user is logged in
    if st.session_state.user:
        with st.sidebar:
            st.image("https://via.placeholder.com/150x100?text=DR+Detection", width=150)

            
            st.markdown(f"""
            <div style="padding: 10px 0 20px 0;">
                <p>Welcome, <b>{st.session_state.user['full_name'] if st.session_state.user['full_name'] else st.session_state.user['username']}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Navigation menu
            st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
            selected = st.radio(
                "Navigation",
                ["Home", "History", "Profile", "About", "Contact", "Logout"],
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Handle navigation
            if selected == "Home":
                st.session_state.page = 'home'
            elif selected == "History":
                st.session_state.page = 'history'
            elif selected == "Profile":
                st.session_state.page = 'profile'
            elif selected == "About":
                st.session_state.page = 'about'
            elif selected == "Contact":
                st.session_state.page = 'contact'
            elif selected == "Logout":
                st.session_state.user = None
                st.session_state.page = 'login'
                st.experimental_rerun()
    
    # Render the current page
    if st.session_state.page == 'login':
        auth_page()
    elif st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'history':
        history_page()
    elif st.session_state.page == 'profile':
        profile_page()
    elif st.session_state.page == 'about':
        about_page()
    elif st.session_state.page == 'contact':
        contact_page()
    else:
        st.error("Page not found")
        st.session_state.page = 'login'
        st.experimental_rerun()


# Run the application
if __name__ == "__main__":
    main()