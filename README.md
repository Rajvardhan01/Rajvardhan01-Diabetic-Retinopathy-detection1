# Diabetic Retinopathy Detection System

A comprehensive web application for detecting diabetic retinopathy from retinal images using deep learning.

## Features

- User authentication and profile management
- Image upload and preprocessing
- Deep learning model prediction
- Prediction history tracking
- Interactive data visualization
- Remedies and recommendations based on prediction results
- Color-coded severity indicators
- User-friendly interface

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your trained model file (`model.h5`) in the `model` directory

4. Create a `remedies.json` file with remedies for each class:
```json
{
    "Mild": "Your remedy for mild cases",
    "Moderate": "Your remedy for moderate cases",
    "Severe": "Your remedy for severe cases",
    "Proliferative DR": "Your remedy for proliferative DR cases"
}
```

5. Create an `assets` directory and add:
   - `logo.png` - Application logo
   - `login_image.png` - Login page image

## Running the Application

Run the following command in your terminal:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Usage

1. Sign up for an account or login if you already have one
2. Upload a retinal image
3. View prediction results with confidence scores
4. Check your prediction history
5. Update your profile information

## Project Structure

```
.
├── app.py                 # Main application file
├── database.py            # Database management
├── utils.py               # Utility functions
├── requirements.txt       # Project dependencies
├── model/                 # Model directory
│   └── model.h5          # Trained model
├── assets/               # Static assets
│   ├── logo.png         # Application logo
│   └── login_image.png  # Login page image
└── remedies.json        # Remedies data
```

## Model Information

The model is trained on a dataset of retinal images and can classify diabetic retinopathy into the following categories:
- Mild
- Moderate
- Severe
- Proliferative DR

## Note

This application is for educational and research purposes only. Always consult with a medical professional for actual diagnosis and treatment. 