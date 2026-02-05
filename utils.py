import os
import uuid
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Constants
UPLOAD_FOLDER = 'uploads'
IMAGE_SIZE = (150, 150)  # Must match the model's expected input size

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to the uploads folder with a unique filename."""
    # Create uploads directory if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Generate a unique filename
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    unique_filename = f"{str(uuid.uuid4())}{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def preprocess_image(image_path):
    """Preprocess the image for model prediction."""
    # Load and resize image
    img = Image.open(image_path)
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    
    # Convert to RGB if image is grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    
    # Handle RGBA images (e.g., PNGs with transparency)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Normalize pixel values
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_class_color(class_name):
    """Get color for class visualization."""
    colors = {
        'Mild': '#f1c40f',        # Yellow
        'Moderate': '#e67e22',    # Orange
        'Severe': '#e74c3c',      # Red
        'Proliferative DR': '#c0392b'  # Dark Red
    }
    return colors.get(class_name, '#3498db')  # Default blue

def plot_prediction_confidence(prediction, class_names):
    """Create a bar chart of prediction confidence levels."""
    confidences = prediction[0]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    for i, (class_name, confidence) in enumerate(zip(class_names, confidences)):
        fig.add_trace(go.Bar(
            x=[class_name],
            y=[confidence * 100],  # Convert to percentage
            name=class_name,
            marker_color=get_class_color(class_name),
            text=[f"{confidence * 100:.1f}%"],
            textposition='auto'
        ))
    
    # Update layout
    fig.update_layout(
        title="Prediction Confidence Levels",
        yaxis=dict(
            title="Confidence (%)",
            range=[0, 100]
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    
    return fig

def plot_prediction_history(predictions, class_names):
    """Create a line chart showing prediction history over time."""
    # Sort predictions by timestamp (oldest first)
    sorted_preds = sorted(predictions, key=lambda x: x['timestamp'])
    
    # Extract dates and classes
    dates = [datetime.fromisoformat(pred['timestamp']) for pred in sorted_preds]
    classes = [pred['predicted_class'] for pred in sorted_preds]
    
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add line chart for severity over time
    severity_levels = {class_name: i for i, class_name in enumerate(class_names)}
    severity_y = [severity_levels[cls] for cls in classes]
    
    # Create mapping for y-axis ticks
    severity_y_ticks = list(range(len(class_names)))
    severity_y_ticktext = class_names
    
    # Add the line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=severity_y,
            mode='lines+markers',
            name='Severity Level',
            line=dict(color='#3498db', width=3),
            marker=dict(
                size=10,
                color=[get_class_color(cls) for cls in classes],
                line=dict(color='white', width=2)
            )
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Retinopathy Progression Over Time",
        xaxis=dict(
            title="Date",
            tickformat="%b %d, %Y"
        ),
        yaxis=dict(
            title="Severity Level",
            tickvals=severity_y_ticks,
            ticktext=severity_y_ticktext
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
    )
    
    return fig

def format_date(iso_date_string):
    """Format ISO date string to a readable format."""
    try:
        dt = datetime.fromisoformat(iso_date_string)
        return dt.strftime("%b %d, %Y, %I:%M %p")
    except (ValueError, TypeError):
        return "Unknown date"

def generate_report(prediction, user_info):
    """Generate a printable report for the prediction."""
    # This would generate an HTML or PDF report
    # Implementation depends on specific requirements
    pass

def apply_image_enhancements(image_path):
    """Apply image enhancements to improve quality for analysis."""
    try:
        img = Image.open(image_path)
        
        # Apply some basic enhancements
        # This is a placeholder - actual image processing would depend on specific requirements
        
        # Save enhanced image
        enhanced_path = f"{os.path.splitext(image_path)[0]}_enhanced{os.path.splitext(image_path)[1]}"
        img.save(enhanced_path)
        
        return enhanced_path
    except Exception as e:
        print(f"Image enhancement error: {str(e)}")
        return image_path  # Return original if enhancement fails