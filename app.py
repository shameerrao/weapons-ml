# Import necessary modules
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_bootstrap import Bootstrap
from flask_httpauth import HTTPBasicAuth
from dotenv import load_dotenv
import os, uuid, scrypt, cv2, logging
from roboflow import Roboflow
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG

# Load environment variables from .env file
load_dotenv()

# Create Flask application instance
app = Flask(__name__)
Bootstrap(app)  # Initialize Flask Bootstrap extension
app.secret_key = os.urandom(24)  # Set secret key for session management
auth = HTTPBasicAuth()  # Initialize HTTP Basic Authentication

# Define user credentials (username:password hash) for authentication
users = {
    "admin": os.getenv('ADMIN_PASSWORD_HASH', 'default_hash')
}

# Define function to verify user's password during authentication
@auth.verify_password
def verify_password(username, password):
    stored_hash = users.get(username)
    if stored_hash is not None:
        _, salt, hash = stored_hash.split('$')
        password_bytes = password.encode('utf-8')
        salt_bytes = salt.encode('utf-8')
        computed_hash = scrypt.hash(password_bytes, salt_bytes, N=32768, r=8, p=1).hex()
        if computed_hash == hash:
            return username
    return False

# Define routes and associated functions

# Home page route
@app.route('/')
@auth.login_required  # Require authentication for accessing this route
def index():
    return render_template('index.html')

# About Us page route
@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/result')
def result(): 
    return render_template('result.html')

# Model Visualization page route
@app.route('/modelvisualization')
def modelvisualization():
    return render_template('modelvisualization.html')

# Future Improvements page route
@app.route('/futureimprovements')
def futureimprovements():
    return render_template('futureimprovements.html')

# Code Models page route
@app.route('/codemodels')
def codemodels():
    return render_template('codemodels.html')

# Function to handle file uploads
def handle_upload(file):
    _, file_extension = os.path.splitext(file.filename)
    unique_filename = str(uuid.uuid4()) + file_extension
    filepath = os.path.join(app.static_folder, 'uploads', unique_filename)
    file.save(filepath)
    return filepath, unique_filename

ROBOFLOW_SIZE = 416  # Define the desired size for Roboflow

os.environ['ROBOFLOW_API_KEY'] = 'UGY1VgI8gHZ4Xf7S6UdA'

def draw_instance_segmentation(image, predictions, class_name):
    color_map = {
        'Rifle': (255, 0, 0),    # Red
        'Handgun': (0, 255, 0),  # Green
        'Knife': (0, 0, 255),    # Blue
        'Toy': (255, 255, 0),    # Cyan
        'Null': (255, 0, 255)    # Magenta
    }
    
    for prediction in predictions:
        class_name = predictions[0]['class']
        points = prediction['points']
        points = np.array([(int(point['x']), int(point['y'])) for point in points], dtype=np.int32)
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 0), thickness=2)
        
        # Get the color for the class from the color map
        color = color_map.get(class_name, (0, 0, 0))  # Default to black if class is not found
        
        # Draw filled polygon with color corresponding to the class
        overlay = image.copy()
        cv2.fillPoly(overlay, [points], color=color)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # Draw class name on the image
        cv2.putText(image, class_name, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# Update the upload_v3 route to use instance segmentation for annotation
@app.route('/upload_v3', methods=['GET','POST'])
def upload_v3():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        filepath, unique_filename = handle_upload(file)
        if filepath is None:
            return redirect(request.url)

    try:
        image = cv2.imread(filepath)
        # Run model prediction on the image using Roboflow
        rf = Roboflow(api_key="UGY1VgI8gHZ4Xf7S6UdA")
        project = rf.workspace("morgan-hacks-u4xp9").project("gun-instance-segmentation-model")
        model = project.version(1).model
        predictions_json = model.predict(filepath, confidence=50).json()
      
        # Log successful inference
        logging.debug("Inference successful.")

        # Draw instance segmentation on the image
        draw_instance_segmentation(image, predictions_json['predictions'],class_name = predictions_json['predictions'][0]['class'])

        # Save the annotated image
        annotated_filename = f'annotated_{unique_filename}'
        annotated_filepath = os.path.join(app.static_folder, 'uploads', annotated_filename)
        cv2.imwrite(annotated_filepath, image)

        # Log the annotated image filepath
        logging.debug(f"Annotated image saved to: {annotated_filepath}")

        # Construct the URL for the annotated image
        image_url = url_for('static', filename=f'uploads/{annotated_filename}')

        # Log the image URL
        logging.debug(f"Annotated image URL: {image_url}")
    
        # Render result template with annotated image and detection details
        return render_template('result.html', image_url=image_url, detection_found=True, class_name = predictions_json['predictions'][0]['class'], confidence = predictions_json['predictions'][0]['confidence']) 

    except Exception as e:
        # Log the exception
        logging.error(f"An error occurred during inference: {str(e)}")
        flash(f"An error occurred during inference: {str(e)}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Ensure uploads directory exists
    uploads_dir = os.path.join(app.static_folder, 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
    app.run(debug=True)  # Run the Flask application in debug mode