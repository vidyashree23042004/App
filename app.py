import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import firebase_admin
from firebase_admin import credentials, db
import smtplib
from email.message import EmailMessage
import pickle
import json
from keras.models import load_model
app = Flask(__name__)

# Load trained models
model = load_model('resnet50.keras')
with open('crop_MODEL.pkl', 'rb') as f:
    irrigation_model = pickle.load(f)

print('Models loaded. Check http://127.0.0.1:5000/')
# cred = credentials.Certificate("agriculture.json")
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://agricultural-project-85c18-default-rtdb.asia-southeast1.firebasedatabase.app/'
# })

THRESHOLDS = {
    "temperature": {"min": 20, "max": 35},
    "humidity": {"min": 50, "max": 85},
    "soil_moisture": {"min": 2, "max": 100}
}

last_alert_time = {}
alert_interval = 300

# Class labels for the new model
classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper_bell___Bacterial_spot",
    "Pepper_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Solutions for each class
solutions = {
    "Apple___Apple_scab": "Solutions: 1. Apply fungicides like Captan or Myclobutanil. 2. Prune and destroy infected leaves and branches. 3. Ensure good air circulation by proper spacing.",
    "Apple___Black_rot": "Solutions: 1. Remove mummified fruits and infected branches. 2. Apply fungicides during bloom period. 3. Practice good sanitation in orchard.",
    "Apple___Cedar_apple_rust": "Solutions: 1. Remove nearby juniper plants if possible. 2. Apply fungicides in early spring. 3. Plant resistant varieties.",
    "Apple___healthy": "Solution: The plant is healthy. Keep monitoring and maintain optimal growing conditions.",
    "Blueberry___healthy": "Solution: The plant is healthy. Continue proper care and monitoring.",
    "Cherry_(including_sour)___Powdery_mildew": "Solutions: 1. Apply sulfur or potassium bicarbonate. 2. Improve air circulation. 3. Avoid overhead watering.",
    "Cherry_(including_sour)___healthy": "Solution: The plant is healthy. Maintain current care practices.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Solutions: 1. Use resistant varieties. 2. Practice crop rotation. 3. Apply fungicides if necessary.",
    "Corn_(maize)___Common_rust_": "Solutions: 1. Plant resistant hybrids. 2. Apply fungicides when needed. 3. Remove crop debris after harvest.",
    "Corn_(maize)___Northern_Leaf_Blight": "Solutions: 1. Use resistant hybrids. 2. Rotate crops. 3. Apply fungicides during silking if disease is severe.",
    "Corn_(maize)___healthy": "Solution: The plant is healthy. Continue good agricultural practices.",
    "Grape___Black_rot": "Solutions: 1. Apply fungicides during early growth stages. 2. Remove infected plant material. 3. Improve air circulation.",
    "Grape___Esca_(Black_Measles)": "Solutions: 1. Prune infected vines. 2. Apply wound protectants after pruning. 3. Maintain vine health.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Solutions: 1. Apply fungicides. 2. Remove infected leaves. 3. Improve air circulation.",
    "Grape___healthy": "Solution: The plant is healthy. Continue proper vineyard management.",
    "Orange___Haunglongbing_(Citrus_greening)": "Solutions: 1. Remove infected trees. 2. Control psyllid vectors. 3. Use disease-free nursery stock.",
    "Peach___Bacterial_spot": "Solutions: 1. Apply copper-based sprays. 2. Plant resistant varieties. 3. Avoid overhead irrigation.",
    "Peach___healthy": "Solution: The plant is healthy. Maintain current care practices.",
    "Pepper_bell___Bacterial_spot": "Solutions: 1. Use disease-free seeds. 2. Apply copper sprays. 3. Avoid working with wet plants.",
    "Pepper_bell___healthy": "Solution: The plant is healthy. Continue proper care.",
    "Potato___Early_blight": "Solutions: 1. Rotate crops. 2. Apply fungicides. 3. Remove infected leaves.",
    "Potato___Late_blight": "Solutions: 1. Destroy infected plants. 2. Apply fungicides preventatively. 3. Avoid overhead irrigation.",
    "Potato___healthy": "Solution: The plant is healthy. Maintain good growing conditions.",
    "Raspberry___healthy": "Solution: The plant is healthy. Continue current care.",
    "Soybean___healthy": "Solution: The plant is healthy. Keep monitoring fields.",
    "Squash___Powdery_mildew": "Solutions: 1. Apply sulfur or potassium bicarbonate. 2. Plant resistant varieties. 3. Improve air circulation.",
    "Strawberry___Leaf_scorch": "Solutions: 1. Remove infected leaves. 2. Apply fungicides. 3. Improve air flow.",
    "Strawberry___healthy": "Solution: The plant is healthy. Maintain current practices.",
    "Tomato___Bacterial_spot": "Solutions: 1. Use disease-free seeds. 2. Apply copper sprays. 3. Avoid overhead watering.",
    "Tomato___Early_blight": "Solutions: 1. Rotate crops. 2. Apply fungicides. 3. Remove lower leaves.",
    "Tomato___Late_blight": "Solutions: 1. Destroy infected plants. 2. Apply fungicides. 3. Avoid wet conditions.",
    "Tomato___Leaf_Mold": "Solutions: 1. Improve ventilation. 2. Apply fungicides. 3. Reduce humidity.",
    "Tomato___Septoria_leaf_spot": "Solutions: 1. Remove infected leaves. 2. Apply fungicides. 3. Avoid overhead watering.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Solutions: 1. Apply miticides. 2. Increase humidity. 3. Introduce predatory mites.",
    "Tomato___Target_Spot": "Solutions: 1. Apply fungicides. 2. Remove infected leaves. 3. Improve air circulation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Solutions: 1. Remove infected plants. 2. Control whiteflies. 3. Use resistant varieties.",
    "Tomato___Tomato_mosaic_virus": "Solutions: 1. Remove infected plants. 2. Disinfect tools. 3. Control aphids.",
    "Tomato___healthy": "Solution: The plant is healthy. Continue proper care and monitoring."
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

def getResult(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_name = classes[predicted_class_index]
    solution = solutions.get(predicted_class_name, "No specific solution available.")
    
    return predicted_class_name, solution

def predict_irrigation(temperature, moisture):
    try:
        input_data = np.array([[moisture, temperature]])
        prediction = irrigation_model.predict(input_data)
        return int(prediction[0])  # Convert numpy int64 to Python int
    except Exception as e:
        print(f"Error in irrigation prediction: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index1.html')

@app.route('/sensor-data')
def sensor_data():
    ref = db.reference('/sensor_data')
    data = ref.get()
    if data:
        return jsonify(data)
    return jsonify({'error': 'No data found'})

@app.route('/get_sensor_data')
def get_sensor_data():
    sensor_ref = db.reference('/sensor_data')
    sensor_values = sensor_ref.get()
    alerts = []
    irrigation_prediction = None
    
    if sensor_values:
        # Convert sensor values to float and handle None cases
        temp = float(sensor_values.get('temperature', 0))
        moisture = float(sensor_values.get('soil_moisture', 0))
        
        # Get irrigation prediction
        irrigation_prediction = predict_irrigation(temp, moisture)
        
        # Check thresholds
        for sensor, value in sensor_values.items():
            current_time = time.time()
            value = float(value)
            
            if value < THRESHOLDS[sensor]['min']:
                if sensor not in last_alert_time or (current_time - last_alert_time[sensor] > alert_interval):
                    alerts.append(f"{sensor.replace('_', ' ').title()} is TOO LOW! ({value})")
                    last_alert_time[sensor] = current_time
            elif value > THRESHOLDS[sensor]['max']:
                if sensor not in last_alert_time or (current_time - last_alert_time[sensor] > alert_interval):
                    alerts.append(f"{sensor.replace('_', ' ').title()} is TOO HIGH! ({value})")
                    last_alert_time[sensor] = current_time
    
    response = {
        "sensor_data": sensor_values,
        "alerts": alerts,
        "irrigation_prediction": irrigation_prediction
    }
    
    return json.dumps(response, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}


@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'})
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, secure_filename(f.filename))
    f.save(file_path)
    predicted_label, solution = getResult(file_path)
    return jsonify({'prediction': predicted_label, 'solution': solution})

@app.route('/auto_email', methods=['POST'])
def auto_email():
    data = request.json
    subject = "Automated Plant Disease Detection Report"
    prediction = data.get('prediction')
    solution_html = data.get('solution')
    temperature = data.get('temperature')
    humidity = data.get('humidity')
    soil_moisture = data.get('soil_moisture')
    irrigation = data.get('irrigation')

    message = f"""
    <p style="font-style: italic; color: gray;">
        This is an AI-generated email. Please do not reply. The following is an auto-generated report.
    </p>
    <hr>
    <h2>Plant Disease Detection Report</h2>
    <p><strong>Prediction:</strong> {prediction}</p>
    <p><strong>Temperature:</strong> {temperature}°C</p>
    <p><strong>Humidity:</strong> {humidity}%</p>
    <p><strong>Soil Moisture:</strong> {soil_moisture}%</p>
    <p><strong>Irrigation Recommendation:</strong> {irrigation}</p>
    <br>
    <h3>Solutions:</h3>
    {solution_html}
    """

    try:
        email = EmailMessage()
        email['Subject'] = subject
        email['From'] = "yasmfathima03@gmail.com"
        email['To'] = "yasmfathima03@gmail.com"  # Change to the static recipient
        email.set_content("HTML version required", subtype='plain')
        email.add_alternative(message, subtype='html')

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login("yasmfathima03@gmail.com", "upcybfxqosazglwt")  # App password
            smtp.send_message(email)

        return jsonify({'message': 'Auto email sent successfully!'}), 200
    except Exception as e:
        print("Error sending auto email:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/send_email', methods=['POST'])
def send_email():
    data = request.json
    recipient = data.get('email')
    subject = "Plant Disease Detection Report"
    prediction = data.get('prediction')
    solution_html = data.get('solution')
    temperature = data.get('temperature')
    humidity = data.get('humidity')
    soil_moisture = data.get('soil_moisture')
    irrigation = data.get('irrigation')

    message = f"""
    <p style="font-style: bold; color: black;">
        This is an AI-generated email. Please do not reply to this address. Kindly find the analyzed report below.
    </p>
    <hr>
    <h2>Plant Disease Detection Report</h2>
    <p><strong>Prediction:</strong> {prediction}</p>
    <p><strong>Temperature:</strong> {temperature}°C</p>
    <p><strong>Humidity:</strong> {humidity}%</p>
    <p><strong>Soil Moisture:</strong> {soil_moisture}%</p>
    <p><strong>Irrigation Recommendation:</strong> {irrigation}</p>
    <br>
    <h3>Solutions:</h3>
    {solution_html}
    """

    try:
        email = EmailMessage()
        email['Subject'] = subject
        email['From'] = "yasmfathima03@gmail.com"
        email['To'] = recipient
        email.set_content("HTML version required", subtype='plain')
        email.add_alternative(message, subtype='html')

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login("yasmfathima03@gmail.com", "upcy bfxq osaz glwt")
            smtp.send_message(email)

        return jsonify({'message': 'Email sent successfully!'}), 200
    except Exception as e:
        print("Error sending email:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)