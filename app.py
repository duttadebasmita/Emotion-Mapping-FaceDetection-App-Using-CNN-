import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Defining the emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def get_emotion_label(prediction):
    maxindex = int(np.argmax(prediction))
    return emotion_dict[maxindex]

def get_result(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('C:\\Emotion-detection\\src\\haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return "No face detected"

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        emotion = get_emotion_label(prediction)
        return emotion

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static/uploads', secure_filename(f.filename))
        if not os.path.exists(os.path.join(basepath, 'static/uploads')):
            os.makedirs(os.path.join(basepath, 'static/uploads'))
        f.save(file_path)
        result = get_result(file_path)
        return render_template('result.html', image_url=url_for('static', filename='uploads/' + secure_filename(f.filename)), emotion=result)
    return None

if __name__ == '__main__':
    model_file_path = 'C:\Emotion-detection\model.h5'
    
    # Defining the model architecture again
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    
    # Load the weights into the model
    model.load_weights(model_file_path)
    print("Model weights loaded successfully from:", model_file_path)
    
    app.run(debug=True)
