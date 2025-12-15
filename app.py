from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import json
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque


app = Flask(__name__)

# Load the model
model = load_model("asl_alphabet_model_finetuned.keras")
print("✅ Model loaded successfully!")

# Load dictionary
with open("static/words.json", "r") as f:
    WORD_LIST = json.load(f)
print("✅ Word list loaded successfully!")

labels = [chr(i) for i in range(65, 91)]

# Mediapipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

preds_queue = deque(maxlen=10)

camera = None
current_prediction = ""


#camera 
def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera



def process_frame(frame):
    global current_prediction

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        x_min, y_min, x_max, y_max = w, h, 0, 0

        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)

        margin = 30
        x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
        x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        roi = frame[y_min:y_max, x_min:x_max]

        if roi.size != 0:
            # Preprocess ROI
            roi_resized = cv2.resize(roi, (224, 224))
            roi_processed = roi_resized.astype("float32") / 255.0
            roi_processed = np.expand_dims(roi_processed, axis=0)

            # Predict
            predictions = model.predict(roi_processed, verbose=0)
            pred_label = labels[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            # Update prediction queue
            preds_queue.append(pred_label)
            most_common = max(set(preds_queue), key=preds_queue.count)
            current_prediction = most_common

            # Display
            cv2.putText(frame, f"Sign: {most_common}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

            cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        # Draw hand landmarks
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame


# --------------------------------------------------------
# Video Streaming Generator
# --------------------------------------------------------
def generate_frames():
    cap = get_camera()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# --------------------------------------------------------
# ROUTES
# --------------------------------------------------------

# Landing Page
@app.route('/')
def index():
    return render_template('index.html')


# Prediction Page (Webcam UI)
@app.route('/predict')
def predict():
    return render_template('predict.html')


# Video Feed Route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# API Route for AJAX Prediction Text
@app.route('/get_prediction')
def get_prediction():
    return jsonify({'prediction': current_prediction})


# Clear Prediction
@app.route('/clear_prediction')
def clear_prediction():
    global current_prediction
    preds_queue.clear()
    current_prediction = ""
    return jsonify({'status': 'cleared'})

@app.route('/suggest/<prefix>')
def suggest(prefix):
    prefix = prefix.lower()

    # Simple offline autocomplete: return words starting with prefix
    suggestions = [w for w in WORD_LIST if w.startswith(prefix)]

    # Limit to 5 suggestions
    suggestions = suggestions[:5]

    return jsonify({"suggestions": suggestions})


@app.route("/alphabets")
def alphabets():
    # List of images and characters
    letters = [
        {"char": "A", "image": "A.jpg"},
        {"char": "B", "image": "B.jpg"},
        {"char": "C", "image": "C.jpg"},
        # Continue till Z
    ]
    return render_template("alphabets.html", alphabets=letters)



# --------------------------------------------------------
# Run Server
# --------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
