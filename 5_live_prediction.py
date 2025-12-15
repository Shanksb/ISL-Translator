import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

model = load_model(r"c:\Project\Sem 3\asl_alphabet_model_finetuned.keras")

labels = [chr(i) for i in range(65, 91)] 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

preds_queue = deque(maxlen=10) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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

        
        roi = frame[y_min:y_max, x_min:x_max]

        if roi.size != 0:
            
            roi = cv2.resize(roi, (224, 224))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=0)

            
            predictions = model.predict(roi, verbose=0)
            pred_label = labels[np.argmax(predictions)]

            preds_queue.append(pred_label)
            most_common = max(set(preds_queue), key=preds_queue.count)

            
            cv2.putText(frame, f"Prediction: {most_common}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    
    cv2.imshow("ISL Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
