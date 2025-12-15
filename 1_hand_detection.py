# 1_hand_detection.py
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                h, w, _ = image.shape
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                x1, x2 = int(min(xs) * w), int(max(xs) * w)
                y1, y2 = int(min(ys) * h), int(max(ys) * h)
                cv2.rectangle(image, (x1-10, y1-10), (x2+10, y2+10), (0,255,0), 2)

        cv2.imshow("Hand Detection", image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()

