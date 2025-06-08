# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib
#
# clf = joblib.load('../models/hand_sign_rf.pkl')
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp_draw = mp.solutions.drawing_utils
#
# cap = cv2.VideoCapture(0)
# labels = ['A', 'B', 'C']  # Update with your actual labels
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
#             prediction = clf.predict([landmarks])[0]
#             cv2.putText(frame, f'Predicted: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#     cv2.imshow("Frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

#
# import cv2
# import numpy as np
# from cvzone.HandTrackingModule import HandDetector
# from tensorflow.keras.models import load_model
#
# # Initialize camera and detector
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
#
# # Load model and labels
# model = load_model('model/keras_model.h5')
# with open('model/labels.txt', 'r') as f:
#     labels = [line.strip() for line in f]
#
# while True:
#     success, img = cap.read()
#     hands, img = detector.findHands(img)
#
#     if hands:
#         hand = hands[0]
#         bbox = hand["bbox"]
#
#         # Extract hand region with boundary checks
#         x, y, w, h = bbox
#         x1 = max(0, x - 20)
#         y1 = max(0, y - 20)
#         x2 = min(img.shape[1], x + w + 20)
#         y2 = min(img.shape[0], y + h + 20)
#
#         hand_img = img[y1:y2, x1:x2]
#         hand_img = cv2.resize(hand_img, (224, 224))
#
#         # Preprocess for Teachable Machine (critical!)
#         hand_img = np.expand_dims(hand_img, axis=0)
#         hand_img = hand_img.astype(np.float32) / 255.0  # Normalize
#
#         # Make prediction
#         predictions = model.predict(hand_img)
#         class_id = np.argmax(predictions)
#         confidence = predictions[0][class_id]
#
#         # Get actual label text
#         label_text = labels[class_id]
#
#         # Display results
#         cv2.putText(img, f"{label_text} ({confidence:.2f})",
#                     (bbox[0], bbox[1] - 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     cv2.imshow("Hand Gestures", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#


import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D  # Critical fix

# Initialize camera and detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load model with custom layer mapping
model = load_model(
    'model/keras_model.h5',
    custom_objects={'DepthwiseConv2D': DepthwiseConv2D}
)

with open('model/labels.txt', 'r') as f:
    labels = [line.strip() for line in f]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        bbox = hand["bbox"]

        # Extract hand region
        x, y, w, h = bbox
        x1 = max(0, x - 20)
        y1 = max(0, y - 20)
        x2 = min(img.shape[1], x + w + 20)
        y2 = min(img.shape[0], y + h + 20)

        hand_img = img[y1:y2, x1:x2]
        hand_img = cv2.resize(hand_img, (224, 224))

        # Preprocess (BGR→RGB + normalize)
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
        hand_img = np.expand_dims(hand_img, axis=0).astype(np.float32) / 255.0

        # Predict
        predictions = model.predict(hand_img)
        class_id = np.argmax(predictions)
        confidence = predictions[0][class_id]

        # Display
        cv2.putText(img, f"{labels[class_id]} {confidence * 100:.1f}%",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Hand Gestures", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
