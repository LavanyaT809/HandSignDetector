# import cv2
# import mediapipe as mp
# import os
# import numpy as np
#
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
#
# # Initialize MediaPipe Hands with proper parameters
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )
#
# # Create data folder if it doesn't exist
# sign_label = 'A'
# save_dir = os.path.join('data', sign_label)
# os.makedirs(save_dir, exist_ok=True)
# count = 0
#
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         continue
#
#     # Convert to RGB and ensure contiguous array
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img_rgb = np.ascontiguousarray(img_rgb)  # Critical fix
#
#     # Process image
#     results = hands.process(img_rgb)
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Extract landmarks
#             landmarks = np.array([[lm.x, lm.y, lm.z]
#                                   for lm in hand_landmarks.landmark]).flatten()
#
#             # Save landmarks
#             np.save(os.path.join(save_dir, f'landmark_{count}.npy'), landmarks)
#             count += 1
#
#             # Visual feedback
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             cv2.putText(frame, f"Saved: {count}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     cv2.imshow('Hand Data Collection', frame)
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()





# import cv2
# import mediapipe as mp
# import os
# import numpy as np
#
# # Initialize mediapipe hands module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )
# mp_draw = mp.solutions.drawing_utils
#
# # Configuration
# folder = "data/A"
# os.makedirs(folder, exist_ok=True)
# counter = 0
# imgSize = 300
# offset = 20
#
# cap = cv2.VideoCapture(0)
#
# # Initialize imgWhite outside loop
# imgWhite = None
#
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         continue
#
#     frame = cv2.flip(frame, 1)  # Mirror display
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)
#
#     hand_detected = False
#     imgWhite = None  # Reset for each frame
#
#     if results.multi_hand_landmarks:
#         hand_detected = True
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Get hand bounding box
#             x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
#             y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
#             x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
#             y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
#
#             # Convert to integers and add offset
#             x, y = int(x_min) - offset, int(y_min) - offset
#             w = int(x_max - x_min) + 2 * offset
#             h = int(y_max - y_min) + 2 * offset
#
#             # Ensure coordinates are within frame boundaries
#             x = max(0, x)
#             y = max(0, y)
#             w = min(w, frame.shape[1] - x)
#             h = min(h, frame.shape[0] - y)
#
#             if w > 0 and h > 0:  # Only process valid regions
#                 try:
#                     # Crop and process hand image
#                     imgCrop = frame[y:y + h, x:x + w]
#
#                     # Create white background
#                     imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#
#                     # Maintain aspect ratio
#                     aspectRatio = h / w
#                     if aspectRatio > 1:
#                         k = imgSize / h
#                         wCal = int(k * w)
#                         imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                         wGap = (imgSize - wCal) // 2
#                         imgWhite[:, wGap:wCal + wGap] = imgResize
#                     else:
#                         k = imgSize / w
#                         hCal = int(k * h)
#                         imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                         hGap = (imgSize - hCal) // 2
#                         imgWhite[hGap:hCal + hGap, :] = imgResize
#
#                     # Show processed images
#                     cv2.imshow("Hand Crop", imgCrop)
#                     cv2.imshow("Processed Hand", imgWhite)
#
#                 except Exception as e:
#                     print(f"Image processing error: {e}")
#                     imgWhite = None
#
#             # Draw landmarks on original frame
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#     # Display instructions
#     cv2.putText(frame, f"Images Saved: {counter}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     cv2.putText(frame, "Press 'S' to Save | 'Q' to Quit", (10, 60),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#     cv2.imshow("Hand Sign Collector", frame)
#
#     # Key handling
#     key = cv2.waitKey(10)
#     if key == ord('q') or key == ord('Q'):
#         break
#     elif key == ord('s') or key == ord('S'):
#         if imgWhite is not None and hand_detected:
#             try:
#                 cv2.imwrite(f"{folder}/Image_{counter}.jpg", imgWhite)
#                 print(f"Successfully saved image {counter}")
#                 counter += 1
#             except Exception as e:
#                 print(f"Failed to save image: {e}")
#         else:
#             print("Can't save - no hand detected!")
#
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# import os
# import numpy as np
# import sys  # safer exit method
#
# # Ask user for label
# label = input("Enter the label for the dataset (e.g., A, B, C): ").upper()
# folder = f"data/{label}"
# os.makedirs(folder, exist_ok=True)
#
# # Initialize MediaPipe
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )
# mp_draw = mp.solutions.drawing_utils
#
# # Configuration
# counter = 0
# imgSize = 300
# offset = 20
#
# # Initialize cap safely
# cap = cv2.VideoCapture(0)
# if not cap or not cap.isOpened():
#     print("Error: Could not access the webcam.")
#     sys.exit(1)  # use sys.exit instead of exit()
#
# try:
#     while True:
#         success, frame = cap.read()
#         if not success or frame is None:
#             print("Ignoring empty camera frame.")
#             continue
#
#         frame = cv2.flip(frame, 1)
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(img_rgb)
#
#         imgWhite = None
#         hand_detected = False
#
#         if results.multi_hand_landmarks:
#             hand_detected = True
#             for hand_landmarks in results.multi_hand_landmarks:
#                 x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
#                 y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
#                 x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
#                 y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
#
#                 x = int(x_min) - offset
#                 y = int(y_min) - offset
#                 w = int(x_max - x_min) + 2 * offset
#                 h = int(y_max - y_min) + 2 * offset
#
#                 x = max(0, x)
#                 y = max(0, y)
#                 w = min(w, frame.shape[1] - x)
#                 h = min(h, frame.shape[0] - y)
#
#                 if w > 0 and h > 0:
#                     try:
#                         imgCrop = frame[y:y + h, x:x + w]
#                         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#
#                         aspectRatio = h / w
#                         if aspectRatio > 1:
#                             k = imgSize / h
#                             wCal = int(k * w)
#                             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                             wGap = (imgSize - wCal) // 2
#                             imgWhite[:, wGap:wGap + wCal] = imgResize
#                         else:
#                             k = imgSize / w
#                             hCal = int(k * h)
#                             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                             hGap = (imgSize - hCal) // 2
#                             imgWhite[hGap:hGap + hCal, :] = imgResize
#
#                         # Display
#                         cv2.imshow("Cropped", imgCrop)
#                         cv2.imshow("Processed", imgWhite)
#
#                     except Exception as e:
#                         print(f"Processing error: {e}")
#                         imgWhite = None
#
#                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#         # Instructions
#         cv2.putText(frame, f"Images Saved: {counter}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         cv2.putText(frame, "Press 'S' to Save | 'Q' to Quit", (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         cv2.imshow("Collector", frame)
#
#         key = cv2.waitKey(10)
#         if key == ord('q') or key == ord('Q'):
#             break
#         elif key == ord('s') or key == ord('S'):
#             if imgWhite is not None and hand_detected:
#                 file_path = f"{folder}/Image_{counter}.jpg"
#                 cv2.imwrite(file_path, imgWhite)
#                 print(f"Saved: {file_path}")
#                 counter += 1
#             else:
#                 print("No hand detected or image not processed.")
# finally:
#     # Ensure cleanup
#     if cap:
#         cap.release()
#     cv2.destroyAllWindows()





import cv2
import mediapipe as mp
import os
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Ask user to enter sign label
sign_label = input("Enter the sign label (e.g., 'A', 'B', 'C'): ").strip().upper()
save_dir = os.path.join('data', sign_label)
os.makedirs(save_dir, exist_ok=True)

# Resume from existing count
existing_images = [name for name in os.listdir(save_dir) if name.endswith('.jpg')]
count = len(existing_images)

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
img_size = 300

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb = np.ascontiguousarray(img_rgb)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]

            x, y = int(x_min), int(y_min)
            w, h = int(x_max - x_min), int(y_max - y_min)

            # Ensure safe crop boundaries
            x, y = max(0, x), max(0, y)
            cropped_hand = frame[y:y + h, x:x + w]

            if cropped_hand.size == 0:
                continue  # Skip if crop failed

            cropped_hand_resized = cv2.resize(cropped_hand, (img_size, img_size))

            save_path = os.path.join(save_dir, f'image_{count}.jpg')
            cv2.imwrite(save_path, cropped_hand_resized)
            count += 1

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Saved: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Data Collection', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



