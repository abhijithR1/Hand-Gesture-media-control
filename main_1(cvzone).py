import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pyautogui

cap = cv2.VideoCapture(0 )
cap.set(3, 640)  # Set resolution to 640x480 to speed up processing
cap.set(4, 480)

detector = HandDetector(maxHands=1, detectionCon=0.7)  # Adjusted detection confidence to speed up hand detection
classifier = Classifier('Model/keras_model.h5', 'Model/labels.txt')

offset = 20
imgSize = 300
counter = 0

labels = ["free_hand", "right", "left", "v_up", "v_down", "max", "min", "stop"]

control_delay = 3  # Increased delay to 3 seconds to avoid repeated actions
last_action_time = time.time() - control_delay

# Keep track of the last prediction index to avoid repeated actions
prev_index = None

# To smooth out predictions, store recent predictions
prediction_history = []
history_size = 10  # Length of prediction history

frame_count = 0
frame_skip = 2  # Process every 2nd frame for faster processing

while True:
    frame_count += 1
    success, img = cap.read()
    imgOutput = img.copy()

    if frame_count % frame_skip == 0:  # Skip frames to improve processing speed
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            # Ensure cropping doesn't go out of bounds
            imgCrop = img[max(0, y - offset): min(y + h + offset, img.shape[0]), 
                          max(0, x - offset): min(x + w + offset, img.shape[1])]

            if imgCrop.size == 0:
                print("Empty image crop. Skipping frame.")
                continue

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=True)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=True)

            # Add current prediction to the history
            prediction_history.append(index)
            if len(prediction_history) > history_size:
                prediction_history.pop(0)

            # Determine the most frequent prediction in the history
            final_index = max(set(prediction_history), key=prediction_history.count)

            # Update output display
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255),
                          cv2.FILLED)
            cv2.putText(imgOutput, labels[final_index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

            # Perform YouTube controls based on the final (smoothed) label
            current_time = time.time()

            if final_index != prev_index:  # Only act when the gesture changes
                if labels[final_index] == "v_up":
                    pyautogui.press("up")  # Press up arrow key for volume up
                    last_action_time = current_time
                elif labels[final_index] == "v_down":
                    pyautogui.press("down")  # Press down arrow key for volume down
                    last_action_time = current_time
                elif labels[final_index] == "free_hand":
                    pass  # Do nothing for the "free_hand" gesture
                elif labels[final_index] == "stop" and current_time - last_action_time >= control_delay:
                    pyautogui.press("space")  # Press spacebar to pause/play
                    last_action_time = current_time
                elif labels[final_index] == "max" and current_time - last_action_time >= control_delay:
                    pyautogui.press("f")  # Press f to enter full screen
                    last_action_time = current_time
                elif labels[final_index] == "min" and current_time - last_action_time >= control_delay:
                    pyautogui.press("esc")  # Press ESC to exit full screen
                    last_action_time = current_time
                elif labels[final_index] == "right" and current_time - last_action_time >= control_delay:
                    pyautogui.press("right")  # Press Right Arrow to skip 5 sec
                    last_action_time = current_time
                elif labels[final_index] == "left" and current_time - last_action_time >= control_delay:
                    pyautogui.press("left")  # Press Left Arrow to rewind 5 sec
                    last_action_time = current_time

            prev_index = final_index  # Update the previous index

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()