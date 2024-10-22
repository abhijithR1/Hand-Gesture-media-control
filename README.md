# Hand Gesture Media Control

## Overview

This project implements two models for hand gesture recognition to control media playback, such as volume adjustment, play/pause functionalities, rewind etc. The models utilize different approaches: one based on image classification using **cvzone** and the other using **mediapipe** for hand landmark detection.


## Models

### First Model (Using cvzone)
- **Hand Detection**: Utilizes the `HandDetector` from cvzone to detect one hand at a time.
- **Classification**: Employs a trained Keras model (`keras_model.h5`) to predict specific hand gestures based on cropped hand images.
- **Gesture History**: Smooths predictions by storing recent predictions and selecting the most common one.
- **Control Actions**: Maps predicted gestures to actions (e.g., increase/decrease volume, play/pause) using `pyautogui`.

**Challenges**:
- Classification is image-based and heavily relies on the accuracy of the classifier model, which may not always be optimal.
- Lacks dynamic finger tracking and detailed landmark analysis.

### Second Model (Using Mediapipe)
- **Hand Landmark Detection**: Uses Mediapipe to detect detailed hand landmarks for one hand.
- **Finger Counting**: Implements a custom function to count the number of extended fingers based on landmark positions.
- **Control Actions**: Maps the number of extended fingers to actions (e.g., pressing arrow keys or space for play/pause) using `pyautogui`.

**Strengths**:
- Focuses on the relative positions of key landmarks, allowing for dynamic and potentially more accurate finger-based gesture detection.
- Simpler implementation without the need for external model training, relying entirely on real-time finger positions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abhijithR1/Hand-Gesture-media-control.git
2. Navigate to the project directory:
   cd Hand-Gesture-media-control
3. Install the required packages:
   pip install -r requirements.txt
4. To run the cvzone model:
   python main_1(cvzone).py
5. To run the mediapipe model:
   python main_2(mediapipe).py

## Acknowledgments
- **cvzone** for hand detection and classification modules.
- **mediapipe** for efficient hand landmark detection.
- **pyautogui** for simulating keyboard actions.
