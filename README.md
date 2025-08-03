# Dji-tello-control-body-gesture
🚁 A multi-mode drone navigation project using computer vision and machine learning

This project👨‍💻  enables users to control a drone using multiple input methods powered by Computer Vision and Artificial Intelligence. The system can detect and respond to hand gestures, face movements, voice commands, and body positioning, making drone navigation more interactive and intuitive.

 Features

    🖐 Hand Gesture Control – Control the drone with custom hand gestures trained on a deep learning model (hand_gesture_model_improve.h5).

    🙂 Face Detection Control – Use facial position and orientation to move the drone in specific directions.

    🎙 Voice Command Control (Planned) – Command the drone using simple voice instructions like take off, land, move forward, turn left, etc.

    🚶 Body Follow Mode – The drone can autonomously follow the user’s body movements and maintain a set distance.

    🛡 Safety Feature – Includes an emergency stop gesture or command to land the drone immediately.

🛠️ Technologies Used

    Programming Language: Python

    Libraries & Frameworks:

        OpenCV – Image Processing & Object Detection

        Mediapipe / TensorFlow / Keras – Gesture & Pose Recognition

        Pyttsx3 / SpeechRecognition – (For planned voice commands)

        DJI Tello SDK or DroneKit – Drone communication

        NumPy, Pandas – Data handling and processing

📦 Drone-Control
 ┣ 📂 custom Hand Gesture model
 ┃ ┗ hand_gesture_model_improve.h5
 ┣ 📂 datasets
 ┣ 📜 main_hand_gesture.py
 ┣ 📜 main_face_detection.py
 ┣ 📜 main_body_follow.py
 ┣ 📜 main_voice_command.py (planned)
 ┣ 📜 requirements.txt
 ┗ 📜 README.md

🔮 Future Enhancements

    Full integration of voice command control

    Improved gesture recognition accuracy

    Mobile application support for on-the-go control

    Advanced obstacle avoidance using depth sensors
