# Real-Time-Object-Tracking-and-Human-Pose-Detection-using-Ultralytics-YOLO-Pose-Estimation


📌 Pose Estimation Project
📖 Overview : This project implements human pose estimation using computer vision techniques to detect key body joints (like shoulders, elbows, knees, etc.) from images or videos.

🛠️ Tech Stack
Python 🐍
OpenCV
NumPy
MediaPipe / OpenPose / TensorFlow / PyTorch 
Matplotlib (for visualization)

📂 Project Structure

pose-estimation/
│
├── data/                # Input images/videos
├── output/              # Output results
├── models/              # Pre-trained models
├── src/                 # Source code
│   ├── main.py
│   ├── utils.py
│   └── pose_detector.py
│
├── requirements.txt
└── README.md


Installation
 1️⃣ Clone the Repository
 2️⃣ Create Virtual Environment
 3️⃣ Install Dependencies

🧠 Model Details

- Model Used: (e.g., MediaPipe Pose / OpenPose / YOLO Pose)
- Keypoints Detected: 17 / 33 body landmarks
- Framework: (TensorFlow / PyTorch)


🔍 Working Steps (Pipeline)

1.Input Capture
Read image/video/webcam stream

2.Preprocessing
Resize image
Convert color (BGR → RGB)

3.Pose Detection
Load pre-trained model
Detect body keypoints

4.Post-processing
Draw skeleton on frame
Extract coordinates

5.Output
Display result
Save output image/video



