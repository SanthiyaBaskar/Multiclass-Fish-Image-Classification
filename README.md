🐠 FishVision - Smart Fish Species Classifier

🧩 Problem Statement
- The task is to build a web-based image classification system that can identify different fish species from the given multiclass dataset.
- The application should be able to predict the correct species when a fish image is uploaded and display an alert when a non-fish image is provided.

✅ Solution
- A Convolutional Neural Network (CNN) model was developed and trained on the provided fish dataset containing 8 different species.
- The trained model was integrated with a Streamlit web interface, allowing users to upload an image and instantly view the predicted fish species.
- Confidence-based filtering was implemented — if the prediction confidence is below a threshold, the - system identifies the input as “Not a Fish.”
This ensures reliable classification and accurate detection for incorrect inputs.

🚀 Features
- Classifies 8 unique fish species from images
- Detects and alerts for non-fish images
- Clean, ocean-themed user interface
- Real-time prediction using CNN

⚙️ Technologies Used
- Python
- TensorFlow / Keras
- Streamlit
- NumPy & Pillow

▶️ How to Run
- pip install -r requirements.txt
- python train_model.py
- streamlit run app.py

Upload any fish image 🐟 → Get predicted species 🎯
Upload a random image 🚫 → Get “Not a Fish” warning ⚠️

👩‍💻 Developed by Santhiya Baskar

📘 GUVI Mini Project – Assignment 5
