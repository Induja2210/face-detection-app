# face-detection-app

#  AI-Based Face Detection & Recognition Web App

##  Overview
This project is a full-stack AI web application that detects human faces and performs classification and recognition.  
Users can upload an image or capture it through a camera to analyze facial attributes such as age, gender, and emotion.

---

##  Features
- Face detection using MTCNN  
- Face recognition using FaceNet (InceptionResnetV1)  
- Age prediction using transformer-based model  
- Gender classification  
- Emotion detection  
- Image upload and real-time camera capture  
- Custom database for recognizing known individuals  

---

##  Tech Stack
- Python  
- Streamlit  
- PyTorch  
- OpenCV  
- NumPy  
- FaceNet (facenet-pytorch)  
- Hugging Face Transformers  

---

##  How It Works
1. User uploads or captures an image  
2. MTCNN detects faces in the image  
3. FaceNet generates embeddings for each face  
4. Models predict age, gender, and emotion  
5. Embeddings are compared with stored data for recognition  

---

##  How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

##  Output
<img width="1366" height="768" alt="{D59985BF-2219-4AA5-A7E1-AEB2EC86EB30}" src="https://github.com/user-attachments/assets/762b2e2d-a6fa-4b01-984f-74158071db81" />

##  Author
Induja B
