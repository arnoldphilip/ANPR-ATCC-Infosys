# ANPR-ATCC: Advanced Automatic Number Plate Recognition & Traffic Classification System

## Project Overview

**ANPR-ATCC** is a unified, intelligent traffic monitoring system that combines:

- **Automatic Number Plate Recognition (ANPR):** Real-time detection and OCR-based license plate extraction from images, videos, and camera feeds.  
- **Automatic Traffic Classification and Counting (ATCC):** Multi-class vehicle detection and traffic flow analysis using deep learning.

The project integrates **YOLO-based object detection** with **Tesseract OCR**, providing scalable analytics for **smart cities, law enforcement, parking systems, and academic research**.

---

## 🚀 Key Features

- **High-Accuracy License Plate Recognition** using YOLO + Tesseract OCR.  
- **Vehicle Classification & Counting** (cars, trucks, bikes, etc.) for traffic analytics.  
- **Supports Multiple Inputs:** Images, video files, or live webcam streams.  
- **Automatic Data Storage:** SQLite + JSON logging for every detection cycle.  
- **Interactive Streamlit Dashboard:** Adjustable thresholds, live previews, and data management.  
- **Cross-Platform Support:** Works on Windows, Linux, and macOS.  
- **Modular Architecture:** Independent ANPR and ATCC modules for easy scaling.  

---

## 🧠 Technology Stack

| Component | Purpose | Reason |
|------------|----------|--------|
| **YOLOv10 & YOLOv11n (Ultralytics)** | Vehicle & plate detection | Real-time, high-accuracy models |
| **Tesseract OCR** | Text extraction | Lightweight, multilingual OCR |
| **Streamlit** | UI framework | Interactive web-based dashboard |
| **SQLite** | Local database | Serverless, lightweight data storage |
| **OpenCV** | Image/video processing | Efficient frame handling and preprocessing |
| **Pandas, Matplotlib** | Analytics & visualization | Simplifies insights and plotting |
| **Python (3.11+)** | Core language | Modern syntax, rich ML ecosystem |

---

## 🧩 Use Cases

- **Smart City Traffic Systems**
- **Automated Parking Management**
- **Law Enforcement & Violation Detection**
- **Research & Academic Projects**
- **Vehicle Flow Monitoring**

---

## ⚙️ Installation Guide (Windows)

### ✅ Step 1: Install Python 3.11+

Download from [python.org/downloads](https://www.python.org/downloads/)  
> During installation, enable the option **“Add Python to PATH”**.

---

### ✅ Step 2: Create & Activate Virtual Environment
python -m venv anpr_env
anpr_env\Scripts\activate

### ✅ Step 3: Clone the Repository
git clone https://github.com/nehakumari2003/ANPR-ATCC.git
cd ANPR-ATCC

### ✅ Step 4: Install Dependencies
pip install --upgrade pip
pip install streamlit opencv-python-headless ultralytics numpy pandas matplotlib pillow pytesseract


(Optional for local debugging with GUI windows)

pip install opencv-python

### ✅ Step 5: Install Tesseract OCR

Tesseract is required for number plate text recognition.

Download from Tesseract for Windows

Install to:
C:\Program Files\Tesseract-OCR

Add the path to System Environment Variables:

C:\Program Files\Tesseract-OCR\


Verify installation:

tesseract --version

### ✅ Step 6: Verify Project Structure
ANPR-ATCC/
│
├── app.py
├── weights/
│   ├── best.pt            # License Plate YOLO model
│   ├── yolo11n.pt         # Vehicle YOLO model
│
├── json/
├── traffic_analysis.db
├── licensePlatesDatabase.db
└── requirements.txt

### ✅ Step 7: Run the Application
streamlit run app.py


Once it starts, open the link displayed (usually http://localhost:8501) in your browser.

⚡ GPU Acceleration (Optional)

If you have a CUDA-capable GPU, install compatible Torch and Ultralytics builds:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics

🧩 Troubleshooting
Issue	Solution
No module named 'ultralytics'	pip install ultralytics
pytesseract.pytesseract.TesseractNotFoundError	Add Tesseract path to PATH
YOLO model not found	Verify weights/best.pt and weights/yolo11n.pt exist
Streamlit not opening	Manually open http://localhost:8501
🧾 Detailed Usage
▶️ ANPR Mode

Detects license plates from input image/video/webcam.

OCR extracts and displays text with bounding boxes.

All outputs saved to licensePlatesDatabase.db and json/ logs.

▶️ ATCC Mode

Detects and classifies vehicles (bike, car, bus, truck).

Displays total counts, congestion level, and traffic density analytics.

Logs stored in traffic_analysis.db.

▶️ Data Visualization

View or export detection history via Streamlit dashboard.

Filter by date/time, type, or detection confidence.

🧱 Contributing

Fork and clone the repository.

Create a new branch for your feature.

Follow PEP8 standards and document new modules.

Submit a Pull Request describing your contribution.

🔐 Security

All detections are processed locally (no cloud uploads).

Databases are stored on your device for full privacy.

For production, consider encrypting SQLite data.

🛠️ Future Roadmap

🔤 Multilingual OCR (PaddleOCR, EasyOCR)

🌐 Cloud dashboard integration

💡 Real-time alert system

🧩 Edge deployment (Jetson Nano, Pi)

🖥️ Enhanced UI/UX with dark mode

🧠 Transformer-based detection models

📚 References

YOLOv10 - THU MIG

Ultralytics Documentation

Tesseract OCR Wiki

Streamlit Docs

OpenCV

SQLite Browser

🪪 License

Licensed under the MIT License
 © 2025 Vidzai Digital.

🌐 Vision

Empowering smarter cities through real-time computer vision and AI-driven traffic intelligence.

📦 requirements.txt (for reference)
streamlit
opencv-python-headless
ultralytics
numpy
pandas
matplotlib
pillow
pytesseract
