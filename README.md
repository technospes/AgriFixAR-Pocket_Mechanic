# 🚜 AgriFix AI

### Intelligent AI Repair Assistant for Agricultural Machinery

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Flutter](https://img.shields.io/badge/flutter-3.x-blue)
![Status](https://img.shields.io/badge/status-active-success)

AgriFix AI is a **multimodal AI-powered repair assistant** that helps farmers diagnose and fix agricultural machinery using **voice, images, and video**.

Instead of searching through hundreds of pages of repair manuals, farmers can simply **describe the problem or record a short video**, and AgriFix provides **step-by-step repair guidance with AI verification**.

---

![Demo](Demo_Images/Home_Image.png)

---

# 📚 Table of Contents

* [About the Project](#about-the-project)
* [Key Features](#key-features)
* [Tech Stack & Architecture](#tech-stack--architecture)
* [Getting Started (Local Installation)](#getting-started-local-installation)
* [Usage / API Documentation](#usage--api-documentation)
* [Folder Structure](#folder-structure)
* [Roadmap](#roadmap)
* [Contributing & License](#contributing--license)

---

# 📖 About the Project

Agricultural machinery failures often occur in **rural or remote areas** where access to expert mechanics is limited.

Farmers typically face several challenges:

* Repair manuals are **complex and difficult to interpret**
* Troubleshooting requires **technical knowledge**
* Professional mechanics may take **hours or days to arrive**
* Equipment downtime leads to **lost productivity**

AgriFix AI solves this problem by converting **static machine manuals into an interactive AI repair assistant**.

The system uses:

* **Speech recognition**
* **Computer vision**
* **Retrieval-Augmented Generation (RAG)**
* **Large Language Models**

to guide farmers through machine repair **step-by-step**, while also verifying the repair visually.

The goal is to **democratize technical repair knowledge using AI**.

---

# ✨ Key Features

### 🎤 Voice-Based Problem Diagnosis

Farmers can simply describe the issue:

> "My tractor is not starting and making clicking sounds."

The system automatically:

1. Converts speech → text
2. Detects the machine type
3. Searches repair manuals
4. Generates repair instructions

---

### 🎥 Multimodal Machine Detection

AgriFix analyzes **video frames and images** to identify machinery such as:

* tractors
* irrigation pumps
* threshers
* motors
* power tillers

using a lightweight **MobileCLIP vision model**.

---

### 📚 RAG-Powered Knowledge System

Technical manuals are converted into a searchable AI knowledge base.

Pipeline:

```
PDF Manuals
   ↓
Text Chunking
   ↓
Embedding Generation
   ↓
ChromaDB Vector Store
   ↓
Semantic Retrieval
   ↓
Gemini LLM
   ↓
Repair Instructions
```

---

### 👁️ Visual Repair Verification

After performing a repair step, users can upload an image.

AgriFix verifies whether the repair was done correctly.

Example:

```
Step: Reconnect the battery terminal
```

Result:

```
✓ Correct connection detected
Confidence: 93%
```

---

### ⚡ LLM Cost Optimization

To reduce API cost and improve performance:

* semantic response caching
* Gemini fallback only when necessary
* per-IP rate limiting
* LLM timeout protection

This reduces Gemini API usage by **~60-70%**.

---

### 🔒 Production-Grade Security Layer

The backend includes several protections:

| Protection              | Purpose                 |
| ----------------------- | ----------------------- |
| Rate limiting           | Prevent API abuse       |
| Gemini credit guard     | Prevent token draining  |
| Upload validation       | Prevent malicious files |
| Prompt injection filter | Protect the LLM         |
| API key authentication  | Secure endpoints        |

---

# 🧠 Tech Stack & Architecture

## 📱 Frontend

**Flutter**

Used for building the mobile application.

Capabilities:

* camera video capture
* audio recording
* multilingual UI
* real-time status updates

---

## ⚙️ Backend

**Python FastAPI**

Chosen for:

* async high-throughput APIs
* efficient file streaming
* modern Python ecosystem

Responsibilities:

* processing uploaded media
* orchestrating AI inference
* enforcing security rules
* managing RAG pipeline

---

## 🧠 AI / ML

| Technology | Purpose                     |
| ---------- | --------------------------- |
| Gemini LLM | Diagnosis reasoning         |
| Whisper    | Speech → text transcription |
| MobileCLIP | Machine detection           |
| ChromaDB   | Vector database             |
| LangChain  | RAG orchestration           |

---

## 🗄 Database

**ChromaDB**

Stores vector embeddings for:

* repair manuals
* troubleshooting guides
* semantic search queries

This allows fast retrieval of relevant repair instructions.

---

## 🔐 Security Layer

Custom `security.py` module provides:

* rate limiting
* upload validation
* duration checks
* API key authentication
* prompt injection detection

---

# 🛠 Getting Started (Local Installation)

## Prerequisites

Make sure the following are installed:

* Python **3.10+**
* Flutter **3.x**
* Git
* FFmpeg (for audio/video processing)

---

## Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AgriFix.git
cd AgriFix
```

---

## Backend Setup

Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Flutter Setup

```bash
cd agrifix_app
flutter pub get
```

---

# 🔑 Environment Variables

Create a `.env` file in the backend root.

| Variable               | Description                  |
| ---------------------- | ---------------------------- |
| GEMINI_API_KEY         | Google Gemini API key        |
| APP_SECRET_KEY         | Server authentication key    |
| VIDEO_MAX_MB           | Maximum allowed video upload |
| AUDIO_MAX_MB           | Maximum allowed audio upload |
| VIDEO_MAX_SECONDS      | Maximum video duration       |
| AUDIO_MAX_SECONDS      | Maximum audio duration       |
| GEMINI_TIMEOUT_SECONDS | Timeout for LLM calls        |
| GEMINI_HOURLY_LIMIT    | Rate limit per IP            |

Example:

```env
GEMINI_API_KEY=your_api_key_here
APP_SECRET_KEY=your_secret_here

VIDEO_MAX_MB=20
AUDIO_MAX_MB=5

VIDEO_MAX_SECONDS=20
AUDIO_MAX_SECONDS=20

GEMINI_TIMEOUT_SECONDS=60
GEMINI_HOURLY_LIMIT=10
```

---

# ▶️ Running the Project

## Run Backend

```bash
uvicorn main:app --host 0.0.0.0 --port 7680 --reload
```

API docs available at:

```
http://localhost:7680/docs
```

---

## Run Flutter App

```bash
cd agrifix_app
flutter run
```

---

# 📡 Usage / API Documentation

## Diagnose a Machine Issue

Endpoint:

```
POST /diagnose/stream
```

Input:

* audio description
* video of machine

Example response:

```json
{
  "machine": "tractor",
  "diagnosis": "Starter motor failure",
  "steps": [
    "Check battery voltage",
    "Inspect starter connections",
    "Clean corroded terminals"
  ]
}
```

---

## Verify a Repair Step

Endpoint:

```
POST /verify_step
```

Input:

* image of repaired component
* repair step description

Example response:

```json
{
  "status": "pass",
  "confidence": 0.92,
  "feedback": "Battery terminal appears properly secured."
}
```

---

# 📁 Folder Structure

```
AgriFix_Workspace
│
├── agrifix_app
│   ├── lib
│   ├── android
│   └── ios
│
├── AgriFixAR_Python_Client
│   ├── agent
│   │   ├── repair_agent.py
│   │   └── session_manager.py
│   │
│   ├── services
│   │   ├── diagnosis_service.py
│   │   ├── machine_detection_service.py
│   │   ├── transcription_service.py
│   │   └── verification_service.py
│   │
│   ├── utils
│   │   └── helpers.py
│   │
│   ├── security.py
│   ├── main.py
│   └── requirements.txt
│
├── Demo_Images
│
└── README.md
```

---

# 🗺 Roadmap

Planned improvements:

* AR repair guidance using **Unity**
* offline AI inference
* expanded machine categories
* multilingual voice interaction
* predictive maintenance alerts

---

# 🤝 Contributing

Contributions are welcome.

1️⃣ Fork the repository

2️⃣ Create a branch

```bash
git checkout -b feature/my-feature
```

3️⃣ Commit changes

```bash
git commit -m "Add new feature"
```

4️⃣ Push to branch

```bash
git push origin feature/my-feature
```

5️⃣ Open a Pull Request.

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Ayush Shukla**

B.Tech Computer Science
AI / Computer Vision / Full Stack Development

GitHub
[https://github.com/YOUR_USERNAME](https://github.com/YOUR_USERNAME)

LinkedIn
[https://www.linkedin.com/in/ayushshukla-ar/](https://www.linkedin.com/in/ayushshukla-ar/)

---

⭐ If you find this project interesting, consider **starring the repository**!

```

---
