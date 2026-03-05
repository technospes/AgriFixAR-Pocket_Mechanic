# 🚜 AgriFix AI  
### Intelligent AI Repair Assistant for Agricultural Machinery

AgriFix AI is a multimodal AI system that helps farmers diagnose and repair agricultural machinery using **voice, video, and images**.

The system combines **computer vision, speech recognition, Retrieval-Augmented Generation (RAG), and large language models** to transform traditional repair manuals into an interactive troubleshooting assistant.

Instead of searching through hundreds of pages of manuals, farmers can simply **record a short video or describe the problem in voice**, and AgriFix provides **step-by-step repair guidance with visual verification**.

---

# 🌍 Problem

Agricultural machinery failures often occur in remote areas where:

• mechanics are unavailable  
• repair manuals are complex  
• technical knowledge is limited  
• diagnosis is slow and expensive  

Farmers often lose **hours or days of productivity** waiting for help.

AgriFix aims to **democratize technical repair knowledge using AI**.

---

# 💡 Solution

AgriFix AI converts static service manuals into an **interactive repair assistant**.

Farmers can:

1️⃣ Describe the problem in **voice or text**  
2️⃣ Record a **short video of the machine**  
3️⃣ Receive **AI-generated repair steps**  
4️⃣ Verify each repair step using **computer vision**

This creates a **guided repair workflow**, similar to having an expert mechanic present.

---

# 🚀 Key Features

## 🔎 AI-Powered Machine Diagnosis

AgriFix analyzes:

• machine video frames  
• user voice description  
• technical manuals  

to determine the most likely issue.

It retrieves relevant instructions using **semantic search over technical manuals**.

---

## 🎤 Voice-First Interaction

Farmers can simply say things like:

> "My tractor is not starting and making a clicking sound."

The system automatically:

1. Converts speech → text  
2. Detects the machine type  
3. Retrieves relevant repair steps  
4. Generates step-by-step guidance

---

## 🎥 Multimodal Machine Detection

AgriFix uses **computer vision models** to identify the machine from video frames.

Supported machine categories include:

• tractors  
• irrigation pumps  
• threshers  
• power tillers  
• agricultural motors  

---

## 🧠 RAG-Powered Knowledge System

AgriFix uses **Retrieval-Augmented Generation (RAG)**.

Manuals are:

• parsed from PDFs  
• split into semantic chunks  
• embedded into vectors  
• stored inside **ChromaDB**

During diagnosis:

User Problem
↓
Semantic Search
↓
Relevant Manual Sections
↓
Gemini LLM
↓
Repair Instructions

text

---

## 👁️ Visual Repair Verification

After a repair step is performed, the user can upload a photo.

AgriFix checks if the repair was done correctly.

Example:

Step: Tighten the oil filter

text

User uploads image →

AI Result:
✓ Correct installation detected
Confidence: 94%

text

---

## ⚡ AI Cost Optimization

To prevent excessive LLM usage and reduce API cost:

AgriFix implements:

• **semantic response caching**  
• **Gemini fallback only when needed**  
• **per-IP rate limiting**  
• **LLM call timeouts**

This reduces Gemini usage by **60–70%**.

---

## 🔒 Production-Grade Security

AgriFix includes multiple backend protection layers.

Security features include:

### Rate Limiting
Limits requests per IP to prevent abuse.

### Gemini Credit Guard
Prevents attackers from draining API credits.

### Upload Validation
Server-side validation checks:

• file size  
• file type  
• file duration  

before processing.

### Prompt Injection Protection
User prompts are scanned for malicious patterns before reaching the LLM.

### API Key Protection
All secrets are stored using environment variables.

---

# 🧠 AI Architecture

Flutter Mobile App
│
▼
FastAPI Backend
│
▼
Speech → Text (Whisper)
│
▼
Machine Detection (MobileCLIP)
│
▼
RAG Knowledge Retrieval (ChromaDB)
│
▼
Gemini LLM Diagnosis
│
▼
Step-by-Step Repair Instructions

text

---

# 🧰 Tech Stack

## Frontend

Flutter (Android)

Features:

• video capture  
• audio recording  
• multilingual UI  
• real-time progress feedback  

---

## Backend

FastAPI (Python)

Server responsibilities:

• media processing  
• AI inference orchestration  
• security enforcement  
• RAG pipeline management

---

## AI Models

### LLM
Google Gemini

Used for:

• diagnosis reasoning  
• repair instruction generation  

---

### Speech Recognition

Whisper

Used for:

• farmer voice input  
• machine problem description

---

### Vision Model

MobileCLIP

Used for:

• machine type detection  
• frame classification

---

### Vector Database

ChromaDB

Stores:

• manual embeddings  
• semantic search index

---

# 📂 Project Structure

AgriFix_Workspace
│
├── agrifix_app # Flutter mobile application
│
├── AgriFixAR_Python_Client # FastAPI backend
│ │
│ ├── agent
│ │ ├── repair_agent.py
│ │ ├── session_manager.py
│ │ └── safety_rules.py
│ │
│ ├── services
│ │ ├── diagnosis_service.py
│ │ ├── machine_detection_service.py
│ │ ├── transcription_service.py
│ │ └── verification_service.py
│ │
│ ├── utils
│ │ ├── helpers.py
│ │ └── machine_registry.py
│ │
│ ├── security.py # security & rate limiting
│ ├── main.py # API server
│ └── requirements.txt
│
├── Demo_Images
│
└── README.md

text

---

# ⚙️ Installation

## Clone the Repository

git clone https://github.com/YOUR_USERNAME/AgriFix.git
cd AgriFix

text

---

## Create Python Environment

python -m venv venv
venv\Scripts\activate

text

---

## Install Dependencies

pip install -r requirements.txt

text

---

# 🔑 Environment Variables

Create a `.env` file.

GEMINI_API_KEY=your_key_here

VIDEO_MAX_MB=20
AUDIO_MAX_MB=5

VIDEO_MAX_SECONDS=20
AUDIO_MAX_SECONDS=20

GEMINI_TIMEOUT_SECONDS=60
GEMINI_HOURLY_LIMIT=10

APP_SECRET_KEY=your_generated_secret

text

---

# ▶️ Running the Backend

uvicorn main:app --host 0.0.0.0 --port 7680 --reload

text

API documentation available at:

http://localhost:7680/docs

text

---

# 📱 Running the Flutter App

cd agrifix_app
flutter pub get
flutter run

text

---

# 📊 Performance Optimizations

AgriFix includes several optimizations:

• semantic caching of LLM responses  
• selective Gemini fallback  
• frame sampling for video analysis  
• asynchronous API execution  
• persistent vector database  

These optimizations significantly reduce:

• API cost  
• latency  
• compute load  

---

# ⚠️ Known Limitations

Current limitations include:

• dependent on internet connectivity  
• limited machine categories  
• vision verification accuracy depends on image quality  

---

# 🗺️ Roadmap

Planned future features:

### AR Repair Guidance
Using Unity to overlay repair instructions on real machines.

### Offline Mode
Local LLM inference for rural areas without internet.

### Predictive Maintenance
Detect machine issues before failure.

### Sensor Integration
Integrate IoT data from tractors and pumps.

---

# 🤝 Contributing

Contributions are welcome.

1️⃣ Fork the repository  
2️⃣ Create a feature branch  

git checkout -b feature/new-feature

text

3️⃣ Commit your changes  

git commit -m "Add new feature"

text

4️⃣ Push the branch  

git push origin feature/new-feature

text

5️⃣ Open a Pull Request.

---

# 📜 License

MIT License.

---

# 👨‍💻 Author

**Ayush Shukla**

B.Tech Computer Science  
AI / Computer Vision / Systems Development

GitHub  
[https://github.com/technospes](https://github.com/technospes)  
LinkedIn  
[https://www.linkedin.com/in/ayushshukla-ar/](https://www.linkedin.com/in/ayushshukla-ar/)

---

# ⭐ If you like this project

Consider starring the repository to support development.
