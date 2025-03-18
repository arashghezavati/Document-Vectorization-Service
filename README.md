# 📖 AI-Powered Document Processing & Chatbot Service

This project enables **intelligent document search and chatbot interaction** using **AI-powered embeddings and a fine-tuned LLM**.

## 🔹 How It Works:
✔ **Google Gemini AI** generates **vector embeddings** for document storage & retrieval.  
✔ **Fine-Tuned LLM (Mistral-7B/Phi-2)** is used for chatbot responses, ensuring **accurate and domain-specific answers**.  
✔ **Retrieval-Augmented Generation (RAG)** dynamically **retrieves relevant documents** before generating responses.

---

## 🚀 Features & Benefits

### 📖 AI-Powered Document Search (Google Gemini AI)
✅ **Smart Semantic Search** – Finds relevant documents **even without exact keyword matches**.  
✅ **Multi-Format Support** – Works with **PDF, DOCX, TXT, HTML**.  
✅ **Efficient Storage** – Uses **ChromaDB** to store document embeddings for fast retrieval.  
✅ **Google Gemini AI for Embeddings** – Converts document text into **vector embeddings** for accurate searching.

### 🤖 AI Chatbot with Fine-Tuned LLM (Mistral/Phi-2)
✅ **Fine-Tuned for Accuracy** – Unlike generic models, our chatbot **understands domain-specific knowledge**.  
✅ **Fewer Hallucinations** – Trained on **real-world business data**, reducing irrelevant responses.  
✅ **Flexible Model Usage** – Users can **train their own fine-tuned model** and **replace the API endpoint**.

### 📤 Document & Folder Management
✅ **Upload & organize documents** into **folders**.  
✅ **Automatic text extraction** from multiple file formats.  
✅ **Fast retrieval** with **vector-based search**.  
✅ **Manage stored data** – Delete, update, and query stored documents easily.

---

## 🧠 Fine-Tuning & AI Model Optimization

### 🔹 Fine-Tuning on Custom Data
We trained the model using **domain-specific documents** to improve response accuracy and reduce hallucinations. The fine-tuning process includes:  
✔ **Dataset Curation** – Preparing, cleaning, and structuring custom training data.  
✔ **Hyperparameter Optimization** – Adjusting batch size, learning rate, and training epochs.  
✔ **Efficient Training** – Using **LoRA (Low-Rank Adaptation)** to fine-tune **only essential model parameters**.

### 🔹 Retrieval-Augmented Generation (RAG)
✔ **Retrieves relevant documents dynamically** and **generates AI responses based on real-time data**.  
✔ **Ensures responses are accurate, factual, and context-aware**.

### 🔹 Model Distillation & Inference Optimization
✔ **Model Distillation** – Trained a **lighter model version** for **faster responses**.  
✔ **Inference Optimization** – Implemented **quantization techniques** to reduce memory usage.  
✔ **Batch Processing & Caching** – **Faster query processing** for improved efficiency.

---

## 🔧 Quick Start Guide

### 1️⃣ Clone & Setup
```bash
# Clone the repository
git clone https://github.com/arashghezavati/Document-Vectorization-Service.git
cd Document-Vectorization-Service

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r python-services/requirements.txt

2️⃣ Configure API Keys & Environment Variables
Create a .env file in the root directory:

ini
Copy
Edit
# Google Gemini for Document Embeddings
GOOGLE_GEMINI_API_KEY=your_gemini_api_key
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_DIMENSION=768

# Fine-Tuned LLM API (For Chatbot)
LLM_API_ENDPOINT=https://your-fine-tuned-model.com/api
LLM_API_KEY=your_fine_tuned_model_api_key

# ChromaDB Storage
CHROMADB_PATH=./vector-database/store
3️⃣ Train Your Own Fine-Tuned Model (Optional)
To fine-tune the LLM on your own dataset, use:

bash
Copy
Edit
cd fine-tuning
python fine_tune.py
Fine-tuned models will be saved in: ./fine-tuning/models/

After training, upload the fine-tuned model and update .env with the new API URL.

📢 AI Chatbot API
Start the FastAPI Server
bash
Copy
Edit
cd python-services
python run_server.py
API Base URL: http://localhost:8000
Swagger API Docs: http://localhost:8000/docs

Example API Request (Chatbot Query)
bash
Copy
Edit
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -d '{"query": "What is the CRS score required for Canada PR?", "mode": "strict"}'
🔧 Technical Stack
Backend: FastAPI, ChromaDB
Document Embeddings: Google Gemini API
Chatbot: Fine-Tuned Mistral-7B / Phi-2 (Custom API)
Retrieval-Augmented Generation (RAG) – Real-time document retrieval for AI responses.
Fine-Tuning Techniques: LoRA, Model Distillation, Quantization.
Document Processing: PyPDF2, python-docx, BeautifulSoup.
🤝 Contributing
✔ Fork the repository
✔ Create a feature branch (git checkout -b feature/new-feature)
✔ Commit your changes (git commit -m 'Added feature')
✔ Push to branch (git push origin feature/new-feature)
✔ Open a Pull Request

📩 Support
For issues, open a GitHub issue or contact the maintainers.
