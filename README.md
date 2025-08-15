 # HelpBubble - E-commerce Customer Support Chatbot

## Overview
Built for researching Natural Language Processing. An intelligent customer support chatbot powered by transformer-based question answering models. Built with FastAPI backend and React frontend.

### Features
- Transformer-based QA: Uses DistilBERT fine-tuned on SQuAD dataset
- Real-time Chat Interface: Modern React.js chat UI with styled-components
- Context-Aware Responses: Automatically matches questions to relavent-commerce contexts
- Performance Metrics: Tracks confidence scores, response times, and accuracy
- RESTful API: FastAPI backend with automatic documentation
- Production Ready: Configured for deployment on Render and Vercel

### Backend
- FastAPI - Modern Python web framework
- Hugging Face Transformers - Pre-trained BERT/DistilBERT models
- PyTorch - Deep learning framework
- Uvicorn - ASGI server

### Frontend
- React.js - Component-based UI library
- Styled Components - CSS-in-JS styling (**Likely updated to react jsx with seperate components in future**)
- Axios - HTTP client for API calls

### ML/AI
- DistilBERT - Lightweight transformer model
- SQuAD Dataset - Question answering benchmark
- Hugging Face Hub - Model hosting and management


## 📁 Project Structure
TO BE ADDED


## 🔧 Installation & Setup

### Recomended Prerequisites
 - Python 3.11 (I used 3.11.9). I found 3.12 & 3.13 to not be compatiable with the used pytorch 2.1.1. 
 - Node.js 14+ should be fine. (Used v22.16)

### Backend Setup
1. Create your repo. Then Clone and navigate to project backend diriectory
```bash
git clone your-repo-url
cd help-bubble/backend

```
2. Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

```
3. Install dependencies
```bash
pip install -r requirements.txt

```
4. Create dataset
```bash
python data_preparation.py

```
5. Run the API server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000

```
- The API will be available at http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Frontend Setup
- Ensure you open another terminal

1. From the root navigate to frontend directory
```bash
cd frontend/help-bubble-ui

```
2. Install dependencies
```bash
npm install

```
3. Start development server
```bash
npm start

```
- The frontend will be available at http://localhost:3000

### Model Evaluation

Run the evaluation script from **backend** to test model performance:
```bash
python evaluate.py
```


