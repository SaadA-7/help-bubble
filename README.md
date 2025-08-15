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
2. Create and activate virtual environment
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
Ensure venv script is active in backend.
Run the evaluation script to test model performance:
```bash
python evaluate.py
```
## Performance Metrics

Current model performance on  e-commerce dataset:

- **Exact Match Score**: > 70%
- **F1 Score**: > 75%
- **Average Response Time**: < 0.092s
- **Success Rate**: 100%


## 🌐 API Endpoints
Core Endpoints
- POST /ask - Submit a question and get an answer
- GET /health - Check API health status
- GET /contexts - List available knowledge base contexts
- POST /test-context - Test with custom context and question

Example API Usage:
```bash
import requests

# Ask a question
response = requests.post("http://localhost:8000/ask", json={
  "question": "How long do I have to return an item?",
    user_id": "user123"
})

print(response.json())
# {
#   "answer": "30 days",
#   "confidence": 0.98,
#   "context_used": "Our return policy allows...",
#   "response_time": 0.15,
#   "timestamp": "2024-01-15T10:30:00"
# }
```

## 🚀 Deployment

TODO

## 📊 Knowledge Base
The system includes pre-built knowledge for:

- **Returns & Refunds**: 30-day return policy, conditions, processing times
- **Shipping**: Free shipping thresholds, delivery times, tracking
- **Payment**: Accepted methods, security, verification
- **Product Warranties**: Coverage periods, claim process
- **Account Management**: Registration, password reset, benefits
- **Promotions: Discounts**, student deals, seasonal sales

## Technical Highlights
- **Modern ML Pipeline**: Transformer models with Hugging Face integration
- **Full-Stack Development**: React frontend + FastAPI backend
- **Production Architecture**: RESTful APIs, error handling, monitoring
- **Performance Optimization**: Model evaluation, response time tracking
- **Cloud Deployment**: Containerized deployment on modern platforms

## Buisness Value
- **Customer Support Automation**: Reduces support ticket volume
- **24/7 Availability**: Instant responses to common questions
- **Scalable Solution**: Handles multiple concurrent users
- **Analytics Ready**: Tracks usage patterns and performance metrics
- **Cost Effective**: Reduces human support agent workload

## 🔍 Testing the System

**Example Questions to Try**

**Returns**: "How long do I have to return an item?"
**Shipping**: "When is shipping free?"
**Payment**: "What payment methods do you accept?"
**Warranty**: "What warranty comes with electronics?"
**Account**: "How do I reset my password?"
**Promotions**: "Do you offer student discounts?"

**Expected Performance**
- **Response time**: < 200ms for most queries
- **Availability**: 99.9% uptime in production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

## 📝 License
Copyright © 2025 Saad Ahmad. All rights reserved.
Licensed under the Help-Bubble Custom License (see LICENSE file for details).

## NLP & ML Project
This project demonstrates practical application of transformer-based NLP models in a real-world customer support scenario, showcasing both technical depth and business value.




