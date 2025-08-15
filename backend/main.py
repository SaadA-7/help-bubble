# python env, FastAPI app with QA pipeline

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch
from pydantic import BaseModel
import json
import os
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HelpBubble API",
    description="E-commerce Customer Support Chatbot API",
    version="1.0.0"
)

# Get allowed origins from environment
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # In production, need to specify frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained QA model
MODEL_NAME = "distilbert-base-cased-distilled-squad"
try:
    qa_pipeline = pipeline("question-answering", model=MODEL_NAME, tokenizer=MODEL_NAME)
    logger.info(f"Successfully loaded model: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    qa_pipeline = None

# E-commerce knowledge base: example for now can update later for real world use.
ECOMMERCE_CONTEXTS = {
    "returns": """Our return policy allows customers to return items within 30 days of purchase. Items must be in original condition with tags attached. Electronics must include all original accessories and packaging. Refunds are processed within 5-7 business days after we receive the returned item. To initiate a return, log into your account and click 'Return Item' next to your order.""",
    
    "shipping": """We offer free standard shipping on orders over $50. Standard shipping takes 3-5 business days. Express shipping (1-2 business days) costs $15.99. International shipping is available to most countries and takes 7-14 business days. You can track your order using the tracking number sent to your email after shipment.""",
    
    "payment": """We accept all major credit cards, PayPal, Apple Pay, and Google Pay. Payment is processed securely through our encrypted checkout system. For orders over $500, we may require additional verification. If your payment fails, please check your card details and ensure sufficient funds are available.""",
    
    "products": """All our products come with a 1-year manufacturer warranty. Electronic items include a 30-day satisfaction guarantee. Product specifications and compatibility information can be found on each product page. If you need help choosing the right product, our chat support is available 24/7.""",
    
    "account": """You can create an account during checkout or from our homepage. Account benefits include order tracking, faster checkout, exclusive deals, and order history. If you forgot your password, click 'Forgot Password' on the login page. To update your information, go to 'My Account' after logging in.""",
    
    "promotions": """We regularly offer seasonal sales, flash deals, and newsletter subscriber discounts. Sign up for our newsletter to receive exclusive 10% off your first order. Student discounts are available with valid .edu email addresses. Check our homepage for current promotions and coupon codes."""
}

class QuestionRequest(BaseModel):
    question: str
    context: Optional[str] = None
    user_id: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    confidence: float
    context_used: str
    response_time: float
    timestamp: str

def find_best_context(question: str) -> str:
    """Find the most relevant context for the question using keyword matching"""
    question_lower = question.lower()
    
    # Keyword mapping for different categories
    keywords = {
        "returns": ["return", "refund", "exchange", "send back", "money back"],
        "shipping": ["ship", "delivery", "track", "arrive", "shipping", "when will"],
        "payment": ["payment", "pay", "card", "paypal", "checkout", "billing"],
        "products": ["warranty", "guarantee", "specification", "compatible", "feature"],
        "account": ["account", "login", "password", "profile", "register", "sign up"],
        "promotions": ["discount", "coupon", "sale", "promo", "deal", "offer"]
    }
    
    # Score each context based on keyword matches
    scores = {}
    for category, category_keywords in keywords.items():
        scores[category] = sum(1 for keyword in category_keywords if keyword in question_lower)
    
    # Return the context with the highest score, default to returns if no match
    best_category = max(scores, key=scores.get) if max(scores.values()) > 0 else "returns"
    return ECOMMERCE_CONTEXTS[best_category]

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Process a customer question and return an answer"""
    try:
        start_time = datetime.now()
        
        if not qa_pipeline:
            raise HTTPException(status_code=500, detail="QA model not loaded")
        
        # Use provided context or find the best matching context
        context = request.context if request.context else find_best_context(request.question)
        
        # Get answer from the model
        result = qa_pipeline(question=request.question, context=context)
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        # Log the interaction
        logger.info(f"Question: {request.question[:100]}... | Answer: {result['answer']} | Confidence: {result['score']:.3f}")
        
        return QuestionResponse(
            answer=result['answer'],
            confidence=result['score'],
            context_used=context[:200] + "..." if len(context) > 200 else context,
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": qa_pipeline is not None,
        "model_name": MODEL_NAME,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/contexts")
async def get_contexts():
    """Get available knowledge base contexts"""
    return {
        "contexts": list(ECOMMERCE_CONTEXTS.keys()),
        "total_contexts": len(ECOMMERCE_CONTEXTS)
    }

@app.post("/test-context")
async def test_with_context(context: str, question: str):
    """Test the model with custom context and question"""
    try:
        if not qa_pipeline:
            raise HTTPException(status_code=500, detail="QA model not loaded")
            
        result = qa_pipeline(question=question, context=context)
        
        return {
            "question": question,
            "context": context[:200] + "..." if len(context) > 200 else context,
            "answer": result['answer'],
            "confidence": result['score'],
            "start": result['start'],
            "end": result['end']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)