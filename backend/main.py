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
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HelpBubble API",
    description="E-commerce Customer Support Chatbot API",
    version="1.0.0"
)

# Get allowed origins from environment variable
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
logger.info(f"CORS allowed origins: {allowed_origins}")

# CORS middleware - this was broken in your original code
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Use environment variable or allow all
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Load pre-trained QA model
MODEL_NAME = "distilbert-base-cased-distilled-squad"
qa_pipeline = None

try:
    logger.info(f"Starting to load model: {MODEL_NAME}")
    logger.info(f"Available device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Force CPU usage on Render (free tier doesn't have GPU)
    qa_pipeline = pipeline("question-answering", 
                          model=MODEL_NAME, 
                          tokenizer=MODEL_NAME,
                          device=-1)  # Force CPU
    logger.info(f"Successfully loaded model: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(f"Full traceback: {traceback.format_exc()}")
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
        logger.info(f"=== NEW REQUEST ===")
        logger.info(f"Received question: {request.question}")
        logger.info(f"User ID: {request.user_id}")
        logger.info(f"Custom context provided: {bool(request.context)}")
        
        start_time = datetime.now()
        
        # Check if model is loaded
        if not qa_pipeline:
            logger.error("QA model is not loaded! Cannot process request.")
            raise HTTPException(
                status_code=500, 
                detail="AI model is not available. Please try again later."
            )
        
        logger.info("Model is loaded, processing question...")
        
        # Use provided context or find the best matching context
        if request.context:
            context = request.context
            logger.info("Using custom context provided by user")
        else:
            context = find_best_context(request.question)
            logger.info(f"Found best context category, context length: {len(context)}")
        
        # Log context preview
        logger.info(f"Context preview: {context[:100]}...")
        
        # Get answer from the model
        logger.info("Calling qa_pipeline with question and context...")
        
        try:
            result = qa_pipeline(question=request.question, context=context)
            logger.info(f"Pipeline successful! Result keys: {list(result.keys())}")
            logger.info(f"Answer: {result.get('answer', 'NO ANSWER')}")
            logger.info(f"Confidence: {result.get('score', 'NO SCORE')}")
            
        except Exception as pipeline_error:
            logger.error(f"Pipeline execution failed: {str(pipeline_error)}")
            logger.error(f"Pipeline error type: {type(pipeline_error)}")
            logger.error(f"Pipeline traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"AI processing failed: {str(pipeline_error)}"
            )
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        # Log the successful interaction
        logger.info(f"SUCCESS - Question processed in {response_time:.3f}s")
        logger.info(f"Final answer: {result['answer']}")
        logger.info(f"Final confidence: {result['score']:.3f}")
        
        response = QuestionResponse(
            answer=result['answer'],
            confidence=result['score'],
            context_used=context[:200] + "..." if len(context) > 200 else context,
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info("=== REQUEST COMPLETED SUCCESSFULLY ===")
        return response
        
    except HTTPException as he:
        # Re-raise HTTP exceptions (they're already handled)
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
        
    except Exception as e:
        logger.error(f"=== UNEXPECTED ERROR ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        logger.error(f"=== END ERROR LOG ===")
        
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = qa_pipeline is not None
    logger.info(f"Health check - Model loaded: {model_status}")
    
    return {
        "status": "healthy" if model_status else "degraded",
        "model_loaded": model_status,
        "model_name": MODEL_NAME if model_status else "Not loaded",
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
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
        logger.error(f"Test context error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HelpBubble API is running!",
        "docs_url": "/docs",
        "health_url": "/health",
        "model_loaded": qa_pipeline is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)