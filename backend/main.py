# python env, FastAPI app with QA pipeline

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import torch
from pydantic import BaseModel
import os
from typing import Optional
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

# CORS configuration - temporarily hardcode for debugging
FRONTEND_URL = "https://help-bubble-1p8hpfb05-saada-7s-projects.vercel.app"
allowed_origins = [
    FRONTEND_URL,
    "http://localhost:3000",  # For local development
    "https://localhost:3000",
    "*"  # Temporary - remove in production
]

logger.info(f"CORS allowed origins: {allowed_origins}")
logger.info(f"Environment ALLOWED_ORIGINS: {os.getenv('ALLOWED_ORIGINS', 'NOT SET')}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Model configuration
MODEL_NAME = "distilbert-base-cased-distilled-squad"
qa_pipeline = None

# Try to load model
try:
    logger.info(f"Attempting to load model: {MODEL_NAME}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    qa_pipeline = pipeline(
        "question-answering", 
        model=MODEL_NAME, 
        tokenizer=MODEL_NAME,
        device=-1,  # Force CPU
        return_tensors="pt"
    )
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load model: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    qa_pipeline = None

# E-commerce knowledge base
ECOMMERCE_CONTEXTS = {
    "returns": """Our return policy allows customers to return items within 30 days of purchase. Items must be in original condition with tags attached. Electronics must include all original accessories and packaging. Refunds are processed within 5-7 business days after we receive the returned item. To initiate a return, log into your account and click 'Return Item' next to your order.""",
    
    "shipping": """We offer free standard shipping on orders over $50. Standard shipping takes 3-5 business days. Express shipping (1-2 business days) costs $15.99. International shipping is available to most countries and takes 7-14 business days. You can track your order using the tracking number sent to your email after shipment.""",
    
    "payment": """We accept all major credit cards, PayPal, Apple Pay, and Google Pay. Payment is processed securely through our encrypted checkout system. For orders over $500, we may require additional verification. If your payment fails, please check your card details and ensure sufficient funds are available.""",
    
    "products": """All our products come with a 1-year manufacturer warranty. Electronic items include a 30-day satisfaction guarantee. Product specifications and compatibility information can be found on each product page. If you need help choosing the right product, our chat support is available 24/7.""",
    
    "account": """You can create an account during checkout or from our homepage. Account benefits include order tracking, faster checkout, exclusive deals, and order history. If you forgot your password, click 'Forgot Password' on the login page. To update your information, go to 'My Account' after logging in.""",
    
    "promotions": """We regularly offer seasonal sales, flash deals, and newsletter subscriber discounts. Sign up for our newsletter to receive exclusive 10% off your first order. Student discounts are available with valid .edu email addresses. Check our homepage for current promotions and coupon codes."""
}

# Fallback responses for when model isn't available
FALLBACK_RESPONSES = {
    "returns": "You can return items within 30 days of purchase. Items must be in original condition. Refunds take 5-7 business days to process.",
    "shipping": "We offer free standard shipping on orders over $50 (3-5 days) and express shipping for $15.99 (1-2 days).",
    "payment": "We accept major credit cards, PayPal, Apple Pay, and Google Pay through our secure checkout system.",
    "products": "All products come with a 1-year warranty. Electronics have a 30-day satisfaction guarantee.",
    "account": "Create an account for order tracking, faster checkout, and exclusive deals. Use 'Forgot Password' if needed.",
    "promotions": "Sign up for our newsletter for 10% off your first order. Student discounts available with .edu email."
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

def get_fallback_answer(question: str) -> tuple:
    """Get a fallback answer when the model is not available"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ["return", "refund", "exchange"]):
        return FALLBACK_RESPONSES["returns"], "returns"
    elif any(word in question_lower for word in ["ship", "delivery", "track"]):
        return FALLBACK_RESPONSES["shipping"], "shipping"
    elif any(word in question_lower for word in ["payment", "pay", "card", "paypal"]):
        return FALLBACK_RESPONSES["payment"], "payment"
    elif any(word in question_lower for word in ["warranty", "guarantee"]):
        return FALLBACK_RESPONSES["products"], "products"
    elif any(word in question_lower for word in ["account", "login", "password"]):
        return FALLBACK_RESPONSES["account"], "account"
    elif any(word in question_lower for word in ["discount", "coupon", "sale"]):
        return FALLBACK_RESPONSES["promotions"], "promotions"
    else:
        return "I'm currently having technical difficulties, but I'd be happy to help! For immediate assistance, please contact our support team.", "general"

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Process a customer question and return an answer"""
    try:
        logger.info(f"📥 Received question: {request.question}")
        start_time = datetime.now()
        
        # If model is not loaded, use fallback
        if not qa_pipeline:
            logger.warning("⚠️  Model not available, using fallback response")
            
            fallback_answer, category = get_fallback_answer(request.question)
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            return QuestionResponse(
                answer=fallback_answer,
                confidence=0.8,
                context_used=f"Fallback response for {category}",
                response_time=response_time,
                timestamp=datetime.now().isoformat()
            )
        
        # Model is available, use it
        logger.info("🤖 Using AI model to process question")
        
        # Get context
        context = request.context if request.context else find_best_context(request.question)
        
        # Process with model
        result = qa_pipeline(question=request.question, context=context)
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Success - Answer: {result['answer'][:50]}... (confidence: {result['score']:.3f})")
        
        return QuestionResponse(
            answer=result['answer'],
            confidence=result['score'],
            context_used=context[:200] + "..." if len(context) > 200 else context,
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error in ask_question: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Try fallback even on error
        try:
            fallback_answer, category = get_fallback_answer(request.question)
            return QuestionResponse(
                answer=fallback_answer,
                confidence=0.5,
                context_used="Error fallback response",
                response_time=0.1,
                timestamp=datetime.now().isoformat()
            )
        except:
            raise HTTPException(status_code=500, detail="Service temporarily unavailable")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = qa_pipeline is not None
    logger.info(f"🔍 Health check - Model loaded: {model_status}")
    
    return {
        "status": "healthy",  # Always healthy since we have fallback
        "model_loaded": model_status,
        "model_name": MODEL_NAME,
        "fallback_available": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/contexts")
async def get_contexts():
    """Get available knowledge base contexts"""
    return {
        "contexts": list(ECOMMERCE_CONTEXTS.keys()),
        "total_contexts": len(ECOMMERCE_CONTEXTS)
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HelpBubble API is running!",
        "docs_url": "/docs",
        "health_url": "/health",
        "model_loaded": qa_pipeline is not None,
        "fallback_available": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)