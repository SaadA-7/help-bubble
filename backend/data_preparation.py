import json
import os
from typing import List, Dict
import random

def create_ecommerce_qa_dataset():
    """Create a comprehensive e-commerce QA dataset for training and evaluation"""
    
    ecommerce_qa_data = [
        {
            "context": "Our return policy allows customers to return items within 30 days of purchase. Items must be in original condition with tags attached. Electronics must include all original accessories and packaging. Refunds are processed within 5-7 business days after we receive the returned item. To initiate a return, log into your account and click 'Return Item' next to your order.",
            "questions": [
                "How long do I have to return an item?",
                "What condition must items be in for returns?",
                "How long do refunds take?",
                "How do I start a return?",
                "Do electronics need original packaging for returns?"
            ]
        },
        {
            "context": "We offer free standard shipping on orders over $50. Standard shipping takes 3-5 business days. Express shipping costs $15.99 and takes 1-2 business days. Overnight shipping is available for $29.99. International shipping is available to most countries and takes 7-14 business days. You can track your order using the tracking number sent to your email.",
            "questions": [
                "When is shipping free?",
                "How long does standard shipping take?",
                "How much does express shipping cost?",
                "How can I track my order?",
                "Do you ship internationally?"
            ]
        },
        {
            "context": "We accept all major credit cards including Visa, MasterCard, American Express, and Discover. We also accept PayPal, Apple Pay, Google Pay, and Shop Pay. Payment is processed securely through our encrypted checkout system. For orders over $500, we may require additional verification for security purposes.",
            "questions": [
                "What payment methods do you accept?",
                "Is payment processing secure?",
                "Do you accept PayPal?",
                "What happens with large orders?",
                "Can I use Apple Pay?"
            ]
        },
        {
            "context": "All our products come with a manufacturer warranty that varies by product type. Electronics typically have a 1-year warranty. Clothing and accessories have a 90-day quality guarantee. Furniture comes with a 2-year warranty against defects. If you experience any issues, contact our support team with your order number.",
            "questions": [
                "What warranty comes with products?",
                "How long is the electronics warranty?",
                "What about clothing guarantees?",
                "How long is furniture covered?",
                "What do I need to make a warranty claim?"
            ]
        },
        {
            "context": "You can create an account during checkout or from our homepage by clicking 'Sign Up'. Account benefits include faster checkout, order tracking, exclusive member deals, and saved addresses. If you forget your password, click 'Forgot Password' on the login page and we'll send reset instructions to your email.",
            "questions": [
                "How do I create an account?",
                "What are the benefits of having an account?",
                "How do I reset my password?",
                "Can I track orders with an account?",
                "Where do I sign up?"
            ]
        },
        {
            "context": "We regularly offer seasonal sales, flash deals, and exclusive newsletter discounts. New customers get 10% off their first order when they sign up for our newsletter. Students get 15% off with a valid .edu email address. Military personnel receive a 10% discount with ID verification. Check our homepage for current promotions.",
            "questions": [
                "Do you offer student discounts?",
                "How much do new customers save?",
                "Is there a military discount?",
                "Where can I find current promotions?",
                "How do I get the newsletter discount?"
            ]
        },
        {
            "context": "Our customer service team is available 24/7 through live chat, email, and phone. Live chat is the fastest way to get help and is available on every page. Email support typically responds within 2 hours during business days. For urgent issues, call our toll-free number. We also have an extensive FAQ section and video tutorials.",
            "questions": [
                "How can I contact customer service?",
                "Is live chat available 24/7?",
                "How quickly do you respond to emails?",
                "Do you have a phone number?",
                "Where can I find help videos?"
            ]
        },
        {
            "context": "Order modifications can be made within 30 minutes of placing the order. After that, the order enters processing and cannot be changed. To modify an order, log into your account and click 'Modify Order' if available, or contact customer service immediately. Cancellations are possible until the item ships.",
            "questions": [
                "Can I change my order after placing it?",
                "How long do I have to modify an order?",
                "How do I cancel an order?",
                "What if my order is already processing?",
                "Where do I modify my order?"
            ]
        },
        {
            "context": "We take data privacy seriously and follow industry-standard security practices. Your personal information is encrypted and never shared with third parties without your consent. You can view and update your privacy settings in your account dashboard. We only send marketing emails to subscribers who have opted in.",
            "questions": [
                "How is my personal information protected?",
                "Do you share my data with others?",
                "Where can I update privacy settings?",
                "Will you send me marketing emails?",
                "Is my payment information secure?"
            ]
        },
        {
            "context": "Product reviews help other customers make informed decisions. You can leave a review 3 days after your order is delivered. Reviews must be honest and based on your experience with the product. We don't allow fake reviews and may remove reviews that violate our guidelines. Verified purchase reviews are marked with a special badge.",
            "questions": [
                "When can I leave a product review?",
                "What makes a review verified?",
                "Are fake reviews allowed?",
                "How do reviews help other customers?",
                "What are the review guidelines?"
            ]
        }
    ]
    
    # Convert to format suitable for training/evaluation
    formatted_data = []
    for item in ecommerce_qa_data:
        context = item["context"]
        for question in item["questions"]:
            formatted_data.append({
                "id": f"ecomm_{len(formatted_data)}",
                "context": context,
                "question": question,
                "answer": extract_answer_from_context(context, question)
            })
    
    return formatted_data

def extract_answer_from_context(context: str, question: str) -> Dict:
    """
    Simple heuristic to extract answers from context based on question keywords.
    In a real scenario, you'd use the trained model or manual annotation.
    """
    question_lower = question.lower()
    
    # Define answer extraction rules based on question patterns
    if "how long" in question_lower and "return" in question_lower:
        return {"text": "30 days", "answer_start": context.find("30 days")}
    elif "how long" in question_lower and ("shipping" in question_lower or "delivery" in question_lower):
        if "standard" in question_lower:
            return {"text": "3-5 business days", "answer_start": context.find("3-5 business days")}
        elif "express" in question_lower:
            return {"text": "1-2 business days", "answer_start": context.find("1-2 business days")}
    elif "how much" in question_lower and "express" in question_lower:
        return {"text": "$15.99", "answer_start": context.find("$15.99")}
    elif "free" in question_lower and "shipping" in question_lower:
        return {"text": "orders over $50", "answer_start": context.find("orders over $50")}
    elif "warranty" in question_lower and "electronics" in question_lower:
        return {"text": "1-year warranty", "answer_start": context.find("1-year")}
    elif "student" in question_lower and "discount" in question_lower:
        return {"text": "15% off", "answer_start": context.find("15% off")}
    elif "new customer" in question_lower:
        return {"text": "10% off", "answer_start": context.find("10% off")}
    
    # Default: return first sentence as answer
    first_sentence = context.split('.')[0] + '.'
    return {"text": first_sentence, "answer_start": 0}

def save_dataset_to_file(data: List[Dict], filename: str = "ecommerce_qa_dataset.json"):
    """Save the dataset to a JSON file"""
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to {filepath}")
    print(f"Total QA pairs: {len(data)}")
    
    return filepath

def load_squad_sample():
    """Load a sample from SQuAD dataset for comparison"""
    try:
        from datasets import load_dataset
        squad = load_dataset("squad", split="train[:10]")
        return squad
    except Exception as e:
        print(f"Could not load SQuAD dataset: {e}")
        return None

def create_evaluation_split(data: List[Dict], train_ratio: float = 0.8):
    """Split data into training and evaluation sets"""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    return train_data, eval_data

if __name__ == "__main__":
    # Create the e-commerce QA dataset
    print("Creating e-commerce QA dataset...")
    qa_data = create_ecommerce_qa_dataset()
    
    # Split into train/eval
    train_data, eval_data = create_evaluation_split(qa_data)
    
    # Save datasets
    save_dataset_to_file(train_data, "ecommerce_qa_train.json")
    save_dataset_to_file(eval_data, "ecommerce_qa_eval.json")
    save_dataset_to_file(qa_data, "ecommerce_qa_full.json")
    
    # Load SQuAD sample for reference
    squad_sample = load_squad_sample()
    if squad_sample:
        print(f"Loaded {len(squad_sample)} SQuAD samples for reference")
    
    print("\nDataset creation complete!")
    print(f"Training samples: {len(train_data)}")
    print(f"Evaluation samples: {len(eval_data)}")
    print(f"Total samples: {len(qa_data)}")