import json
import os
import re
from typing import List, Dict, Tuple
from collections import Counter
from transformers import pipeline
import time
from datasets import load_dataset
import string

class QAEvaluator:
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """Initialize the evaluator with a QA model"""
        self.model_name = model_name
        self.qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)
        
    def normalize_answer(self, s: str) -> str:
        """Normalize answer text for comparison"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def exact_match_score(self, prediction: str, ground_truth: str) -> int:
        """Calculate exact match score"""
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score between prediction and ground truth"""
        pred_tokens = self.normalize_answer(prediction).split()
        truth_tokens = self.normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
        
        common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common_tokens.values())
        
        if num_same == 0:
            return 0
        
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def evaluate_sample(self, question: str, context: str, ground_truth: str) -> Dict:
        """Evaluate a single question-answer pair"""
        start_time = time.time()
        
        try:
            result = self.qa_pipeline(question=question, context=context)
            prediction = result['answer']
            confidence = result['score']
            inference_time = time.time() - start_time
            
            em_score = self.exact_match_score(prediction, ground_truth)
            f1_score_val = self.f1_score(prediction, ground_truth)
            
            return {
                'prediction': prediction,
                'ground_truth': ground_truth,
                'exact_match': em_score,
                'f1_score': f1_score_val,
                'confidence': confidence,
                'inference_time': inference_time,
                'success': True
            }
        except Exception as e:
            return {
                'prediction': '',
                'ground_truth': ground_truth,
                'exact_match': 0,
                'f1_score': 0.0,
                'confidence': 0.0,
                'inference_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def evaluate_dataset(self, dataset: List[Dict]) -> Dict:
        """Evaluate the model on a dataset"""
        print(f"Evaluating {len(dataset)} samples...")
        
        results = []
        total_em = 0
        total_f1 = 0
        total_time = 0
        successful_predictions = 0
        
        for i, sample in enumerate(dataset):
            if i % 10 == 0:
                print(f"Processing sample {i+1}/{len(dataset)}")
            
            result = self.evaluate_sample(
                sample['question'], 
                sample['context'], 
                sample['answer']['text'] if isinstance(sample['answer'], dict) else sample['answer']
            )
            
            results.append(result)
            
            if result['success']:
                total_em += result['exact_match']
                total_f1 += result['f1_score']
                successful_predictions += 1
            
            total_time += result['inference_time']
        
        # Calculate aggregate metrics
        avg_em = total_em / len(dataset) if dataset else 0
        avg_f1 = total_f1 / len(dataset) if dataset else 0
        avg_time = total_time / len(dataset) if dataset else 0
        success_rate = successful_predictions / len(dataset) if dataset else 0
        
        evaluation_results = {
            'model_name': self.model_name,
            'dataset_size': len(dataset),
            'metrics': {
                'exact_match': avg_em,
                'f1_score': avg_f1,
                'average_inference_time': avg_time,
                'success_rate': success_rate
            },
            'detailed_results': results
        }
        
        return evaluation_results

def load_evaluation_data() -> List[Dict]:
    """Load evaluation dataset"""
    try:
        # Try to load our custom e-commerce dataset
        with open('data/ecommerce_qa_eval.json', 'r') as f:
            ecommerce_data = json.load(f)
        print(f"Loaded {len(ecommerce_data)} e-commerce samples")
        return ecommerce_data
    except FileNotFoundError:
        print("E-commerce dataset not found, creating sample data...")
        # Create sample data if file doesn't exist
        sample_data = [
            {
                "question": "How long do I have to return an item?",
                "context": "Our return policy allows customers to return items within 30 days of purchase. Items must be in original condition with tags attached.",
                "answer": "30 days"
            },
            {
                "question": "How much does express shipping cost?",
                "context": "We offer free standard shipping on orders over $50. Express shipping costs $15.99 and takes 1-2 business days.",
                "answer": "$15.99"
            },
            {
                "question": "What warranty comes with electronics?",
                "context": "All our products come with a manufacturer warranty. Electronics typically have a 1-year warranty.",
                "answer": "1-year warranty"
            }
        ]
        return sample_data

def load_squad_validation_sample(sample_size: int = 100) -> List[Dict]:
    """Load a sample from SQuAD validation set"""
    try:
        squad_dataset = load_dataset("squad", split=f"validation[:{sample_size}]")
        squad_data = []
        
        for sample in squad_dataset:
            squad_data.append({
                'question': sample['question'],
                'context': sample['context'],
                'answer': sample['answers']['text'][0] if sample['answers']['text'] else ""
            })
        
        print(f"Loaded {len(squad_data)} SQuAD validation samples")
        return squad_data
    
    except Exception as e:
        print(f"Could not load SQuAD dataset: {e}")
        return []

def compare_models(models: List[str], eval_data: List[Dict]) -> Dict:
    """Compare multiple models on the same dataset"""
    results = {}
    
    for model_name in models:
        print(f"\nEvaluating {model_name}...")
        evaluator = QAEvaluator(model_name)
        model_results = evaluator.evaluate_dataset(eval_data)
        results[model_name] = model_results
    
    return results

def save_results(results: Dict, filename: str = "evaluation_results.json"):
    """Save evaluation results to file"""
    os.makedirs("models", exist_ok=True)
    filepath = os.path.join("models", filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filepath}")

def print_results_summary(results: Dict):
    """Print a summary of evaluation results"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    if 'metrics' in results:
        # Single model results
        metrics = results['metrics']
        print(f"Model: {results.get('model_name', 'Unknown')}")
        print(f"Dataset Size: {results.get('dataset_size', 0)}")
        print(f"Exact Match Score: {metrics['exact_match']:.3f} ({metrics['exact_match']*100:.1f}%)")
        print(f"F1 Score: {metrics['f1_score']:.3f} ({metrics['f1_score']*100:.1f}%)")
        print(f"Average Inference Time: {metrics['average_inference_time']:.3f}s")
        print(f"Success Rate: {metrics['success_rate']:.3f} ({metrics['success_rate']*100:.1f}%)")
    else:
        # Multiple model comparison
        print(f"{'Model':<25} {'EM Score':<10} {'F1 Score':<10} {'Avg Time':<12} {'Success Rate':<12}")
        print("-" * 75)
        
        for model_name, model_results in results.items():
            metrics = model_results['metrics']
            print(f"{model_name:<25} {metrics['exact_match']:.3f}     {metrics['f1_score']:.3f}     {metrics['average_inference_time']:.3f}s       {metrics['success_rate']:.3f}")

def main():
    """Main evaluation function"""
    print("Starting HelpBubble Model Evaluation...")
    
    # Load evaluation datasets
    ecommerce_data = load_evaluation_data()
    squad_data = load_squad_validation_sample(50)  # Small sample for demo
    
    # Models to evaluate
    models_to_test = [
        "distilbert-base-cased-distilled-squad",
        # "bert-large-uncased-whole-word-masking-finetuned-squad",  # Uncomment if you want to test
        # "roberta-base-squad2"  # Uncomment if you want to test
    ]
    
    # Evaluate on e-commerce dataset
    print("\n" + "="*50)
    print("EVALUATING ON E-COMMERCE DATASET")
    print("="*50)
    
    if len(models_to_test) == 1:
        evaluator = QAEvaluator(models_to_test[0])
        ecommerce_results = evaluator.evaluate_dataset(ecommerce_data)
        save_results(ecommerce_results, "ecommerce_evaluation_results.json")
        print_results_summary(ecommerce_results)
    else:
        ecommerce_results = compare_models(models_to_test, ecommerce_data)
        save_results(ecommerce_results, "ecommerce_model_comparison.json")
        print_results_summary(ecommerce_results)
    
    # Evaluate on SQuAD dataset if available
    if squad_data:
        print("\n" + "="*50)
        print("EVALUATING ON SQUAD VALIDATION SAMPLE")
        print("="*50)
        
        if len(models_to_test) == 1:
            evaluator = QAEvaluator(models_to_test[0])
            squad_results = evaluator.evaluate_dataset(squad_data)
            save_results(squad_results, "squad_evaluation_results.json")
            print_results_summary(squad_results)
        else:
            squad_results = compare_models(models_to_test, squad_data)
            save_results(squad_results, "squad_model_comparison.json")
            print_results_summary(squad_results)
    
    print("\nEvaluation complete! Check the 'models' folder for detailed results.")

if __name__ == "__main__":
    main()