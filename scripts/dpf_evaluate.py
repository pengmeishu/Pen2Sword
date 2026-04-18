#!/usr/bin/env python3
"""
DPF Evaluation Script.

Uses GPT-4o to evaluate candidate models and select the best expert model
for a given target domain.

Usage:
    python scripts/dpf_evaluate.py --DOMAIN code --CANDIDATE_MODELS "Llama-2-7B-Chat,CodeLlama-7B,Vicuna-7B-v1.3"

Note:
    This script requires OpenAI API key. Set OPENAI_API_KEY environment variable
    or pass it via --OPENAI_API_KEY argument.
"""
import os
import sys
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dpf module directly to avoid triggering src/__init__.py (which imports torch)
import importlib.util
spec = importlib.util.spec_from_file_location("dpf_prompt_template", "src/methods/dpf/prompt_template.py")
dpf_prompt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dpf_prompt_module)

spec2 = importlib.util.spec_from_file_location("dpf_evaluator", "src/methods/dpf/evaluator.py")
dpf_evaluator_module = importlib.util.module_from_spec(spec2)
sys.modules['src.methods.dpf.prompt_template'] = dpf_prompt_module
sys.modules['src.methods.dpf.evaluator'] = dpf_evaluator_module
spec2.loader.exec_module(dpf_evaluator_module)

DPFEvaluator = dpf_evaluator_module.DPFEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pre-defined evaluation questions for code domain
CODE_QUESTIONS_WITH_REFERENCES = [
    (
        "Write a Python function to reverse a linked list.",
        "def reverse_linked_list(head):\n    prev = None\n    current = head\n    while current:\n        next_node = current.next\n        current.next = prev\n        prev = current\n        current = next_node\n    return prev"
    ),
    (
        "Explain the difference between a stack and a queue. Provide examples of when to use each.",
        "Stack: LIFO (Last In First Out). Use for function calls, undo operations.\nQueue: FIFO (First In First Out). Use for task scheduling, breadth-first search."
    ),
    (
        "What is the time complexity of binary search? Explain why.",
        "O(log n). Because each comparison eliminates half of the remaining elements."
    ),
    (
        "Write a SQL query to find the second highest salary from an employees table.",
        "SELECT MAX(salary) FROM employees WHERE salary < (SELECT MAX(salary) FROM employees)"
    ),
    (
        "What is the difference between deep copy and shallow copy in Python?",
        "Shallow copy: creates new object but copies references of nested objects.\nDeep copy: creates new object and recursively copies all nested objects."
    ),
]

# Pre-defined evaluation questions for math domain
MATH_QUESTIONS_WITH_REFERENCES = [
    (
        "Solve for x: 2x + 5 = 15. Show your work.",
        "2x + 5 = 15\n2x = 10\nx = 5"
    ),
    (
        "What is the derivative of f(x) = x^2 + 3x + 2?",
        "f'(x) = 2x + 3"
    ),
    (
        "Explain the Pythagorean theorem and provide an example.",
        "a² + b² = c². Example: 3² + 4² = 5² (9 + 16 = 25)"
    ),
    (
        "What is the probability of rolling a sum of 7 with two dice?",
        "6/36 = 1/6. There are 6 favorable outcomes: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1)"
    ),
    (
        "Solve the quadratic equation x² - 5x + 6 = 0.",
        "(x-2)(x-3) = 0, so x = 2 or x = 3"
    ),
]

DOMAINS = {
    "code": {
        "description": "Programming and code generation",
        "questions": CODE_QUESTIONS_WITH_REFERENCES,
    },
    "math": {
        "description": "Mathematics and problem solving",
        "questions": MATH_QUESTIONS_WITH_REFERENCES,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="DPF Evaluation - Select best expert model using GPT-4o",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate candidate models for code domain
  python scripts/dpf_evaluate.py --DOMAIN code --CANDIDATE_MODELS "Llama-2-7B-Chat,CodeLlama-7B,Vicuna-7B-v1.3"

  # Evaluate with custom questions
  python scripts/dpf_evaluate.py --DOMAIN code --CANDIDATE_MODELS "Llama-2-7B-Chat" --QUESTIONS_FILE questions.json

  # Save results
  python scripts/dpf_evaluate.py --DOMAIN code --CANDIDATE_MODELS "Llama-2-7B-Chat,CodeLlama-7B" --SAVE_RESULTS
        """
    )

    parser.add_argument("--DOMAIN", type=str, required=True,
                        choices=list(DOMAINS.keys()),
                        help="Target domain for evaluation")
    parser.add_argument("--CANDIDATE_MODELS", type=str, required=True,
                        help="Comma-separated list of candidate model names")
    parser.add_argument("--OPENAI_API_KEY", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--MODEL", type=str, default="gpt-4o",
                        help="GPT model to use for evaluation (default: gpt-4o)")
    parser.add_argument("--TOP_K", type=int, default=1,
                        help="Number of top models to select (default: 1)")
    parser.add_argument("--SAVE_RESULTS", action="store_true",
                        help="Save evaluation results to JSON")
    parser.add_argument("--OUTPUT_DIR", type=str, default="./dpf_results",
                        help="Directory to save results")
    parser.add_argument("--DRY_RUN", action="store_true",
                        help="Dry run: show evaluation questions without calling API")

    return parser.parse_args()


def main(args):
    # Parse candidate models
    candidate_models = [m.strip() for m in args.CANDIDATE_MODELS.split(",")]

    # Dry run - show questions without calling API
    if args.DRY_RUN:
        domain_data = DOMAINS.get(args.DOMAIN, {})
        questions_with_references = domain_data.get("questions", [])

        print("\n" + "=" * 60)
        print(f"DRY RUN - {args.DOMAIN} domain evaluation")
        print("=" * 60)
        print(f"Candidate models: {', '.join(candidate_models)}")
        print(f"Top-K: {args.TOP_K}")
        print(f"\nEvaluation Questions ({len(questions_with_references)}):")
        for i, (q, _) in enumerate(questions_with_references, 1):
            print(f"\nQ{i}: {q}")
        print("\n" + "=" * 60)
        print("To run actual evaluation, set OPENAI_API_KEY and remove --DRY_RUN")
        print("=" * 60)
        return

    # Get API key
    api_key = args.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key is required. "
            "Set OPENAI_API_KEY environment variable or pass --OPENAI_API_KEY"
        )

    logger.info(f"Domain: {args.DOMAIN}")
    logger.info(f"Candidates: {candidate_models}")
    logger.info(f"Top-K: {args.TOP_K}")

    # Get domain data
    domain_data = DOMAINS.get(args.DOMAIN, {})
    questions_with_references = domain_data.get("questions", [])

    if not questions_with_references:
        raise ValueError(f"No predefined questions for domain: {args.DOMAIN}")

    logger.info(f"Using {len(questions_with_references)} predefined questions")

    # Initialize evaluator
    evaluator = DPFEvaluator(api_key=api_key, model=args.MODEL)

    # Note: collect_responses is not implemented in this script
    # You need to provide candidate predictions separately
    # For now, we demonstrate the scoring flow with placeholder predictions

    logger.warning(
        "collect_responses is not implemented. "
        "This script currently only demonstrates the scoring flow. "
        "To use DPF properly, you need to:\n"
        "1. Run your candidate models on the evaluation questions\n"
        "2. Load the predictions into candidate_predictions dict\n"
        "3. Call evaluator.evaluate_candidates()"
    )

    # For demonstration, create dummy predictions
    # In practice, you should replace this with actual model outputs
    candidate_predictions = {
        model: [f"Dummy prediction for question {i}" for i in range(len(questions_with_references))]
        for model in candidate_models
    }

    # Evaluate (with dummy predictions for demonstration)
    # In real usage, you would run models first and pass actual predictions
    logger.info("=" * 60)
    logger.info("Demonstration mode - using dummy predictions")
    logger.info("In practice, run your models first and pass actual predictions")
    logger.info("=" * 60)

    # For real evaluation, uncomment and use actual predictions:
    # scores = evaluator.evaluate_candidates(
    #     domain=args.DOMAIN,
    #     candidate_models=candidate_models,
    #     questions_with_references=questions_with_references,
    #     candidate_predictions=candidate_predictions
    # )

    # Print evaluation questions for reference
    print("\n" + "=" * 60)
    print(f"Evaluation Questions for Domain: {args.DOMAIN}")
    print("=" * 60)
    for i, (q, _) in enumerate(questions_with_references, 1):
        print(f"\nQ{i}: {q}")

    # Select top models
    # selected = evaluator.select_top_k(scores, k=args.TOP_K)

    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print(f"1. Run your candidate models ({', '.join(candidate_models)})")
    print(f"   on the {len(questions_with_references)} questions above")
    print("2. Collect predictions and call evaluator.evaluate_candidates()")
    print(f"3. Use evaluator.select_top_k() to get top-{args.TOP_K} model(s)")
    print("=" * 60)

    # Save results if requested
    if args.SAVE_RESULTS:
        import json
        os.makedirs(args.OUTPUT_DIR, exist_ok=True)

        results = {
            "domain": args.DOMAIN,
            "candidate_models": candidate_models,
            "questions": [{"question": q, "reference": r} for q, r in questions_with_references],
            "note": "Demonstration mode - replace predictions with actual model outputs"
        }

        output_path = os.path.join(args.OUTPUT_DIR, f"dpf_evaluation_{args.DOMAIN}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
