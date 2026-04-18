"""
DPF Evaluator.

Uses GPT-4o to evaluate candidate models and select the best expert model.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DPFEvaluator:
    """
    Domain-Preserving Function evaluator.

    Uses GPT-4o to evaluate candidate models across four dimensions:
    - Accuracy
    - Domain Coverage
    - Domain Depth
    - Terminology Appropriateness
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        """
        Initialize DPF Evaluator.

        Args:
            openai_api_key: OpenAI API key
            model: GPT model to use for evaluation (default: gpt-4o)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for DPF evaluation. "
                "Install with: pip install openai"
            )

        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        logger.info(f"Initialized DPF Evaluator with model: {model}")

    def generate_questions(self, domain: str, num_questions: int = 10) -> List[str]:
        """
        Generate domain-specific questions using GPT-4o.

        Args:
            domain: The target domain (e.g., "code", "math")
            num_questions: Number of questions to generate

        Returns:
            List of generated questions
        """
        from .prompt_template import generate_question_prompt

        prompt = generate_question_prompt(domain, num_questions)

        logger.info(f"Generating {num_questions} questions for domain: {domain}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates domain-specific questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )

        content = response.choices[0].message.content
        questions = self._parse_questions(content)

        logger.info(f"Generated {len(questions)} questions")
        return questions

    def _parse_questions(self, text: str) -> List[str]:
        """Parse questions from GPT response."""
        lines = text.strip().split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            # Match numbered questions like "1. Question text"
            match = re.match(r'^\d+\.\s*(.+)', line)
            if match:
                questions.append(match.group(1))
        return questions

    def collect_responses(
        self,
        candidate_models: List[str],
        questions: List[str],
        model_paths: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[str]]:
        """
        Collect responses from candidate models.

        Note: This requires actual model inference which is not implemented here.
        This method returns a placeholder - actual implementation would need
        to load models and run inference.

        Args:
            candidate_models: List of candidate model names
            questions: List of questions to answer
            model_paths: Optional dict mapping model_name -> model_path

        Returns:
            Dict mapping model_name -> list of responses
        """
        logger.warning(
            "collect_responses requires actual model inference. "
            "Please implement model loading and inference separately."
        )
        raise NotImplementedError(
            "collect_responses requires actual model inference. "
            "You need to implement model loading and inference based on your setup."
        )

    def score_model(
        self,
        domain: str,
        question: str,
        reference: str,
        candidate_predictions: Dict[str, str]
    ) -> Dict[str, dict]:
        """
        Score candidate model responses using GPT-4o.

        Args:
            domain: The target domain
            question: The question being evaluated
            reference: The ground truth / reference answer
            candidate_predictions: Dict mapping model_name -> prediction

        Returns:
            Dict mapping model_name -> score dict with accuracy, coverage, depth, terminology, total
        """
        from .prompt_template import format_evaluation_prompt

        prompt = format_evaluation_prompt(
            domain=domain,
            question=question,
            candidates_with_predictions=candidate_predictions,
            reference=reference
        )

        logger.info(f"Scoring {len(candidate_predictions)} candidates for question: {question[:50]}...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a strict domain expert evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=2048
        )

        content = response.choices[0].message.content
        scores = self._parse_scores(content, candidate_predictions.keys())

        return scores

    def _parse_scores(self, text: str, candidate_names: List[str]) -> Dict[str, dict]:
        """Parse scores from GPT evaluation response."""
        scores = {}

        for name in candidate_names:
            # Find the section for this candidate
            pattern = rf'{re.escape(name)}:.*?Total Score:\s*(\d+)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

            if match:
                # Try to extract individual scores
                accuracy = self._extract_score(text, name, 'Accuracy')
                coverage = self._extract_score(text, name, 'Domain Coverage')
                depth = self._extract_score(text, name, 'Depth')
                terminology = self._extract_score(text, name, 'Terminology')
                total = int(match.group(1))

                scores[name] = {
                    'accuracy': accuracy,
                    'domain_coverage': coverage,
                    'depth': depth,
                    'terminology': terminology,
                    'total': total
                }
            else:
                logger.warning(f"Could not parse scores for {name}")
                scores[name] = {
                    'accuracy': 0,
                    'domain_coverage': 0,
                    'depth': 0,
                    'terminology': 0,
                    'total': 0
                }

        return scores

    def _extract_score(self, text: str, candidate_name: str, metric: str) -> int:
        """Extract a specific metric score for a candidate."""
        # Find the section for this candidate
        pattern = rf'{re.escape(candidate_name)}:(.*?)(?:Total Score:|$)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

        if match:
            section = match.group(1)
            # Look for the metric pattern
            metric_pattern = rf'{re.escape(metric)}:\s*(\d+)'
            metric_match = re.search(metric_pattern, section)
            if metric_match:
                return int(metric_match.group(1))

        return 0

    def evaluate_candidates(
        self,
        domain: str,
        candidate_models: List[str],
        questions_with_references: List[Tuple[str, str]],
        candidate_predictions: Dict[str, List[str]]
    ) -> Dict[str, dict]:
        """
        Full evaluation of candidate models.

        Args:
            domain: The target domain
            candidate_models: List of candidate model names
            questions_with_references: List of (question, reference) tuples
            candidate_predictions: Dict mapping model_name -> list of predictions (aligned with questions)

        Returns:
            Dict mapping model_name -> aggregated scores
        """
        logger.info(f"Evaluating {len(candidate_models)} candidates on {len(questions_with_references)} questions")

        # Aggregate scores across all questions
        aggregated = {name: {'total': 0, 'count': 0, 'details': []} for name in candidate_models}

        for i, (question, reference) in enumerate(questions_with_references):
            # Build predictions dict for this question
            preds_for_q = {
                name: candidate_predictions[name][i] if i < len(candidate_predictions[name]) else ""
                for name in candidate_models
            }

            # Score this question
            scores = self.score_model(domain, question, reference, preds_for_q)

            # Aggregate
            for name, score_dict in scores.items():
                aggregated[name]['total'] += score_dict['total']
                aggregated[name]['count'] += 1
                aggregated[name]['details'].append(score_dict)

        # Compute average scores
        final_scores = {}
        for name, agg in aggregated.items():
            if agg['count'] > 0:
                avg_total = agg['total'] / agg['count']
                final_scores[name] = {
                    'total_score': agg['total'],
                    'num_questions': agg['count'],
                    'average_score': avg_total,
                    'details': agg['details']
                }
            else:
                final_scores[name] = {
                    'total_score': 0,
                    'num_questions': 0,
                    'average_score': 0,
                    'details': []
                }

        # Sort by average score
        ranked = sorted(
            final_scores.items(),
            key=lambda x: x[1]['average_score'],
            reverse=True
        )

        logger.info("Evaluation Results:")
        for name, scores in ranked:
            logger.info(f"  {name}: avg={scores['average_score']:.2f}, total={scores['total_score']}")

        return dict(ranked)

    def select_top_k(self, scores: Dict[str, dict], k: int = 1) -> List[str]:
        """
        Select top-k models based on evaluation scores.

        Args:
            scores: Output from evaluate_candidates
            k: Number of top models to select

        Returns:
            List of top-k model names
        """
        ranked = sorted(
            scores.items(),
            key=lambda x: x[1]['average_score'],
            reverse=True
        )

        return [name for name, _ in ranked[:k]]
