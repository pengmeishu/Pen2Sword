"""
DPF - Domain-Preserving Function.

Uses GPT-4o to evaluate candidate models and select the best expert model
for a given target domain.

Reference:
    See paper Appendix A.2 - Details for Domain Preserve Function
"""
from .evaluator import DPFEvaluator
from .prompt_template import DPF_EVALUATION_PROMPT, generate_question_prompt

__all__ = [
    "DPFEvaluator",
    "DPF_EVALUATION_PROMPT",
    "generate_question_prompt",
]
