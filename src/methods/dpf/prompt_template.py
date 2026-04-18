"""
DPF prompt templates.

Contains the evaluation prompt used for GPT-4o scoring in the
Domain-Preserving Function (DPF) process.
"""

DPF_EVALUATION_PROMPT = """Please evaluate the following model responses in the specified domain. Score each dimension from 0 to 5 for each candidate response, with the total score being the sum (minimum 0), and provide brief reasoning. For each candidate, start with: "Accuracy: x, Domain Coverage: x, Depth: x, Terminology Appropriateness: x, Total Score: x".

Evaluate each candidate response based on the question, the reference answer, and the standards of the specified domain.

1. Accuracy:

- Does the prediction correctly answer the core question?

- Are there any errors or misunderstandings of key information?

- Deduct points for incorrect facts, wrong logic, or misleading statements.

2. Domain Coverage:

- Does the prediction cover all key subdomains from the reference answer?

- Deduct points for missing important content.

- Consider whether the response addresses the main components, steps, or aspects required by the question and reference.

3. Depth:

- Does the prediction provide detailed reasoning, explanation, and insight?

- Deduct points for shallow or superficial answers.

- Higher scores should be given to responses that explain not only what the answer is, but also why.

4. Terminology Appropriateness:

- Are domain-specific terms used correctly and consistently in the specified domain?

- Deduct points for incorrect or missing terminology.

- Consider whether the terminology is precise and appropriate for the domain.

Keep the evaluation concise, strict, and evidence-based. Minor wording differences should not be penalized if the meaning is correct. Evaluate each candidate independently based on the same standard.

Domain: {domain}

Question: {question}

Candidate Model Responses:

- {candidate_1_name}: {prediction_1}

- {candidate_2_name}: {prediction_2}

- {candidate_3_name}: {prediction_3}

...

Reference: {ground_truth}"""

DPF_QUESTION_GENERATION_PROMPT = """Generate {num_questions} domain-specific questions for the following domain: {domain}

Requirements:
- Questions should cover different subdomains within {domain}
- Questions should test knowledge depth, not just surface-level facts
- Include a mix of:
  - Conceptual questions (understanding definitions and principles)
  - Practical questions (applying knowledge to solve problems)
  - Analytical questions (comparing, contrasting, evaluating)

Format each question on a new line, numbered 1 through {num_questions}.
Do not include answers - only questions.
"""


def generate_question_prompt(domain, num_questions=10):
    """
    Generate the prompt for creating domain-specific questions.

    Args:
        domain: The target domain (e.g., "code", "math", "medical")
        num_questions: Number of questions to generate

    Returns:
        Formatted prompt string
    """
    return DPF_QUESTION_GENERATION_PROMPT.format(
        domain=domain,
        num_questions=num_questions
    )


def format_evaluation_prompt(domain, question, candidates_with_predictions, reference):
    """
    Format the evaluation prompt with specific question and candidate responses.

    Args:
        domain: The target domain
        question: The question to evaluate
        candidates_with_predictions: Dict mapping candidate_name -> prediction
        reference: The ground truth / reference answer

    Returns:
        Formatted prompt string
    """
    # Build candidate strings
    candidate_lines = []
    for i, (name, pred) in enumerate(candidates_with_predictions.items(), 1):
        candidate_lines.append(f"- {name}: {pred}")

    candidates_str = "\n".join(candidate_lines)

    return DPF_EVALUATION_PROMPT.format(
        domain=domain,
        question=question,
        candidate_1_name=list(candidates_with_predictions.keys())[0] if candidates_with_predictions else "candidate_1",
        prediction_1=list(candidates_with_predictions.values())[0] if candidates_with_predictions else "",
        candidate_2_name=list(candidates_with_predictions.keys())[1] if len(candidates_with_predictions) > 1 else "candidate_2",
        prediction_2=list(candidates_with_predictions.values())[1] if len(candidates_with_predictions) > 1 else "",
        candidate_3_name=list(candidates_with_predictions.keys())[2] if len(candidates_with_predictions) > 2 else "candidate_3",
        prediction_3=list(candidates_with_predictions.values())[2] if len(candidates_with_predictions) > 2 else "",
        ground_truth=reference
    )
