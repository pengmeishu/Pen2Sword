"""
Random utilities for embedding expansion.
"""
import random


def random_choice(x: int) -> int:
    """
    Randomly select a number in [0, x)

    Args:
        x: upper bound (exclusive)

    Returns:
        Random integer in range [0, x)
    """
    return random.randint(0, x - 1)


def generate_random_match(org_size: int, target_size: int) -> dict:
    """
    Generate Random Matching Dictionary.

    Maps each index in range [org_size, target_size) to a random index in [0, org_size).

    Args:
        org_size: Original size (number of columns/rows to keep)
        target_size: Target size (desired expanded size)

    Returns:
        Dictionary mapping {i: random_choice} for i in range(org_size, target_size)

    Example:
        >>> generate_random_match(2, 6)
        {2: 0, 3: 1, 4: 0, 5: 0}
        >>> generate_random_match(3, 6)
        {3: 1, 4: 1, 5: 2}
    """
    match_dict = {}
    for i in range(org_size, target_size):
        match_dict[i] = random_choice(org_size)
    return match_dict
