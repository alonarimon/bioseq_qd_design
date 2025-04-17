import Levenshtein


def evaluate_solutions_set(solutions: list[list[int]], ref_solutions: list[list[int]], scores: list[float]):
    """
    Evaluate the solutions using the oracle scores.
    :param solutions: List of solutions to evaluate.
    :param ref_solutions: List of reference solutions.
    :param scores: List of scores for the solutions.
    :param k: The number of top solutions to consider.
    :return: max, diversity, mean, and novelty scores
    """
    assert len(solutions) == len(scores), "Solutions and scores must have the same length."

    N = len(solutions)
    # Calculate max, diversity, mean, and novelty scores
    max_score = float(max(scores))
    diversity_score = 1 / (N * (N - 1)) * sum(
        [sum([Levenshtein.distance(s1, s2) for s2 in solutions]) for s1 in solutions]
    )
    mean_score = float(sum(scores) / N)
    novelty_score = -1
    if ref_solutions is not None:
        novelty_score = sum([min([Levenshtein.distance(s, ref) for ref in ref_solutions]) for s in solutions]) / N

    return max_score, diversity_score, mean_score, novelty_score