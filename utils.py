import pandas as pd
from copy import deepcopy


def convert_results_to_pd(
    prompts, prompts_results
):
    """
    Convert prompts results to data frame
    Args:
        prompts: dictionary from word (e.g., profession)
        prompts_results: dictionary from word to intervention results
    """

    results = []
    for word in prompts_results:
        intervention = prompts[word]
        (
            candidate1_base_prob,
            candidate2_base_prob,
        ) = prompts_results[word]

        results_base = {  # strings
            "word": word,
            "base_string": intervention.base_strings[0],
            "candidate1": intervention.candidates[0],
            "candidate2": intervention.candidates[1],
            # base probs
            "candidate1_base_prob": float(candidate1_base_prob),
            "candidate2_base_prob": float(candidate2_base_prob)
        }

        results.append(results_base)
    return pd.DataFrame(results)
