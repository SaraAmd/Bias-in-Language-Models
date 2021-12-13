import argparse
import os
from datetime import datetime
import pandas as pd
import pdb
import random
import torch
import numpy as np
from transformers import (
    BertTokenizer, DistilBertTokenizer, RobertaTokenizer, AutoTokenizer, AutoModelForMaskedLM
)


from prompt import Model, Prompt
from utils import convert_results_to_pd

parser = argparse.ArgumentParser(description="Run a set of neuron experiments.")

parser.add_argument(
    "-model",
    type=str,
    default="bert-base-uncased",
    help="""Model type [roberta, berta, etc.].""",
)

parser.add_argument(
    "-out_dir", default=".", type=str, help="""Path of the result folder."""
)

parser.add_argument(
    "-template_indices",
    nargs="+",
    type=int,
    help="Give the indices of templates if you want to run on only a subset",
)

parser.add_argument(
    "--randomize", default=False, action="store_true", help="Randomize model weights."
)

opt = parser.parse_args()


def get_profession_list():
    # Get the list of all considered professions
    profession_list = []
    with open("experiment_data/professions.json", "r") as f:
        for l in f:
            # there is only one line that eval"s to an array
            for j in eval(l):
                profession_list.append(j[0])

    return profession_list


def get_template_list(indices=None):
    # Get list of all considered templates
    # "That" sentences are ours
    # "Because" sentences are a subset
    # from https://arxiv.org/pdf/1807.11714.pdf (Lu et al.)
    templates = [
        "The {} said that",
        "The {} yelled that",
        "The {} whispered that",
        "The {} wanted that",
        "The {} desired that",
        "The {} wished that",
        "The {} ate because",
        "The {} ran because",
        "The {} drove because",
        "The {} slept because",
        "The {} cried because",
        "The {} laughed because",
        "The {} went home because",
        "The {} stayed up because",
        "The {} was fired because",
        "The {} was promoted because",
        "The {} yelled because",
    ]
    if indices:
        subset_templates = [templates[i - 1] for i in indices]
        print("subset of templates:", subset_templates)
        return subset_templates

    return templates




def construct_prompts(base_sent, professions, tokenizer, DEVICE):
    prompts = {}

    for p in professions:
        prompts[p] = Prompt(
        tokenizer, base_sent, [p], ["he", "she"], device = DEVICE
            )
    return prompts


def run_all(
    model_type="bert-base-uncase",
    device="cuda",
    out_dir=".",
    random_weights=False,
    template_indices=None,
):
    print("Model:", model_type, flush=True)
    # Set up all the potential combinations.
    professions = get_profession_list()
    templates = get_template_list(template_indices)

    # Initialize Model and Tokenizer.
    model = Model(device=device, version=model_type, random_weights=random_weights)
    tokenizer = (
                 BertTokenizer if model.is_bert else
                 RobertaTokenizer if model.is_distilroberta else
                 DistilBertTokenizer if model.is_distilbert else
                 XLMRobertaTokenizer if model.is_xlm_roberta else
                 RobertaTokenizer).from_pretrained(model_type)

    # Set up folder if it does not exist.
    dt_string = datetime.now().strftime("%Y%m%d")

    folder_name = dt_string + "_" + "_bias_probing_"+model_type
    base_path = os.path.join(out_dir, "results", folder_name)
    if random_weights:
        base_path = os.path.join(base_path, "random")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Iterate over all possible templates.
    for temp in templates:
        print("Running template '{}' now...".format(temp), flush=True)


        # Fill in all professions into current template
        prompts = construct_prompts(temp, professions, tokenizer, device)
        #get results from the model
        prompts_results = model.experiment(prompts)

        df = convert_results_to_pd(prompts, prompts_results)

        # Generate file name.
        temp_string = "_".join(temp.replace("{}", "X").split())
        model_type_string = model_type
        fname = "_".join([temp_string, model_type_string])
        #Finally, save each exp separately.
        df.to_csv(os.path.join(base_path, fname + ".csv"))


if __name__ == "__main__":
    seed_val = 10
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_all(
        opt.model,
        device,
        opt.out_dir,
        random_weights=opt.randomize,
        template_indices=opt.template_indices,
    )