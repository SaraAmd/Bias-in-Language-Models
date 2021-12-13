
import os
import sys
import pdb
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import operator
import itertools

def get_profession_list():
    # Get the list of all considered professions
    profession_list = []
    with open("experiment_data/professions.json", "r") as f:
        for l in f:
            # there is only one line that eval"s to an array
            for j in eval(l):
                profession_list.append(j[0])

    return profession_list

def analyze_effect_results(results_df, word, stereotypicality, savefig=None):
    # calculate ratio.
    if stereotypicality == "male":
        results_df["bias_ratio"]=(
            results_df["candidate2_base_prob"] / results_df["candidate1_base_prob"]
        )

    else:

        results_df["bias_ratio"] = (
            results_df["candidate1_base_prob"] / results_df["candidate2_base_prob"]
        )

    results_df = results_df[results_df["word"] == word]


#returns bias ratio formula
def get_all_effects(result_df, direction="female"):
    """
    Give fname from a direct effect file
    """

    analyze_effect_results(
       # results_df=indirect_result_df, effect="indirect", word="word", stereotypicality=direction
        results_df=result_df, word="word", stereotypicality=direction
    )

 # the total df just has the odd_ratio and the base sting culumns
    total_df = result_df[
        [

            "base_string",
            "candidate1", #candidate one is token "he"
            "candidate2", #candidate two is token "she"
            "candidate1_base_prob",
            "candidate2_base_prob",
            "profession",
            "stereotypicality",
            "bias_ratio",

        ]
    ]

    return total_df


def main(folder_name="./results/20211205_bert-base-uncased_bias_probing/", model_name="bert-base-uncased"):
    profession_stereotypicality = {}
    with open("experiment_data/professions.json") as f:
        for l in f:
            for p in eval(l):
                profession_stereotypicality[p[0]] = {
                    "definitional": p[1],
                    "stereotypicality": p[2]

                }

    fnames = [
        f
        for f in os.listdir(folder_name)
        if "_" + model_name + ".csv" in f and f.endswith("csv")
    ]


    files = [os.path.join(folder_name, f) for f in fnames]

    def get_profession(s):
        return s.split()[1]

    def get_template(s):
        initial_string = s.split()
        initial_string[1] = "_"
        return " ".join(initial_string)

    def get_stereotypicality(vals):
        return profession_stereotypicality[vals]["stereotypicality"]

    dfs = []
    for csv in files:

        df =pd.read_csv(csv)

        df["profession"] = df["base_string"].apply(get_profession)

        # get the sterotypicalit for each profession in profession column and add a column named sterotypicality
        df["stereotypicality"] = df["profession"].apply(get_stereotypicality)
        #apply a threshold on the sterotypicality
        indexS = df[(df['stereotypicality'] >= 0.5) & (df['stereotypicality'] <= -0.5) & (df['stereotypicality'] == 0)].index
        df.drop(indexS, inplace=True)

        subsetDataFrame = df[(df['stereotypicality'] > -0.5 ) & (df['stereotypicality'] < 0)]
        dfs.append(get_all_effects(subsetDataFrame, direction="female"))

        subsetDataFrame = df[(df['stereotypicality'] > 0) & (df['stereotypicality'] < 0.5)]

        dfs.append(get_all_effects(subsetDataFrame, direction="male"))


    #all the 17 csv files are concatenated horizontally
    overall_df = pd.concat(dfs)

    print("The mean of the bias in " + model_name + " is: ", overall_df['bias_ratio'].mean())
    print("The std of the bias in " + model_name + " is: ", overall_df['bias_ratio'].std())
    path_name_sara = os.path.join(folder_name, model_name + "_final_results.csv")
    overall_df.to_csv(path_name_sara)

if __name__ == "__main__":
    # set the seeds
    seed_val = 10
    random.seed(seed_val)
    np.random.seed(seed_val)

    if len(sys.argv) != 3:
        print("USAGE: python ", sys.argv[0], "<folder_name> <model_name>")
    # e.g., results/20211114...
    folder_name = sys.argv[1]
    # model type , bert, robert, ..
    model_name = sys.argv[2]
    main(folder_name, model_name)