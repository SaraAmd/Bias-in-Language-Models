**Probing Gender Bias in BERT and RoBERT**

The Experiments are done on profession dataset. The jason file for this data set is in experiment_data
folder

You can run the experiments by runing the run.py file. To run the experiment for the models in the following table 
type
pyhton3 run.py -model model-name for example for roberta-base model:
run.py -model roberta-base

|model-name |model-name|
|-----|--------|
|bert-base-cased|bert-base-uncased|
| bert-large-cased     | bert-large-uncased |
| bert-base-multilingual-cased | bert-base-multinlingual-uncased |
|distilbert-base-cased | distilbert-base-uncased |
| distilbert-base-multilingual-cased | 
|roberta-base|roberta-large|
|roberta-large-mnli|distilroberta-base|
After runing the run.py the results of prompt tuning are saved in "results" folder.
For measuring the bias you need to run:

evaluation.py folder_name model-name

example:

python3 evaluation.py 20211206__bias_probing_roberta-base roberta-base

For visualsation have the model_name_final_results.csv in your directory and run the note book attached to this project