import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import torch
import gc
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models_dict = {"bert": 'nateraw/bert-base-uncased-emotion', "roberta": "cardiffnlp/twitter-roberta-base-emotion", "roberta_multilabel": "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"}
model_labels = {"bert": ["prob_sadness", "prob_joy", "prob_love", "prob_anger", "prob_fear", "prob_surprise"], "roberta": ["prob_joy", "prob_optimism", "prob_anger", "prob_sadness"], "roberta_multilabel": ["prob_anger", "prob_anticipation", "prob_disgust", "prob_fear", "prob_joy", "prob_love", "prob_optimism", "prob_pessimism", "prob_sadness", "prob_surprise", "prob_trust"]}

def evaluate_emotions_using_specific_model(dataframe, model_name, prob=False):
    PRE_TRAINED_MODEL_NAME =  models_dict[model_name]
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME)
    text = list(dataframe['text'])
    model_inputs = tokenizer(text, padding=True, return_tensors="pt", truncation=True).to(device)
    model = model.to(device)
    output = model(**model_inputs)
    if prob:
        scores = output.logits.detach().cpu().numpy()
        result = softmax(scores, axis=-1)
    else:
        predicted_label_classes = output.logits.detach().cpu().argmax(-1)
        result = predicted_label_classes.flatten().tolist()
    torch.cuda.empty_cache()
    gc.collect()
    return result

def add_emotions_in_data(dataframe, model_name, results, prob):
    if prob:
        labels = model_labels[model_name]
        result_df = pd.DataFrame(results, columns=[f"{model_name}"+"_"+label for label in labels])
        dataframe = dataframe.reset_index(drop=True)
        result_df = result_df.reset_index(drop=True)
        dataframe = pd.concat([dataframe, result_df], axis=1)
    else:
        dataframe[f"{model_name}_emotions"] = results
    return dataframe

def get_emotions_from_transformers(dataframe, model_name = None, prob = False):
    if model_name is not None:
        output = evaluate_emotions_using_specific_model(dataframe, model_name, prob)
        dataframe = add_emotions_in_data(dataframe, model_name, output, prob)
    else:
        for key in models_dict.keys():
            output = evaluate_emotions_using_specific_model(dataframe, key, prob)
            dataframe = add_emotions_in_data(dataframe, key, output, prob)
    return dataframe
