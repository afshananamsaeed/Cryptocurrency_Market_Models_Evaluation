import warnings
warnings.filterwarnings('ignore')
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

models_dict = {"bert": 'nlptown/bert-base-multilingual-uncased-sentiment', "roberta": "cardiffnlp/twitter-roberta-base-sentiment"}
model_labels = {"bert": ["prob_1", "prob_2", "prob_3", "prob_4", "prob_5"], "roberta": ["prob_negative", "prob_neutral", "prob_positive"]}

def evaluate_sentiments_using_specific_model(dataframe, model_name, prob=False):
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

def add_sentiments_in_data(dataframe, model_name, results, prob):
    if prob:
        labels = model_labels[model_name]
        result_df = pd.DataFrame(results, columns=[f"{model_name}"+"_"+label for label in labels])
        dataframe = dataframe.reset_index(drop=True)
        result_df = result_df.reset_index(drop=True)
        dataframe = pd.concat([dataframe, result_df], axis=1)
    else:
        dataframe[f"{model_name}_sentiment"] = results
    return dataframe

def get_sentiments_from_transformers(dataframe, model_name = None, prob = False):
    if model_name is not None:
        output = evaluate_sentiments_using_specific_model(dataframe, model_name, prob)
        dataframe = add_sentiments_in_data(dataframe, model_name, output, prob)
    else:
        for key in models_dict.keys():
            output = evaluate_sentiments_using_specific_model(dataframe, key, prob)
            dataframe = add_sentiments_in_data(dataframe, key, output, prob)
    return dataframe
