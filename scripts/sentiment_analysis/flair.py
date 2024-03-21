import pandas as pd
import numpy as np
from flair.data import Sentence
from flair.models import TextClassifier

def evaluate_flair_sentiments_with_score(df):
    classifier = TextClassifier.load('en-sentiment')
    df['sentence'] = df['text'].progress_apply(Sentence)
    sentences = df['sentence'].values.tolist()
    classifier.predict(sentences)
    df['flair_sentiment'] = df['sentence'].progress_apply(lambda x: x.labels[0].value)
    df['flair_sentiment_score'] = df['sentence'].progress_apply(lambda x: x.labels[0].score)
    df = df.drop(columns = ['sentence'])
    return df

# def get_flair_sentiments_and_score(dataframe):
#     result_df = dataframe.apply(lambda row: pd.Series(evaluate_flair_sentiments_with_score(row), index=['flair_sentiment', 'flair_sentiment_score']), axis=1, result_type='expand')
#     data = pd.concat([dataframe, result_df], axis=1)
#     return data