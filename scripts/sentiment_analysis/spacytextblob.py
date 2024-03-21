import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

def evaluate_spacytextblob_sentiments(df):
    df['spacytextblob_polarity'] = df['text'].apply(lambda x: nlp(x)._.blob.polarity)
    df['spacytextblob_subjectivity'] = df['text'].apply(lambda x: nlp(x)._.blob.subjectivity)
    return df

# def get_spacytextblob_sentiment_and_score(dataframe):
#     result_df = dataframe.apply(lambda row: pd.Series(evaluate_spacytextblob_sentiments(row), index=['spacytextblob_polarity', 'spacytextblob_subjectivity']), axis=1, result_type='expand')
#     data = pd.concat([dataframe, result_df], axis=1)
#     return data