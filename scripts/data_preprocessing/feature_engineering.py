from sentiment_analysis.flair import evaluate_flair_sentiments_with_score
from sentiment_analysis.spacytextblob import evaluate_spacytextblob_sentiments
from sentiment_analysis.textblob import evaluate_textblob_sentiments
from sentiment_analysis.vader import extract_vader_sentiment_scores
from sentiment_analysis.transformers import get_sentiments_from_transformers
from emotion_detection.transformer import get_emotions_from_transformers
from data_preprocessing.multiprocess import MultiProcessText
from tqdm import tqdm
import pandas as pd
import numpy as np
tqdm.pandas()

class FeatureEngineering:
    '''
    Main function for obtaining influencer score and sentiment features.
    The sentiment are obtained from 'get_sentiments_and_emotions' for different libraries.
    Influencer Score comes from 'get_influencer_score'.
    '''
    def __init__(self, data):
        self.data = data

    def make_user_age(self):
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['user_created'] = pd.to_datetime(self.data['user_created'])
        self.data["user_age"]  = (self.data["date"] - self.data["user_created"]).dt.days
        self.data.drop(columns={"user_created"}, inplace=True)
        return self.data

    @staticmethod
    def get_sentiments_and_emotions(data):
        # data = evaluate_flair_sentiments_with_score(data)
        # data = evaluate_spacytextblob_sentiments(data)
        # data = evaluate_textblob_sentiments(data)
        data = extract_vader_sentiment_scores(data)
        # data = get_sentiments_from_transformers(data, prob = True)
        # data = get_emotions_from_transformers(data, prob = True)
        return data

    def obtain_sentiments_and_emotions_via_multiprocess(self):
        self.data = MultiProcessText().parallelize_dataframe(self.data, self.get_sentiments_and_emotions)
        return self.data

    def obtain_sentiments_and_emotions_normally(self):
        self.data = self.get_sentiments_and_emotions(self.data)
        return self.data

    def normalise_influencer_score(self):
        self.data['user_influence_score'] = (self.data['user_influence_score'] - self.data['user_influence_score'].min())/(self.data['user_influence_score'].max() - self.data['user_influence_score'].min())

    def get_influencer_score(self):
        self.data['user_influence_score'] = ((self.data['user_followers']+1)/(np.log(self.data['user_friends']+1)+1))*(self.data['user_favourites']+1)*(self.data["user_verified"]+1)
        self.normalise_influencer_score()
        return self.data