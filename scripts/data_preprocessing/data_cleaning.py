# the current work is for the twitter data
import pandas as pd
import numpy as np
from langdetect import detect
# nltk.download('punkt')
from data_preprocessing.multiprocess import MultiProcessText
from tqdm import tqdm

tqdm.pandas()

class DataPreprocessing:
    '''
    Includes all processing steps except text pre-processing. 
    The main function is 'run_preprocess' which handles all pre-processing, including Bot removal.
    '''

    def __init__(self, dataframe):
        self.data = dataframe

    def run_preprocess(self):
        self.remove_extra_columns()
        self.change_to_datetime()
        self.remove_null_text_and_date_values()
        # self.get_english_tweets_via_multiprocess()
        self.data = self.remove_scam_phishing_bot_texts()
        self.data = self.data.reset_index(drop=True)
        return self.data

    def remove_null_text_and_date_values(self):
        self.data = self.data.dropna(subset=['text'])
        self.data = self.data.dropna(subset=['date'])

    def remove_extra_columns(self):
        self.data = self.data.drop(columns={"Unnamed: 0"})

    def change_to_datetime(self):
        try:
            data = self.data[(self.data['date'].apply(lambda x: isinstance(x, float))) | (self.data['date'].apply(lambda x: str(x).strip()).str.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', na=False)) | (self.data['date'].apply(lambda x: str(x).strip()).str.match(r'\d{4}-\d{2}-\d{2}', na=False))]
            data = data[data["user_created"].apply(lambda x: str(x).strip()).str.match((r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}') or (r'\d{4}-\d{2}-\d{2}'), na=False)]
            data["user_created"] = pd.to_datetime(data["user_created"])
            data["date"] = pd.to_datetime(data["date"])
            assert len(data) != 0
        except AssertionError:
            data = self.data.copy()
            data['date'] = pd.to_datetime(data['date'], format='%a %b %d %H:%M:%S +0000 %Y')
            data['user_created'] = pd.to_datetime(data['user_created'], format='%a %b %d %H:%M:%S +0000 %Y')
        self.data = data.copy()

    @staticmethod
    def is_english(text):
        try:
            language = detect(text)
            return language == 'en'
        except:
            return False

    def remove_non_english_language_tweets_for_mp(self, data):
        data = data[data["text"].apply(self.is_english)]
        return data
    
    def get_english_tweets_via_multiprocess(self):
        self.data = MultiProcessText().parallelize_dataframe(self.data, self.remove_non_english_language_tweets_for_mp)
        
    def get_english_tweets_vectorially(self):
        self.data = self.data[self.data["text"].progress_apply(self.is_english)]

    def remove_scam_phishing_bot_texts(self):
        #Heuristic 1
        heuristic_1_mask = self.data['text'].str.lower().str.contains('give away|giving away') #tweet based
        filtered_df = self.data[~heuristic_1_mask]
        
        #Heuristic 2
        heuristic_2_mask = self.data['text'].str.lower().str.contains('pump|register|join') #tweet based
        filtered_df = filtered_df[~heuristic_2_mask]
        
        #Heuristic 3
        filtered_df['ticker_count'] = filtered_df['text'].str.count(r'\$\w+') #tweet based
        filtered_df = filtered_df[filtered_df['ticker_count'] <= 14]
        filtered_df = filtered_df.drop(columns=['ticker_count'])

        #Heuristic 4
        filtered_df['hashtag_count'] = filtered_df['text'].str.count(r'\#\w+') #tweet based
        filtered_df = filtered_df[filtered_df['hashtag_count'] <= 14]
        filtered_df = filtered_df.drop(columns=['hashtag_count'])
        
        # #Heuristic 5
        # all_sources = list(set(self.data['source']))
        # all_sources.remove(np.nan)
        # bot_sources = [i for i in all_sources if 'bot' in i.lower()]
        # filtered_df = filtered_df[~filtered_df['source'].isin(bot_sources)] #data_based
        
        #Heuristic 6
        filtered_df = filtered_df[(filtered_df['user_friends']/filtered_df['user_followers'])<10] #data based        
        return filtered_df