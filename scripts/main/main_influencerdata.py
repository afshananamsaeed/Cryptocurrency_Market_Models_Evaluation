import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os
os.chdir('/home/ubuntu/Masters_Thesis/scripts')
import numpy as np
from data_preprocessing.data_cleaning import DataPreprocessing
from data_preprocessing.text_preprocessing import PreProcessTextInDataFrame
from data_preprocessing.feature_engineering import FeatureEngineering
from sentiment_weighted_scores.flair import create_flair_weighted_score
from sentiment_weighted_scores.spacytextblob import get_spacytextblob_weighted_polarity_scores
from sentiment_weighted_scores.spacytextblob import get_spacytextblob_weighted_subjectivity_scores
from sentiment_weighted_scores.textblob import get_textblob_weighted_polarity_scores
from sentiment_weighted_scores.textblob import get_textblob_weighted_subjectivity_scores
from sentiment_weighted_scores.vader import get_vader_weighted_sentiment_scores
from sentiment_weighted_scores.transformers import get_bert_weighted_sentiment_scores
from sentiment_weighted_scores.transformers import get_roberta_weighted_sentiment_scores

def process_batch(data):
    print("cleaning data...")
    print(f"Shape of data before data cleaning: {data.shape}")
    data = DataPreprocessing(data).run_preprocess()
    print(f"Shape of data after data cleaning: {data.shape}")
    
    print("Preprocessing text...")
    data_processed = PreProcessTextInDataFrame(data).preprocess_text_via_multiprocess()
    print(data_processed.shape)
    
    print("Obtaining Sentiments for processed text data...")
    data_processed = FeatureEngineering(data_processed).obtain_sentiments_and_emotions_normally()
    print(data_processed.info())
    
    print("Obtaining Sentiments for non processed text data...")
    data_nonprocessed = FeatureEngineering(data).obtain_sentiments_and_emotions_normally()
    print(data_nonprocessed.info())
    
    return data_processed, data_nonprocessed

def get_sentiment_data():
    print("loading data...")
    # filepath = "/home/ubuntu/Masters Thesis/data/influencer_data/influencer_data.csv"
    filepath = "/home/ubuntu/Masters Thesis/data/bot_data/bot_data.csv"
    BATCH_SIZE = 64
    
    batch_iterator = pd.read_csv(filepath, chunksize=BATCH_SIZE, lineterminator='\n')
    
    faulty_batches = []
    batches_wo_preprocess = []
    batches_with_preprocess = []
    
    for iter, batch_df in enumerate(batch_iterator):
        try:
            print(f"Processing Batch Number {iter}")
            processed, non_processed = process_batch(batch_df)
            batches_with_preprocess.append(processed)
            batches_wo_preprocess.append(non_processed)
        except KeyboardInterrupt:
            break
        except:
            print(batch_df)
            faulty_batches.append(batch_df)
            print(f"Error encountered in Batch Number {iter}")
            
    final_non_processed_df = pd.concat(batches_wo_preprocess)
    final_processed_df = pd.concat(batches_with_preprocess)
    return final_processed_df, final_non_processed_df

def get_weighted_scores(data, interval, processed):
    data_flair = create_flair_weighted_score(data, interval, processed)
    data_spacy_polarity = get_spacytextblob_weighted_polarity_scores(data, interval, processed)
    data_spacy_subjectivity = get_spacytextblob_weighted_subjectivity_scores(data, interval, processed)
    data_textblob_polarity = get_textblob_weighted_polarity_scores(data, interval, processed)
    data_textblob_subjectivity = get_textblob_weighted_subjectivity_scores(data, interval, processed)
    data_vader = get_vader_weighted_sentiment_scores(data, interval, processed)
    data_bert = get_bert_weighted_sentiment_scores(data, interval, processed)
    data_roberta = get_roberta_weighted_sentiment_scores(data, interval, processed)
    
    all_data = [data_flair, data_spacy_polarity, data_spacy_subjectivity, data_textblob_polarity, data_textblob_subjectivity, data_vader, data_bert, data_roberta]
    
    weighted_data = all_data[0]
    for data in all_data[1:]:
        weighted_data = pd.merge(weighted_data, data, on='date', how='inner')
    # weighted_data.to_csv(f"/home/ubuntu/Masters Thesis/data/influencer_data/weighted_scores_influencer_processed_{processed}_interval_{interval}.csv")
    weighted_data.to_csv(f"/home/ubuntu/Masters Thesis/data/bot_data/weighted_scores_bots_processed_{processed}_interval_{interval}.csv")

def get_weighted_scores_for_time_intervals(processed_data, non_processed_data):
    processed_df_influencer_score = FeatureEngineering(processed_data).get_influencer_score() 
    non_processed_df_influencer_score = FeatureEngineering(non_processed_data).get_influencer_score()
    
    datasets = [processed_df_influencer_score, non_processed_df_influencer_score]
    
    intervals = ['30T', '1H', '4H', '6H', 'D']
    
    for i, data in enumerate(datasets):
        if i == 0:
            processed = True
        else:
            processed = False
        for interval in intervals:
            get_weighted_scores(data, interval, processed)

if __name__ == "__main__":
    # final_processed_df, final_non_processed_df = get_sentiment_data()
    # final_processed_df.to_csv("/home/ubuntu/Masters Thesis/data/bot_data/influencer_data_processed_with_sentiments.csv")
    # final_non_processed_df.to_csv("/home/ubuntu/Masters Thesis/data/bot_data/influencer_data_non_processed_with_sentiments.csv")
    
    final_processed_df = pd.read_csv("/home/ubuntu/Masters Thesis/data/bot_data/influencer_data_processed_with_sentiments.csv", lineterminator='\n')
    final_non_processed_df = pd.read_csv("/home/ubuntu/Masters Thesis/data/bot_data/influencer_data_non_processed_with_sentiments.csv", lineterminator='\n')
    
    get_weighted_scores_for_time_intervals(final_processed_df, final_non_processed_df)