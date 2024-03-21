import warnings
warnings.filterwarnings('ignore')
import os
os.chdir('/home/ubuntu/Masters_Thesis/scripts')
import pandas as pd
import numpy as np
from data_preprocessing.data_cleaning import DataPreprocessing
from data_preprocessing.text_preprocessing import PreProcessTextInDataFrame
from data_preprocessing.feature_engineering import FeatureEngineering
from tqdm import tqdm

def process_batch(data):
    print("cleaning data...")
    print(f"Shape of data before data cleaning: {data.shape}")
    data = DataPreprocessing(data).run_preprocess()
    print(f"Shape of data after data cleaning: {data.shape}")
    
    print("Preprocessing text...")
    data_processed = PreProcessTextInDataFrame(data).preprocess_text_via_multiprocess()
    print(data_processed.shape)
    
    print("Obtaining Sentiments for processed text data...")
    # data_processed = FeatureEngineering(data_processed).obtain_sentiments_and_emotions_via_multiprocess()
    data_processed = FeatureEngineering(data_processed).obtain_sentiments_and_emotions_normally()
    print(data_processed.info())
    
    print("Obtaining Sentiments for non processed text data...")
    data_nonprocessed = FeatureEngineering(data).obtain_sentiments_and_emotions_normally()
    print(data_nonprocessed.info())
    
    return data_nonprocessed, data_processed

if __name__ == "__main__":
    print("loading data...")
    # filepath = "/home/ubuntu/Masters Thesis/data/final_datasets/Bitcoin_twitter_data_english_non_textprocessed.csv"
    filepath = "/home/ubuntu/Masters Thesis/data/final_datasets/problem_data/problem_sentiment_transformer_wobots_textprocesseddemojised.csv"
    # filepath = "/home/ubuntu/Masters Thesis/data/final_datasets/Bitcoin_tweets_filtered.csv"
    # filepath = "/home/ubuntu/Masters Thesis/data/final/raw_data/Bitcoin_twitter_data_english_transformertextprocessed_botremovedtweets.csv"
    BATCH_SIZE = 32
    
    batch_iterator = pd.read_csv(filepath, chunksize=BATCH_SIZE, lineterminator='\n')
    
    faulty_batches = []
    batches_wo_preprocess = []
    batches_with_preprocess = []
    
    for iter, batch_df in enumerate(batch_iterator):
        try:
            print(f"Processing Batch Number {iter}")
            non_processed, processed = process_batch(batch_df)
            batches_with_preprocess.append(processed)
            batches_wo_preprocess.append(non_processed)
        except KeyboardInterrupt:
            break
        except:
            print(batch_df)
            faulty_batches.append(batch_df)
            print(f"Error encountered in Batch Number {iter}")

    print("Saving files...")
    
    final_non_processed_df = pd.concat(batches_wo_preprocess)
    final_processed_df = pd.concat(batches_with_preprocess)
    final_non_processed_df.to_csv("/home/ubuntu/Masters Thesis/data/final_datasets/Bitcoin_twitter_transformers_sentiments_withouttextpreprocess_demojised_wobots.csv")
    final_processed_df.to_csv("/home/ubuntu/Masters Thesis/data/final_datasets/Bitcoin_twitter_transformers_sentiments_withtextpreprocess_demojised_wobots.csv")
    # if len(faulty_batches)!=0:
    #     faulty_batches_df = pd.concat(faulty_batches)
    #     faulty_batches_df.to_csv("/home/ubuntu/Masters Thesis/data/final_datasets/problem_data/problem_sentiments_transformer_wobots_textprocesseddemojised1.csv")