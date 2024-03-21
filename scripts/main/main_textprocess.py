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
    print("Preprocessing text...")
    # data_processed = PreProcessTextInDataFrame(data).preprocess_text_normally()
    data_processed = FeatureEngineering(data).obtain_sentiments_and_emotions_normally()
    return data_processed

if __name__ == "__main__":
    print("loading data...")
    filepath = "/home/ubuntu/Masters_Thesis/Data/final/raw_data/Bitcoin_twitter_data_english_transformertextprocessed_botremovedtweets.csv"
    
    BATCH_SIZE = 100
    
    batch_iterator = pd.read_csv(filepath, chunksize=BATCH_SIZE, lineterminator='\n')
    
    faulty_batches = []
    batches_wo_preprocess = []
    
    for iter, batch_df in enumerate(batch_iterator):
        try:
            print(f"Processing Batch Number {iter}")
            non_processed = process_batch(batch_df)
            batches_wo_preprocess.append(non_processed)
        except KeyboardInterrupt:
            break
        except:
            print(batch_df)
            faulty_batches.append(batch_df)
            print(f"Error encountered in Batch Number {iter}")

    print("Saving files...")
    final_non_processed_df = pd.concat(batches_wo_preprocess)
    final_non_processed_df.to_csv("/home/ubuntu/Masters Thesis/data/final/sentiment_data/without_bots/de-emojized_processed/flair_processed_demojised_wo_bots.csv")
    if len(faulty_batches)!=0:
        faulty_batches_df = pd.concat(faulty_batches)
        faulty_batches_df.to_csv("/home/ubuntu/Masters Thesis/data/final_datasets/problem_data/problem_text_flair_processeddemoji.csv")