import pandas as pd
import os
os.chdir('/home/ubuntu/Masters_Thesis/scripts')
from data_preprocessing.feature_engineering import FeatureEngineering
from sentiment_weighted_scores.flair import create_flair_weighted_score
from sentiment_weighted_scores.spacytextblob import get_spacytextblob_weighted_polarity_scores
from sentiment_weighted_scores.spacytextblob import get_spacytextblob_weighted_subjectivity_scores
from sentiment_weighted_scores.textblob import get_textblob_weighted_polarity_scores
from sentiment_weighted_scores.textblob import get_textblob_weighted_subjectivity_scores
from sentiment_weighted_scores.vader import get_vader_weighted_sentiment_scores
from sentiment_weighted_scores.transformers import get_bert_weighted_sentiment_scores
from sentiment_weighted_scores.transformers import get_roberta_weighted_sentiment_scores

# without bots and demojised processed tweets

if __name__ == "__main__":
    print("Loading data...")
    flair_nonprocessed = pd.read_csv("/mnt/dataset1/sentiment_data/without_bots/non-processed/flair_nonprocessed_wo_bots.csv", lineterminator='\n')
    spacytextblob_nonprocessed = pd.read_csv("/mnt/dataset1/sentiment_data/without_bots/non-processed/spacytextblob_nonprocessed_wo_bots.csv", lineterminator='\n')
    vader_nonprocessed = pd.read_csv("/mnt/dataset1/sentiment_data/without_bots/non-processed/vader_nonprocessed_wo_bots.csv", lineterminator='\n')
    transformers_nonprocessed = pd.read_csv("/mnt/dataset1/sentiment_data/without_bots/non-processed/transformers_nonprocessed_wo_bots.csv", lineterminator='\n')
    flair_processed = pd.read_csv("/mnt/dataset1/sentiment_data/without_bots/de-emojized_processed/flair_processed_demojised_wo_bots.csv", lineterminator='\n')
    spacytextblob_processed = pd.read_csv("/mnt/dataset1/sentiment_data/without_bots/de-emojized_processed/spacyandtextblob_processed_demojised_wo_bots.csv", lineterminator='\n')
    vader_processed = pd.read_csv("/mnt/dataset1/sentiment_data/without_bots/de-emojized_processed/vader_processed_demojised_wo_bots.csv", lineterminator='\n')
    transformers_processed = pd.read_csv("/mnt/dataset1/sentiment_data/without_bots/de-emojized_processed/Bitcoin_twitter_transformers_sentiments_withtextpreprocess_demojised_wobots.csv", lineterminator='\n')
    print("Loaded data...")
    
    # get influencer scores
    # flair_nonprocessed = FeatureEngineering(flair_nonprocessed).get_influencer_score()
    # spacytextblob_nonprocessed = FeatureEngineering(spacytextblob_nonprocessed).get_influencer_score()
    # vader_nonprocessed = FeatureEngineering(vader_nonprocessed).get_influencer_score()
    # transformers_nonprocessed = FeatureEngineering(transformers_nonprocessed).get_influencer_score()
    # flair_processed = FeatureEngineering(flair_processed).get_influencer_score()
    # spacytextblob_processed = FeatureEngineering(spacytextblob_processed).get_influencer_score()
    # vader_processed = FeatureEngineering(vader_processed).get_influencer_score()
    # transformers_processed = FeatureEngineering(transformers_processed).get_influencer_score()
    print("Obtained influencer scores...")
    
    # intervals: 30 minutes, 4 hours, 6 hours and 1 day
    intervals = ['30T', '1H', '4H', '6H', 'D']
    
    # get weighted scores
    
    for interval in intervals:
        flair_non_processed_weights = create_flair_weighted_score(flair_nonprocessed, interval, processed=False, weight_measure='follower_weighted')
        flair_processed_weights = create_flair_weighted_score(flair_processed, interval, processed=True, weight_measure='follower_weighted')
        print("Obtained flair weights")
        
        spacytextblob_polarity_weights_nonprocessed = get_spacytextblob_weighted_polarity_scores(spacytextblob_nonprocessed, interval, processed=False, weight_measure='follower_weighted')
        spacytextblob_polarity_weights_processed = get_spacytextblob_weighted_polarity_scores(spacytextblob_processed, interval, processed=True, weight_measure='follower_weighted')
        spacytextblob_subjectivity_weights_nonprocessed = get_spacytextblob_weighted_subjectivity_scores(spacytextblob_nonprocessed, interval, processed=False, weight_measure='follower_weighted')
        spacytextblob_subjectivity_weights_processed = get_spacytextblob_weighted_subjectivity_scores(spacytextblob_processed, interval, processed=True, weight_measure='follower_weighted')
        print("Obtained spacytextblob weights")
        
        textblob_polarity_weights_nonprocessed = get_textblob_weighted_polarity_scores(spacytextblob_nonprocessed, interval, processed=False, weight_measure='follower_weighted')
        textblob_polarity_weights_processed = get_textblob_weighted_polarity_scores(spacytextblob_processed, interval, processed=True, weight_measure='follower_weighted')
        textblob_subjectivity_weights_nonprocessed = get_textblob_weighted_subjectivity_scores(spacytextblob_nonprocessed, interval, processed=False, weight_measure='follower_weighted')
        textblob_subjectivity_weights_processed = get_textblob_weighted_subjectivity_scores(spacytextblob_processed, interval, processed=True, weight_measure='follower_weighted')
        print("Obtained textblob weights")
        
        vader_non_processed_weights = get_vader_weighted_sentiment_scores(vader_nonprocessed, interval, processed=False, weight_measure='follower_weighted')
        vader_processed_weights = get_vader_weighted_sentiment_scores(vader_processed, interval, processed=True, weight_measure='follower_weighted')
        print("Obtained vader weights")
        
        bert_non_processed_weights = get_bert_weighted_sentiment_scores(transformers_nonprocessed, interval, processed=False, weight_measure='follower_weighted')
        roberta_non_processed_weights = get_roberta_weighted_sentiment_scores(transformers_nonprocessed, interval, processed=False, weight_measure='follower_weighted')
        bert_processed_weights = get_bert_weighted_sentiment_scores(transformers_processed, interval, processed=True, weight_measure='follower_weighted')
        roberta_processed_weights = get_roberta_weighted_sentiment_scores(transformers_processed, interval, processed=True, weight_measure='follower_weighted')
        print("Obtained transformers weights")
        
        # concatenate the data
        all_data_nonprocessed = [flair_non_processed_weights, spacytextblob_polarity_weights_nonprocessed, spacytextblob_subjectivity_weights_nonprocessed, textblob_polarity_weights_nonprocessed, textblob_subjectivity_weights_nonprocessed, vader_non_processed_weights, bert_non_processed_weights, roberta_non_processed_weights]
        all_data_processed = [flair_processed_weights, spacytextblob_polarity_weights_processed, spacytextblob_subjectivity_weights_processed, textblob_polarity_weights_processed, textblob_subjectivity_weights_processed, vader_processed_weights, bert_processed_weights, roberta_processed_weights]
        
        weighted_data_np = all_data_nonprocessed[0]
        for data in all_data_nonprocessed[1:]:
            weighted_data_np = pd.merge(weighted_data_np, data, on='date', how='inner')
        weighted_data_np.to_csv(f"/mnt/dataset1/follower_weighted_scores/Without_bots/weighted_scores_bitcoin_tweets_wo_bots_interval_{interval}_nonprocessed.csv")

        weighted_data_processed = all_data_processed[0]
        for data in all_data_processed[1:]:
            weighted_data_processed = pd.merge(weighted_data_processed, data, on='date', how='inner')
        weighted_data_processed.to_csv(f"/mnt/dataset1/follower_weighted_scores/Without_bots/weighted_scores_bitcoin_tweets_wo_bots_interval_{interval}_processed.csv")
