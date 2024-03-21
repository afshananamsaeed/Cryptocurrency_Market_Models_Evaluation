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


if __name__ == "__main__":
    print("Loading data...")
    flair_nonprocessed = pd.read_csv("/home/ubuntu/Masters Thesis/data/final/sentiment_data/with_bots/non-processed/Bitcoin_twitter_data_flair_sentiments_notextpreprocessing.csv", lineterminator='\n')
    spacytextblob_nonprocessed = pd.read_csv("/home/ubuntu/Masters Thesis/data/final/sentiment_data/with_bots/non-processed/Bitcoin_twitter_data_spacyandtextblob_sentiments_notextpreprocessing.csv", lineterminator='\n')
    vader_nonprocessed = pd.read_csv("/home/ubuntu/Masters Thesis/data/final/sentiment_data/with_bots/non-processed/Bitcoin_twitter_data_vader_sentiments_notextpreprocessing.csv", lineterminator='\n')
    transformers_nonprocessed = pd.read_csv("/home/ubuntu/Masters Thesis/data/final/sentiment_data/with_bots/non-processed/Bitcoin_twitter_sentiments_transformers_nontextprocess.csv", lineterminator='\n')
    flair_processed = pd.read_csv("/home/ubuntu/Masters Thesis/data/final/sentiment_data/with_bots/processed/Bitcoin_twitter_flair_sentiments_with_textpreprocessing.csv", lineterminator='\n')
    spacytextblob_processed = pd.read_csv("/home/ubuntu/Masters Thesis/data/final/sentiment_data/with_bots/processed/Bitcoin_twitter_spacyandtextblob_sentiments_with_textpreprocessing.csv", lineterminator='\n')
    vader_processed = pd.read_csv("/home/ubuntu/Masters Thesis/data/final/sentiment_data/with_bots/processed/Bitcoin_twitter_vader_sentiments_with_textpreprocessing.csv", lineterminator='\n')
    print("Loaded data...")
    
    # get influencer scores
    flair_nonprocessed = FeatureEngineering(flair_nonprocessed).get_influencer_score()
    spacytextblob_nonprocessed = FeatureEngineering(spacytextblob_nonprocessed).get_influencer_score()
    vader_nonprocessed = FeatureEngineering(vader_nonprocessed).get_influencer_score()
    transformers_nonprocessed = FeatureEngineering(transformers_nonprocessed).get_influencer_score()
    flair_processed = FeatureEngineering(flair_processed).get_influencer_score()
    spacytextblob_processed = FeatureEngineering(spacytextblob_processed).get_influencer_score()
    vader_processed = FeatureEngineering(vader_processed).get_influencer_score()
    print("Obtained influencer scores...")
    
    # intervals: 30 minutes, 1 hour, 4 hours, 6 hours and 1 day
    intervals = ['30T', '1H', '4H', '6H', 'D']
    
    # get weighted scores
    
    for interval in intervals:
        flair_non_processed_weights = create_flair_weighted_score(flair_nonprocessed, interval, processed=False)
        flair_processed_weights = create_flair_weighted_score(flair_processed, interval, processed=True)
        print("Obtained flair weights")
        
        spacytextblob_polarity_weights_nonprocessed = get_spacytextblob_weighted_polarity_scores(spacytextblob_nonprocessed, interval, processed=False)
        spacytextblob_polarity_weights_processed = get_spacytextblob_weighted_polarity_scores(spacytextblob_processed, interval, processed=True)
        spacytextblob_subjectivity_weights_nonprocessed = get_spacytextblob_weighted_subjectivity_scores(spacytextblob_nonprocessed, interval, processed=False)
        spacytextblob_subjectivity_weights_processed = get_spacytextblob_weighted_subjectivity_scores(spacytextblob_processed, interval, processed=True)
        print("Obtained spacytextblob weights")
        
        textblob_polarity_weights_nonprocessed = get_textblob_weighted_polarity_scores(spacytextblob_nonprocessed, interval, processed=False)
        textblob_polarity_weights_processed = get_textblob_weighted_polarity_scores(spacytextblob_processed, interval, processed=True)
        textblob_subjectivity_weights_nonprocessed = get_textblob_weighted_subjectivity_scores(spacytextblob_nonprocessed, interval, processed=False)
        textblob_subjectivity_weights_processed = get_textblob_weighted_subjectivity_scores(spacytextblob_processed, interval, processed=True)
        print("Obtained textblob weights")
        
        vader_non_processed_weights = get_vader_weighted_sentiment_scores(vader_nonprocessed, interval, processed=False)
        vader_processed_weights = get_vader_weighted_sentiment_scores(vader_processed, interval, processed=True)
        print("Obtained vader weights")
        
        bert_non_processed_weights = get_bert_weighted_sentiment_scores(transformers_nonprocessed, interval, processed=False)
        roberta_non_processed_weights = get_roberta_weighted_sentiment_scores(transformers_nonprocessed, interval, processed=False)
        print("Obtained transformers weights")
        
        # concatenate the data
        all_data = [flair_non_processed_weights, flair_processed_weights, spacytextblob_polarity_weights_nonprocessed, spacytextblob_polarity_weights_processed, spacytextblob_subjectivity_weights_nonprocessed, spacytextblob_subjectivity_weights_processed, textblob_polarity_weights_nonprocessed, textblob_polarity_weights_processed, textblob_subjectivity_weights_nonprocessed, textblob_subjectivity_weights_processed, vader_non_processed_weights, vader_processed_weights, bert_non_processed_weights, roberta_non_processed_weights]
        # all_data = [spacytextblob_polarity_weights_nonprocessed, spacytextblob_polarity_weights_processed, spacytextblob_subjectivity_weights_nonprocessed, spacytextblob_subjectivity_weights_processed, textblob_polarity_weights_nonprocessed, textblob_polarity_weights_processed, textblob_subjectivity_weights_nonprocessed, textblob_subjectivity_weights_processed, vader_non_processed_weights, vader_processed_weights, bert_non_processed_weights, roberta_non_processed_weights]
        weighted_data = all_data[0]
        for data in all_data[1:]:
            weighted_data = pd.merge(weighted_data, data, on='date', how='inner')
        weighted_data.to_csv(f"/home/ubuntu/Masters Thesis/data/final/sentiment_weighted_scores/weighted_scores_bitcoin_tweets_with_bots_interval_{interval}.csv")
