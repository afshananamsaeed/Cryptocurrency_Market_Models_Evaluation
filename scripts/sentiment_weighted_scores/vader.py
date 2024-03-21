import pandas as pd
import numpy as np


def get_vader_weighted_sentiment_scores(data, interval, processed = False, weight_measure = 'user_weighted'):
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    grouped_data = data.resample(interval)
    if weight_measure == 'user_weighted':
        weighted_average_score = get_user_weighted_vader_score(grouped_data)
    elif weight_measure == 'mean_weighted':
        weighted_average_score = get_mean_weighted_vader_score(grouped_data)
    elif weight_measure == 'follower_weighted':
        weighted_average_score = get_follower_weighted_vader_score(grouped_data)
    if processed:
        result_df = pd.DataFrame({'date': weighted_average_score.index.values, 'vader_processed_data_score': weighted_average_score.values})
    else:
        result_df = pd.DataFrame({'date': weighted_average_score.index.values, 'vader_nonprocessed_data_score': weighted_average_score.values})
    return result_df

def get_mean_weighted_vader_score(grouped_data):
    weighted_average_score = (
        grouped_data.apply(lambda x: np.sum(x['vader_compound']/len(x)))
    )
    return weighted_average_score
    
def get_user_weighted_vader_score(grouped_data):
    weighted_average_score = (
        grouped_data.apply(lambda x: np.sum(x['vader_compound'] * x['user_influence_score'])/x['user_influence_score'].sum())
    )
    return weighted_average_score
    
def get_follower_weighted_vader_score(grouped_data):
    weighted_average_score = (
        grouped_data.apply(lambda x: np.sum(x['vader_compound'] * (np.log(x['user_followers']+1)+1))/(np.log(x['user_followers']+1)+1).sum())
    )
    return weighted_average_score