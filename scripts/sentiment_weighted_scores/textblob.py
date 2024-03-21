import pandas as pd
import numpy as np


def get_textblob_weighted_polarity_scores(data, interval, processed = False, weight_measure = 'user_weighted'):
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    grouped_data = data.resample(interval)
    if weight_measure == 'user_weighted':
        weighted_average_score = get_user_weighted_textblob_score(grouped_data, type='polarity')
    elif weight_measure == 'mean_weighted':
        weighted_average_score = get_mean_weighted_textblob_score(grouped_data, type='polarity')
    elif weight_measure == 'follower_weighted':
        weighted_average_score = get_follower_weighted_textblob_score(grouped_data, type='polarity')
    if processed:
        result_df = pd.DataFrame({'date': weighted_average_score.index.values, 'textblob_processed_data_polarity_score': weighted_average_score.values})
    else:
        result_df = pd.DataFrame({'date': weighted_average_score.index.values, 'textblob_nonprocessed_data_polarity_score': weighted_average_score.values})
    return result_df


def get_textblob_weighted_subjectivity_scores(data, interval, processed = False, weight_measure = 'user_weighted'):
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    grouped_data = data.resample(interval)
    if weight_measure == 'user_weighted':
        weighted_average_score = get_user_weighted_textblob_score(grouped_data, type='subjectivity')
    elif weight_measure == 'mean_weighted':
        weighted_average_score = get_mean_weighted_textblob_score(grouped_data, type='subjectivity')
    elif weight_measure == 'follower_weighted':
        weighted_average_score = get_follower_weighted_textblob_score(grouped_data, type='subjectivity')
    if processed:
        result_df = pd.DataFrame({'date': weighted_average_score.index.values, 'textblob_processed_data_subjectivity_score': weighted_average_score.values})
    else:
        result_df = pd.DataFrame({'date': weighted_average_score.index.values, 'textblob_nonprocessed_data_subjectivity_score': weighted_average_score.values})
    return result_df


def get_mean_weighted_textblob_score(grouped_data, type):
    if type == 'subjectivity':
        weighted_average_score = (
            grouped_data.apply(lambda x: np.sum(x['textblob_subjectivity']/len(x)))
        )
    elif type == 'polarity':
        weighted_average_score = (
            grouped_data.apply(lambda x: np.sum(x['textblob_polarity']/len(x)))
        )
    return weighted_average_score
    
def get_user_weighted_textblob_score(grouped_data, type):
    if type == 'subjectivity':
        weighted_average_score = (
        grouped_data.apply(lambda x: np.sum(x['textblob_subjectivity'] * x['user_influence_score'])/x['user_influence_score'].sum())
    )
    elif type == 'polarity':
        weighted_average_score = (
            grouped_data.apply(lambda x: np.sum(x['textblob_polarity'] * x['user_influence_score'])/x['user_influence_score'].sum())
        )
    return weighted_average_score
    
def get_follower_weighted_textblob_score(grouped_data, type):
    if type == 'subjectivity':
        weighted_average_score = (
            grouped_data.apply(lambda x: np.sum(x['textblob_subjectivity'] * (np.log(x['user_followers']+1)+1))/(np.log(x['user_followers']+1)+1).sum())
        )
    elif type == 'polarity':
        weighted_average_score = (
            grouped_data.apply(lambda x: np.sum(x['textblob_polarity'] * (np.log(x['user_followers']+1)+1))/(np.log(x['user_followers']+1)+1).sum())
        )
    return weighted_average_score
