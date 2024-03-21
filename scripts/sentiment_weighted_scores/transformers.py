import pandas as pd
import numpy as np


def get_bert_weighted_sentiment_scores(data, interval, processed=False, weight_measure = 'user_weighted'):
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    grouped_data = data.resample(interval)
    if weight_measure == 'user_weighted':
        weighted_average_score = get_user_weighted_bert_score(grouped_data)
    elif weight_measure == 'mean_weighted':
        weighted_average_score = get_mean_weighted_bert_score(grouped_data)
    elif weight_measure == 'follower_weighted':
        weighted_average_score = get_follower_weighted_bert_score(grouped_data)
    if processed:
        result_df = pd.DataFrame({'date': weighted_average_score.index.values, 'bert_processed_data_sentiment_score': weighted_average_score.values})
    else:
        result_df = pd.DataFrame({'date': weighted_average_score.index.values, 'bert_nonprocessed_data_sentiment_score': weighted_average_score.values})
    return result_df

def get_mean_weighted_bert_score(grouped_data):
    weighted_average_score = (
        grouped_data.apply(lambda x: np.sum(((-1)*x['bert_prob_1'] + (-0.5)*x['bert_prob_2'] + (0.5)*x['bert_prob_4'] + (1)*x['bert_prob_5'])/len(x)))
    )
    return weighted_average_score
    
def get_user_weighted_bert_score(grouped_data):
    weighted_average_score = (
        grouped_data.apply(lambda x: np.sum(((-1)*x['bert_prob_1'] + (-0.5)*x['bert_prob_2'] + (0.5)*x['bert_prob_4'] + (1)*x['bert_prob_5']) * x['user_influence_score'])/x['user_influence_score'].sum())
    )
    return weighted_average_score
    
def get_follower_weighted_bert_score(grouped_data):
    weighted_average_score = (
        grouped_data.apply(lambda x: np.sum(((-1)*x['bert_prob_1'] + (-0.5)*x['bert_prob_2'] + (0.5)*x['bert_prob_4'] + (1)*x['bert_prob_5']) * (np.log(x['user_followers']+1)+1))/(np.log(x['user_followers']+1)+1).sum())
    )
    return weighted_average_score


def get_roberta_weighted_sentiment_scores(data, interval, processed = False, weight_measure = 'user_weighted'):
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    grouped_data = data.resample(interval)
    if weight_measure == 'user_weighted':
        weighted_average_score = get_user_weighted_roberta_score(grouped_data)
    elif weight_measure == 'mean_weighted':
        weighted_average_score = get_mean_weighted_roberta_score(grouped_data)
    elif weight_measure == 'follower_weighted':
        weighted_average_score = get_follower_weighted_roberta_score(grouped_data)
    if processed:
        result_df = pd.DataFrame({'date': weighted_average_score.index.values, 'roberta_processed_data_sentiment_score': weighted_average_score.values})
    else:
        result_df = pd.DataFrame({'date': weighted_average_score.index.values, 'roberta_nonprocessed_data_sentiment_score': weighted_average_score.values})
    return result_df

def get_mean_weighted_roberta_score(grouped_data):
    weighted_average_score = (
        grouped_data.apply(lambda x: np.sum(((-1)*x['roberta_prob_negative'] + (1)*x['roberta_prob_positive'])/len(x)))
    )
    return weighted_average_score
    
def get_user_weighted_roberta_score(grouped_data):
    weighted_average_score = (
        grouped_data.apply(lambda x: np.sum(((-1)*x['roberta_prob_negative'] + (1)*x['roberta_prob_positive']) * x['user_influence_score'])/x['user_influence_score'].sum())
    )
    return weighted_average_score
    
def get_follower_weighted_roberta_score(grouped_data):
    weighted_average_score = (
        grouped_data.apply(lambda x: np.sum(((-1)*x['roberta_prob_negative'] + (1)*x['roberta_prob_positive']) * (np.log(x['user_followers']+1)+1))/(np.log(x['user_followers']+1)+1).sum())
    )
    return weighted_average_score