import pandas as pd
from utilities.data_cleaning import clean_data

def get_directory(interval, dataset=1, weighted_score = 'user_weighted', processed = True, wo_bots = True):
    '''
    Main function for returning directory for obtaining the weighted sentiment scores
    '''
    if wo_bots:
        bot_dir1, bot_dir2 = 'Without_bots', 'wo_bots'
    else:
        bot_dir1, bot_dir2 = 'With_bots', 'with_bots'
    main_directory = f'/mnt/dataset{dataset}/{weighted_score}_scores/{bot_dir1}/'
    file_name = f'weighted_scores_bitcoin_tweets_{bot_dir2}_interval_{interval}_processed_{processed}.csv'
    return main_directory+file_name

def get_file(interval, dataset=1, weighted_score = 'user_weighted', processed = True, wo_bots = True):
    directory = get_directory(interval, dataset, weighted_score, processed, wo_bots)
    file = pd.read_csv(directory, lineterminator = '\n')
    file = clean_data(file)
    return file

def load_weighted_sentiments_data(dataset, weighted_score):
    ### Without bots data ###
    # load all weighted sentiment data - nonprocessed
    weighted_sentiment_data_np_wob_1H = get_file(interval = '1H', dataset = dataset, weighted_score = weighted_score, processed = False)
    weighted_sentiment_data_np_wob_30T = get_file(interval = '30T', dataset = dataset, weighted_score = weighted_score, processed = False)
    weighted_sentiment_data_np_wob_4H = get_file(interval = '4H', dataset = dataset, weighted_score = weighted_score, processed = False)
    weighted_sentiment_data_np_wob_6H = get_file(interval = '6H', dataset = dataset, weighted_score = weighted_score, processed = False)
    weighted_sentiment_data_np_wob_D = get_file(interval = 'D', dataset = dataset, weighted_score = weighted_score, processed = False)
    
    # load all weighted sentiment data - processed
    weighted_sentiment_data_p_wob_1H = get_file(interval = '1H', dataset = dataset, weighted_score = weighted_score, processed = True)
    weighted_sentiment_data_p_wob_30T = get_file(interval = '30T', dataset = dataset, weighted_score = weighted_score, processed = True)
    weighted_sentiment_data_p_wob_4H = get_file(interval = '4H', dataset = dataset, weighted_score = weighted_score, processed = True)
    weighted_sentiment_data_p_wob_6H = get_file(interval = '6H', dataset = dataset, weighted_score = weighted_score, processed = True)
    weighted_sentiment_data_p_wob_D = get_file(interval = 'D', dataset = dataset, weighted_score = weighted_score, processed = True)
    
    data_dict = {'30T': [weighted_sentiment_data_np_wob_30T, weighted_sentiment_data_p_wob_30T], '1H': [weighted_sentiment_data_np_wob_1H, weighted_sentiment_data_p_wob_1H], '4H': [weighted_sentiment_data_np_wob_4H, weighted_sentiment_data_p_wob_4H], '6H': [weighted_sentiment_data_np_wob_6H, weighted_sentiment_data_p_wob_6H], '1D': [weighted_sentiment_data_np_wob_D, weighted_sentiment_data_p_wob_D]}
    return data_dict


def load_weighted_sentiments_november_dataset_data():
    # load spacyandtextblob processed data
    spacyandtextblob_p_1T = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/spacyandtextblob/weighted_scores_november_spacyandtextblob_processed_False_interval_1T.csv', lineterminator='\n')
    spacyandtextblob_p_30T = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/spacyandtextblob/weighted_scores_november_spacyandtextblob_processed_False_interval_30T.csv', lineterminator='\n')
    spacyandtextblob_p_1H = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/spacyandtextblob/weighted_scores_november_spacyandtextblob_processed_False_interval_1H.csv', lineterminator='\n')
    spacyandtextblob_p_4H = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/spacyandtextblob/weighted_scores_november_spacyandtextblob_processed_False_interval_4H.csv', lineterminator='\n')
    spacyandtextblob_p_6H = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/spacyandtextblob/weighted_scores_november_spacyandtextblob_processed_False_interval_6H.csv', lineterminator='\n')
    spacyandtextblob_p_1D = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/spacyandtextblob/weighted_scores_november_spacyandtextblob_processed_False_interval_D.csv', lineterminator='\n')
    
    # load vader processed data
    vader_p_1T = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/vader/weighted_scores_november_vader_processed_False_interval_1T.csv', lineterminator='\n')
    vader_p_30T = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/vader/weighted_scores_november_vader_processed_False_interval_30T.csv', lineterminator='\n')
    vader_p_1H = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/vader/weighted_scores_november_vader_processed_False_interval_1H.csv', lineterminator='\n')
    vader_p_4H = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/vader/weighted_scores_november_vader_processed_False_interval_4H.csv', lineterminator='\n')
    vader_p_6H = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/vader/weighted_scores_november_vader_processed_False_interval_6H.csv', lineterminator='\n')
    vader_p_1D = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/vader/weighted_scores_november_vader_processed_False_interval_D.csv', lineterminator='\n')
    
    # combine processed data
    weighted_sentiment_data_p_1T = pd.merge(right=spacyandtextblob_p_1T, left=vader_p_1T, on='date', how='outer')
    weighted_sentiment_data_p_30T = pd.merge(right=spacyandtextblob_p_30T, left=vader_p_30T, on='date', how='outer')
    weighted_sentiment_data_p_1H = pd.merge(right=spacyandtextblob_p_1H, left=vader_p_1H, on='date', how='outer')
    weighted_sentiment_data_p_4H = pd.merge(right=spacyandtextblob_p_4H, left=vader_p_4H, on='date', how='outer')
    weighted_sentiment_data_p_6H = pd.merge(right=spacyandtextblob_p_6H, left=vader_p_6H, on='date', how='outer')
    weighted_sentiment_data_p_1D = pd.merge(right=spacyandtextblob_p_1D, left=vader_p_1D, on='date', how='outer')
    
    # load spacyandtextblob non-processed data
    spacyandtextblob_np_1T = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/spacyandtextblob/weighted_scores_november_spacyandtextblob_processed_True_interval_1T.csv', lineterminator='\n')
    spacyandtextblob_np_30T = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/spacyandtextblob/weighted_scores_november_spacyandtextblob_processed_True_interval_30T.csv', lineterminator='\n')
    spacyandtextblob_np_1H = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/spacyandtextblob/weighted_scores_november_spacyandtextblob_processed_True_interval_1H.csv', lineterminator='\n')
    spacyandtextblob_np_4H = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/spacyandtextblob/weighted_scores_november_spacyandtextblob_processed_True_interval_4H.csv', lineterminator='\n')
    spacyandtextblob_np_6H = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/spacyandtextblob/weighted_scores_november_spacyandtextblob_processed_True_interval_6H.csv', lineterminator='\n')
    spacyandtextblob_np_1D = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/spacyandtextblob/weighted_scores_november_spacyandtextblob_processed_True_interval_D.csv', lineterminator='\n')
    
    # load vader non-processed data
    vader_np_1T = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/vader/weighted_scores_november_vader_processed_True_interval_1T.csv', lineterminator='\n')
    vader_np_30T = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/vader/weighted_scores_november_vader_processed_True_interval_30T.csv', lineterminator='\n')
    vader_np_1H = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/vader/weighted_scores_november_vader_processed_True_interval_1H.csv', lineterminator='\n')
    vader_np_4H = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/vader/weighted_scores_november_vader_processed_True_interval_4H.csv', lineterminator='\n')
    vader_np_6H = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/vader/weighted_scores_november_vader_processed_True_interval_6H.csv', lineterminator='\n')
    vader_np_1D = pd.read_csv('/mnt/november_data/weighted_sentiment_scores/vader/weighted_scores_november_vader_processed_True_interval_D.csv', lineterminator='\n')
    
    # combine non processed data
    weighted_sentiment_data_np_1T = pd.merge(right=spacyandtextblob_np_1T, left=vader_np_1T, on='date', how='outer')
    weighted_sentiment_data_np_30T = pd.merge(right=spacyandtextblob_np_30T, left=vader_np_30T, on='date', how='outer')
    weighted_sentiment_data_np_1H = pd.merge(right=spacyandtextblob_np_1H, left=vader_np_1H, on='date', how='outer')
    weighted_sentiment_data_np_4H = pd.merge(right=spacyandtextblob_np_4H, left=vader_np_4H, on='date', how='outer')
    weighted_sentiment_data_np_6H = pd.merge(right=spacyandtextblob_np_6H, left=vader_np_6H, on='date', how='outer')
    weighted_sentiment_data_np_1D = pd.merge(right=spacyandtextblob_np_1D, left=vader_np_1D, on='date', how='outer')
    
    # clean joined data
    weighted_sentiment_data_np_1T = clean_data(weighted_sentiment_data_np_1T)
    weighted_sentiment_data_np_1H = clean_data(weighted_sentiment_data_np_1H)
    weighted_sentiment_data_np_30T = clean_data(weighted_sentiment_data_np_30T)
    weighted_sentiment_data_np_4H = clean_data(weighted_sentiment_data_np_4H)
    weighted_sentiment_data_np_6H = clean_data(weighted_sentiment_data_np_6H)
    weighted_sentiment_data_np_1D = clean_data(weighted_sentiment_data_np_1D)
    
    weighted_sentiment_data_p_1T = clean_data(weighted_sentiment_data_p_1T)
    weighted_sentiment_data_p_1H = clean_data(weighted_sentiment_data_p_1H)
    weighted_sentiment_data_p_30T = clean_data(weighted_sentiment_data_p_30T)
    weighted_sentiment_data_p_4H = clean_data(weighted_sentiment_data_p_4H)
    weighted_sentiment_data_p_6H = clean_data(weighted_sentiment_data_p_6H)
    weighted_sentiment_data_p_1D = clean_data(weighted_sentiment_data_p_1D)
    
    # make output dictionary
    data_dict = {'1T':[weighted_sentiment_data_np_1T, weighted_sentiment_data_p_1T], '30T': [weighted_sentiment_data_np_30T, weighted_sentiment_data_p_30T], '1H': [weighted_sentiment_data_np_1H, weighted_sentiment_data_p_1H], '4H': [weighted_sentiment_data_np_4H, weighted_sentiment_data_p_4H], '6H': [weighted_sentiment_data_np_6H, weighted_sentiment_data_p_6H], '1D': [weighted_sentiment_data_np_1D, weighted_sentiment_data_p_1D]}
    return data_dict