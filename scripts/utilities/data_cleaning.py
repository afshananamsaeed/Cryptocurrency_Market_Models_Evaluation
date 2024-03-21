import datetime
import pandas as pd

def clean_data(weighted_sentiment_data):
    weighted_sentiment_data = weighted_sentiment_data.dropna()
    try:
        weighted_sentiment_data = weighted_sentiment_data.drop(columns=['Unnamed: 0'])
    except:
        weighted_sentiment_data = weighted_sentiment_data.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'])
    return weighted_sentiment_data

def clean_bitcoin_hourly_data(bitcoin_hourly):
    bitcoin_hourly = bitcoin_hourly.rename(columns={'datetimes': 'date'})
    bitcoin_hourly['date'] = pd.to_datetime(bitcoin_hourly['date'])
    return bitcoin_hourly

def clean_bitcoin_minutely_data(bitcoin_minutely):
    bitcoin_minutely = bitcoin_minutely.rename(columns={'time': 'date'})
    bitcoin_minutely['date'] = bitcoin_minutely['date'].apply(datetime.datetime.utcfromtimestamp)
    return bitcoin_minutely