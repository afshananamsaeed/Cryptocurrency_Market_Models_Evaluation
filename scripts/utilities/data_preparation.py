import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utilities.data_cleaning import clean_data, clean_bitcoin_hourly_data, clean_bitcoin_minutely_data
from utilities.modelling_data import GetModellingData
from utilities.weighted_data_cleaning import load_weighted_sentiments_data, load_weighted_sentiments_november_dataset_data

class PrepareModellingData:
    '''
    Returns the final modelling data for Regression, Regression Percentage and Classification (Binary/Multiclass) type of models.
    The Output of the main functions like 'get_modelling_data_and_output_for_regression' is used as the final input for train and testing the Neural Networks.
    '''
    def __init__(self, sentiment_past = 16, price_past = 16, time_forecast=0):
        self.label_encoder = None
        self.sentiment_past = sentiment_past
        self.price_past = price_past
        self.time_forecast = time_forecast

    def get_bitcoin_hourly_data(self):
        bitcoin_hourly = pd.read_csv("/mnt/bitcoin_hourly_prices_till_2023-03-30.csv", lineterminator='\n')
        bitcoin_hourly = clean_bitcoin_hourly_data(bitcoin_hourly)
        self.bitcoin_hourly = bitcoin_hourly

    def get_bitcoin_minutely_data(self):
        bitcoin_minutely = pd.read_csv("/mnt/bitcoin_price_minutely_data_till_2023_03_30.csv", lineterminator='\n')
        bitcoin_minutely = clean_bitcoin_minutely_data(bitcoin_minutely)
        self.bitcoin_minutely = bitcoin_minutely

    @staticmethod
    def load_data_wo_bots(dataset_no, weighted_score):
        '''
        Data returned in the form of a dictionary, 
        where keys are the frequency and values are [data_nonprocessed, data_processed] 
        for all sentiment score types.
        '''
        if dataset_no == 1 or dataset_no == 3:
            data_dict = load_weighted_sentiments_data(dataset=dataset_no, weighted_score=weighted_score)
        
        elif dataset_no == 2:
            data_dict = load_weighted_sentiments_november_dataset_data()
        
        return data_dict

    def encode_labels_for_categorical(self, y):
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(np.ravel(y))
        y = pd.Series(y)
        return y
    
    @staticmethod
    def adjust_training_columns(data, metric, dependent_variable, price_past, sentiment_past, time_forecast, type = 'regression'):
        sentiment_past_columns = [f'{metric}']
        sentiment_past_columns.extend([f'{metric}_t-{i}' for i in range (1, sentiment_past+1)])
        price_past_columns = [f'close_t-{i}' for i in range(1, price_past+1)]
        
        if type == 'regression':
            future_time_columns = [dependent_variable]
            if time_forecast != 0:
                future_time_columns.extend([f'close_t+{i}' for i in range(1, time_forecast+1)])
        elif type == 'regression_percentage':
            future_time_columns = [f'delta_close_t+{i}' for i in range(1, time_forecast+1)]
        elif type == 'classification':
            future_time_columns = [f'delta_close_t+{i}_class' for i in range(1, time_forecast+1)]
        
        try:
            if sentiment_past!=-1:
                output = data[[*sentiment_past_columns, *price_past_columns, *future_time_columns]]
                output = output.dropna()
                assert len(output)!=0
                X = output[[*sentiment_past_columns, *price_past_columns]]
            else:
                output = data[[*price_past_columns, *future_time_columns]]
                output = output.dropna()
                assert len(output)!=0
                X = output[[*price_past_columns]]
            y = output[future_time_columns]
        except:
            X = data[[f'{metric}', *price_past_columns]]
            y = data[future_time_columns]
        return X, y
    
    def get_modelling_data_and_output_for_regression(self, data, metric, frequency, dependent_variable):
        if self.time_forecast!=0:
            assert dependent_variable == 'close'
        if frequency=='30T':
            self.get_bitcoin_minutely_data()
            bitcoin_data = self.bitcoin_minutely
        else:
            self.get_bitcoin_hourly_data()
            bitcoin_data = self.bitcoin_hourly
        gmd = GetModellingData(data, bitcoin_data, frequency, metric, self.sentiment_past, self.price_past, self.time_forecast)
        modelling_data = gmd.get_data_for_regression()

        modelling_data = modelling_data.set_index('date')
        X, y = PrepareModellingData.adjust_training_columns(modelling_data, metric, dependent_variable, self.price_past, self.sentiment_past, self.time_forecast)
        return X, y

    def get_modelling_data_and_output_for_regression_percentage(self, data, metric, frequency, dependent_variable):
        if self.time_forecast!=0:
            assert dependent_variable == 'close'
        if frequency=='30T':
            self.get_bitcoin_minutely_data()
            bitcoin_data = self.bitcoin_minutely
        else:
            self.get_bitcoin_hourly_data()
            bitcoin_data = self.bitcoin_hourly
        gmd = GetModellingData(data, bitcoin_data, frequency, metric, self.sentiment_past, self.price_past, self.time_forecast)
        modelling_data = gmd.get_data_for_regression_percentage()

        modelling_data = modelling_data.set_index('date')
        X, y = PrepareModellingData.adjust_training_columns(modelling_data, metric, dependent_variable, self.price_past, self.sentiment_past, self.time_forecast, type = 'regression_percentage')
        return X, y

    def get_modelling_data_and_output_for_classification_multi_class(self, data, metric, frequency, dependent_variable):
        if frequency=='30T':
            self.get_bitcoin_minutely_data()
            bitcoin_data = self.bitcoin_minutely
        else:
            self.get_bitcoin_hourly_data()
            bitcoin_data = self.bitcoin_hourly
        
        gmd = GetModellingData(data, bitcoin_data, frequency, metric, self.sentiment_past, self.price_past, self.time_forecast)
        modelling_data = gmd.get_data_for_classification_multi_class()
        
        modelling_data = modelling_data.set_index('date')
        X, y = PrepareModellingData.adjust_training_columns(modelling_data, metric, dependent_variable, self.price_past, self.sentiment_past, self.time_forecast, type = 'classification')
        y = self.encode_labels_for_categorical(y)
        return X, y
    
    def get_modelling_data_and_output_for_classification_binary_class(self, data, metric, frequency, dependent_variable):
        if frequency=='30T':
            self.get_bitcoin_minutely_data()
            bitcoin_data = self.bitcoin_minutely
        else:
            self.get_bitcoin_hourly_data()
            bitcoin_data = self.bitcoin_hourly
        gmd = GetModellingData(data, bitcoin_data, frequency, metric, self.sentiment_past, self.price_past, self.time_forecast)
        modelling_data = gmd.get_data_for_classification_binary_class()
        
        modelling_data = modelling_data.set_index('date')
        X, y = PrepareModellingData.adjust_training_columns(modelling_data, metric, dependent_variable, self.price_past, self.sentiment_past, self.time_forecast, type = 'classification')
        # y = self.encode_labels_for_categorical(y)
        return X, y
    