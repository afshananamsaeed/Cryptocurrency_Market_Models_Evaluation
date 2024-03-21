import pandas as pd

time_delta_future_dict = {'1T': [1,2,3,7], '30T': [0.5,1,1.5,3.5], '1H': [1,2,3,7], '4H': [4,8,12,16], '6H': [6,12,18,42], '1D': [1,2,3,7]}

class GetModellingData:
    '''
    Used for obtaining all the historical values of Bitcoin price and sentiment, as well as forecasted Bitcoin prices.
    Can be used for Regression, Regression percentage and Classification model inputs.
    The main functions include 'get_data_for_regression'.
    '''
    
    def __init__(self, data, bitcoin_data, frequency, metric = None, sentiment_past = 16, price_past = 16, time_forecast=0):
        self.data = data
        self.bitcoin_data = bitcoin_data[['date', 'close']]
        self.frequency = frequency
        self.metric = metric
        self.sentiment_past = sentiment_past
        self.price_past = price_past
        self.forecast_period = time_forecast
    
    def get_price_time_past(self):
        return self.time_past
    
    def get_sentiment_time_past(self):
        return self.sentiment_past
    
    def get_forecast_period(self):
        return self.forecast_period
    
    def get_data_for_regression(self):
        data_future = self.get_data_for_feature_analysis(self.forecast_period)
        data_future = GetModellingData.get_cleaned_feature_analysis_data(data_future, self.forecast_period)
        
        if self.sentiment_past!= -1:
            # get sentiment past
            data_past_sentiments = GetModellingData.get_past_time_deltas_for_price(self.data, self.frequency, self.sentiment_past)
            data_past_sentiments = GetModellingData.get_past_sentiments(data_past_sentiments, self.metric, self.sentiment_past)
            
            data_combined = pd.merge(left = data_future, right = data_past_sentiments, left_on='date', right_on='date', how='inner')
            data_mid = data_combined
        else:
            data_mid = data_future
            
        # get bitcoin past
        data_full = GetModellingData.get_past_time_deltas_for_price(data_mid, self.frequency, self.price_past)
        data_full = GetModellingData.get_past_bitcoin_prices(data_full, self.bitcoin_data, self.price_past)
        data_full = GetModellingData.drop_extra_columns(data_full, self.sentiment_past, self.price_past)
        return data_full
    
    def get_data_for_feature_analysis(self, forecast = 0):
        if forecast==0:
            data_new = GetModellingData.get_future_time_deltas_for_price(self.data, self.frequency)
            data_new = GetModellingData.get_future_bitcoin_price(data_new, self.bitcoin_data, forecast)
            return data_new
        elif forecast>0:
            data_new = GetModellingData.get_future_times_for_forecast_modelling(self.data, self.frequency, forecast)
            data_new = GetModellingData.get_future_bitcoin_price(data_new, self.bitcoin_data, forecast)
            return data_new
    
    def get_data_for_regression_percentage(self):
        full_data = self.get_data_for_regression()
        for i in range(0, self.forecast_period+1):
            if i==0:
                full_data[f'delta_close_t+0'] = GetModellingData.get_percentage_change(full_data['close_t-1'], full_data['close'])
            else:
                full_data[f'delta_close_t+{i}'] = GetModellingData.get_percentage_change(full_data['close_t-1'], full_data[f'close_t+{i}'])
        columns_to_drop = ['close']
        columns_to_drop.extend([f'close_t+{i}' for i in range(1, self.forecast_period+1)])
        full_data = full_data.drop(columns = columns_to_drop)
        return full_data

    def get_data_for_classification_multi_class(self):
        full_data = self.get_data_for_regression_percentage()
        for i in range(0, self.forecast_period+1):
            full_data[f'delta_close_t+{i}_class'] = full_data[f'delta_close_t+{i}'].apply(GetModellingData.convert_numbers_to_categorical_multi_class)
        columns_to_drop = [f'delta_close_t+{i}' for i in range(0, self.forecast_period+1)]
        full_data = full_data.drop(columns = columns_to_drop)
        return full_data
    
    def get_data_for_classification_binary_class(self):
        full_data = self.get_data_for_regression_percentage()
        for i in range(0, self.forecast_period+1):
            full_data[f'delta_close_t+{i}_class'] = full_data[f'delta_close_t+{i}'].apply(GetModellingData.convert_numbers_to_categorical_binary_class)
        columns_to_drop = [f'delta_close_t+{i}' for i in range(0, self.forecast_period+1)]
        full_data = full_data.drop(columns = columns_to_drop)
        return full_data

    @staticmethod
    def get_percentage_change(earlier_time_column, later_time_column):
        return ((later_time_column-earlier_time_column)/earlier_time_column)*100

    @staticmethod
    def convert_numbers_to_categorical_multi_class(value):
        if 0<=value<2:
            return '+2'
        elif 2<=value<5:
            return '+5'
        elif value>=5:
            return '+x'
        elif -2<value<0:
            return '-2'
        elif -5<value<=-2:
            return '-5'
        elif value<=-5:
            return '-x'

    @staticmethod
    def convert_numbers_to_categorical_binary_class(value):
        if value<0:
            return 0
        elif value>=0:
            return 1

    @staticmethod
    def get_cleaned_feature_analysis_data(data, forecast):
        if forecast == 0:
            data = data[['date', 'close', 'close_t+1', 'close_t+2', 'close_t+3', 'close_t+7']]
        elif forecast > 0:
            data = data[['date', 'close', *[f'close_t+{i}' for i in range(1, forecast+1)]]]
        return data

    @staticmethod
    def get_past_time_deltas_for_price(data, frequency, delta_range):
        for n in range(1, delta_range+1):
            if frequency[-2]=='D':
                data[f'time-{n}'] = pd.to_datetime(data['date']) - pd.Timedelta(days=n)
            elif frequency=='30T':
                data[f'time-{n}'] = pd.to_datetime(data['date']) - pd.Timedelta(hours=0.5*n)
            elif frequency == '1T':
                data[f'time-{n}'] = pd.to_datetime(data['date']) - pd.Timedelta(minutes=n)
            else:
                data[f'time-{n}'] = pd.to_datetime(data['date']) - pd.Timedelta(hours=n)
        return data

    @staticmethod
    def get_future_time_deltas_for_price(data, frequency):
        for i, n in zip([1,2,3,7], time_delta_future_dict[frequency]):
            if frequency=='1D':
                data[f'time+{i}'] = pd.to_datetime(data['date']) + pd.Timedelta(days=n)
            elif frequency == '1T':
                data[f'time+{i}'] = pd.to_datetime(data['date']) + pd.Timedelta(minutes=n)
            else:
                data[f'time+{i}'] = pd.to_datetime(data['date']) + pd.Timedelta(hours=n)
        return data
    
    @staticmethod
    def get_future_times_for_forecast_modelling(data, frequency, forecast):
        for i in range(1, forecast+1):
            if frequency=='1D':
                data[f'time+{i}'] = pd.to_datetime(data['date']) + pd.Timedelta(days=i)
            elif frequency == '1T':
                data[f'time+{i}'] = pd.to_datetime(data['date']) + pd.Timedelta(minutes=i)
            elif frequency == '30T':
                data[f'time+{i}'] = pd.to_datetime(data['date']) + pd.Timedelta(hours=0.5*i)
            else:
                data[f'time+{i}'] = pd.to_datetime(data['date']) + pd.Timedelta(hours=i)
        return data
    
    @staticmethod
    def get_past_sentiments(data, metric, delta_range):
        time_columns = [f'time-{i}' for i in range(1, delta_range+1)]
        columns_of_interest = ['date', metric]
        columns_of_interest.extend(time_columns)
        merged_data = data[columns_of_interest]
        to_merge_data = data[['date', metric]]
        for i in range(1,delta_range+1):
            merged_data = pd.merge(merged_data, to_merge_data, left_on=f'time-{i}', right_on = 'date', how='left', suffixes=('', f'_t-{i}'))
        merged_data = merged_data.drop(columns = [*time_columns, *[f'date_t-{i}' for i in range(1, delta_range+1)]])
        return merged_data

    @staticmethod
    def get_past_bitcoin_prices(data, bitcoin_data, delta_range):
        merged_data = data.copy()
        for i in range(1, delta_range+1):
            merged_data = pd.merge(merged_data, bitcoin_data, left_on=f'time-{i}', right_on = 'date', how='left', suffixes=('', f'_t-{i}'))
        merged_data = merged_data.drop(columns = [f'date_t-{i}' for i in range(1, delta_range+1)])
        return merged_data

    @staticmethod
    def get_future_bitcoin_price(sentiment_data, bitcoin_data, forecast):
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
        bitcoin_data.loc[:, 'date'] = pd.to_datetime(bitcoin_data['date'])
        merged_data = pd.merge(sentiment_data, bitcoin_data, left_on='date', right_on = 'date', how='left', suffixes=('', '_t'))
        if forecast == 0:
            future_time_list = [1,2,3,7]
        elif forecast > 0:
            future_time_list = [i for i in range(1, forecast+1)]
        for i in future_time_list:
            merged_data = pd.merge(merged_data, bitcoin_data, left_on=f'time+{i}', right_on = 'date', how='left', suffixes=('', f'_t+{i}'))
        merged_data = merged_data.drop(columns = [f'date_t+{i}' for i in future_time_list])
        merged_data = merged_data.drop(columns = [f'time+{i}' for i in future_time_list])
        return merged_data
    
    @staticmethod
    def drop_extra_columns(data, delta_sentiments, delta_time):
        if delta_sentiments>delta_time and delta_sentiments!=-1:
            r = delta_sentiments
        else:
            r = delta_time
        data = data.drop(columns = [f'time-{i}' for i in range(1, r+1)])
        return data
    
    @staticmethod
    def group_and_make_percentages(bitcoin_data, frequency):
        bitcoin_data = bitcoin_data[['date', 'close']].set_index('date')
        bitcoin_data[f'pct_change_{frequency}'] = bitcoin_data.resample(frequency)['close'].mean().pct_change()
        return bitcoin_data
