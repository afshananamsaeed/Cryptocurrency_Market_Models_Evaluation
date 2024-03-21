import os
os.chdir('/home/ubuntu/Masters_Thesis/scripts/')
from data_modelling.nn_modelling_regression import GetNNModelledOutput
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    GetNNModelledOutput(sentiment_past = 15, price_past = 15, time_forecast=0).run_regression_modelling(dataset=1, weighted_score = 'user_weighted')
    GetNNModelledOutput(sentiment_past = 30, price_past = 30, time_forecast=0).run_regression_modelling(dataset=1, weighted_score = 'user_weighted')