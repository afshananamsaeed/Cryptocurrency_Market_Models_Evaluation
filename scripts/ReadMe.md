This directory contains all the main scripts for modelling and testing of the models. 
- 'data_extraction' contains the scripts for the extraction of the Historical Bitcoin Price Data from cryptocompare.com
- 'data_modelling' contains all the neural network architectures, as well as the 'Optuna_Tuning', 'EarlyStopping' modules for Optuna Tuning and Early Stopping respectively. It also contains the model training modules.
- 'data_preprocessing' has all pre-processing modules within it, which includes data cleaning, feature engineering, and text preprocessing. Influencer Score and Sentiment Calculations are done withing the 'FeatureEngineering' module.
- 'emotion_detection' has all the modules associated with detecting emotions of the social media texts.
- 'sentiment_analysis' has all the modules associated with detecting the sentiments of the social media texts.
- 'sentiment_analysis_weighted_scores' has modules associated with calculating the weighted scores for all sentiment libraries.
- 'utilities' has all extra functions which include data preparation and finalization of data for model training, data cleaning, visualisation functions, etc.
- 'main' has all the scripts that are needed to run for obtaining the outputs.