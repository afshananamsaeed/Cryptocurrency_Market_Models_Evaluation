This directory includes all the scripts for obtaining the results. The chronological order for running the scripts are:

- 'sentiment_engineering'/'main_influencerdata' - this script runs pre-processing steps, processes the text, and obtains the sentiments using the various sentiment libraries for processed and non-processed texts.
  - The step for obtaining sentiments using the 'FeatureEngineering' module can be done using multiprocessing or normally. It is recommended to not run Flair, BERT and Roberta using multiprocess, and to run the sentiment analysis libraries one by one by selecting(uncommenting) libraries in the 'get_sentiments_and_emotions' function of 'FeatureEngineering' module.
  - Change the names of the saving files accordingly

- 'main_sentiment_weighting_wobots' - This read the sentiment files, calculates influencer score and the sentiment scores for different libraries. It further stores the weighted sentiment scores for different frequencies.
  - Make sure to set the weight_measure for all sentiment score types. (a) 'user_weighted' calculates sentiment scores by aggregating over User Influence Scores, (b) 'follower_weighted' calculates sentiment scores using the number of followers and (c) 'mean_weighted' calculates sentiment scores by taking the mean of all sentiments in that interval.
  - Change the directory for storing accordingly.
  
- 'manual_output'/'manual_output_priceonly' - This trains the models using different learning rates.
  - Set the frequency ('30T', '1D'), label ('close'), sentiment_past, price_past and time_forecast global variables. These variables help vary the input dimensions.
  - The 'model_config' dictionary imports all neural network classes and their respective config files. To select the Neural Network, it can be selected from this directly.
  - Different non-processed and processed sentiment features can be selected from the 'nonprocessed_features' and 'processed_features' list.
  - The data is extracted using the 'get_modelling_data_and_output_for_regression' function in the 'PrepareModellingData' class of 'data_preparation' module. The data is extracted in the form of a dictionary, where the key represents the frequency and the values is a list of non-processed and processed sentiment score data frames like [non_processed_df, processed_df]. Select the dataframes accordingly for non-processed and processed.
  - The weighted score should be mentioned in the 'PrepareModellingData' class initialisation. It can include 'user_weighted', 'follower_weighted'. This extracts the correct sentiment scores.
  - Make sure to set the epochs = 100 and implement Early Stopping in the 'train_regression' function of the 'TrainNNModel' class in the 'NN_Training' module.
  
- 'selected_lr_tuning'/'selected_lr_tuning_priceonly' runs the same models using selected (best performing) LR values from the previous script output. This output has to be made manually and stored. Set the epoch = 500 and do not run Early Stopping in the 'train_regression' function of the 'TrainNNModel' class in the 'NN_Training' module. All other aspects are the same as the previous scripts.
