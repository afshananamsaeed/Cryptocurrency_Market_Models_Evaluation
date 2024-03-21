This directory contains all the final performance results.
- 100_epoch_results contain the results generated for all models when trained on 100 epochs with Early Stopping.
- 500_epoch_results contain the results generated for all models when trained on 500 epochs with no Early Stopping.
- Concatenated results contain the results concatenated together for all forecasting model types (W=0,1,7) and processing/non-processing counterparts.
- parameter_tuning contain the results for different Learning Rates. These models are trained on 100 epochs with Early Stopping. The best LR is selected from these results with minimum MAE and stored in 100_epoch_results directory.

Naming Conventions:
W = Forecasting Periods (0,1,7)
T = Historical Price Time Periods (5,15,30)
S = Historical Sentiment Score Time Periods (5,15,30)
- "selected_lr_output_price_only_forecast_{W}" is for 100 epochs- Only Price
- "selected_lr_output_SPT_{S}_{T}_{W}" is for 100 epochs - with sentiments
- "Final_model_output_price_only_forecast_{W}" is for 500 epochs- Only Price
- "Final_model_outputs_processed_False_SPT_{S}_{T}_{W}" is for 500 epochs, non-processed with Sentiments
- "Final_model_outputs_processed_True_SPT_{S}_{T}_{W}" is for 500 epochs, processed with Sentiments
- "output_manual_price_only_forecast_{W}" is price only parameter tuning results
- "output_manual_processed_False_SPT_{S}_{T}_{W}" is parameter tuning for non-processed sentiment scores inputs.
- "output_manual_processed_True_SPT_{S}_{T}_{W}" is parameter tuning for processed sentiment scores inputs.