import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import random


frequency_dict = {'D': ["Date", 15], 'W': ["Week", 4], 'M':["Month", 1]}

def create_tweet_count_by_date_plot(data, frequency, bot_removed = False):
    # frequency can be 'D', 'W', 'M'
    counts_per_date = data.resample(frequency, on='date').size()
    plt.figure(figsize=(10, 6))
    counts_per_date.plot(kind='bar')
    plt.xlabel('Date', fontsize = 17)
    plt.ylabel('Count of Tweets', fontsize = 17)
    if bot_removed:
        plt.title(f'Time Series Plot of Tweets Counts per {frequency_dict[frequency][0]} after removing Bot tweets', fontsize = 20)
    else:
        plt.title(f'Time Series Plot of Tweets Counts per {frequency_dict[frequency][0]} without removing Bot tweets', fontsize = 20)
    plt.xticks(rotation=45, ha='right')
    n = frequency_dict[frequency][1]  # Display every n-th label
    labels = [label.strftime('%b %Y') for i, label in enumerate(counts_per_date.index) if i % n == 0]
    plt.xticks(range(0, len(counts_per_date), n), labels)
    plt.tight_layout()
    plt.savefig(f'tweet_count_bot_removed_{bot_removed}.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

def create_unique_user_barplot(user_data, frequency, bot_removed = False):
    # frequency can be 'D', 'W', 'M'
    unique_users_per_date = user_data.drop_duplicates(subset=['date', 'user_name']).resample(frequency, on='date').size()
    plt.figure(figsize=(10, 6))
    unique_users_per_date.plot(kind='bar')
    plt.xlabel('Date', fontsize=17)
    plt.ylabel('Number of Unique Users', fontsize=17)
    if bot_removed:
        plt.title(f'Number of Unique Users who Tweeted per {frequency_dict[frequency][0]} after removing Bot tweets', fontsize = 20)
    else:
        plt.title(f'Number of Unique Users who Tweeted per {frequency_dict[frequency][0]} without removing Bot tweets', fontsize = 20)
    plt.xticks(rotation=45, ha='right')
    n = frequency_dict[frequency][1]  # Display every n-th label
    labels = [label.strftime('%b %Y') for i, label in enumerate(unique_users_per_date.index) if i % n == 0]
    plt.xticks(range(0, len(unique_users_per_date), n), labels)
    plt.tight_layout()
    plt.savefig(f'unique_user_count_bot_removed_{bot_removed}.pdf', bbox_inches='tight', facecolor='white')
    plt.show()
    
def create_unique_user_lineplot(user_data, frequency, bot_removed = False):
    # frequency can be 'D', 'W', 'M'
    unique_users_per_date = user_data.drop_duplicates(subset=['date', 'user_name']).resample(frequency, on='date').size()
    plt.figure(figsize=(10, 6))
    unique_users_per_date.plot(kind='line')
    plt.xlabel('Date')
    plt.ylabel('Number of Unique Users')
    if bot_removed:
        plt.title(f'Number of Unique Users who Tweeted per {frequency_dict[frequency][0]} after removing Bot Tweets')
    else:
        plt.title(f'Number of Unique Users who Tweeted per {frequency_dict[frequency][0]} without removing Bot Tweets')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def make_stacked_barplot_with_bins_by_date(user_data, metric, bins):
    fig, ax = plt.subplots(figsize=(50, 15))

    hist_data = []
    labels = []

    user_data['date'] = pd.to_datetime(user_data['date']).dt.date
    unique_users_per_date = user_data.drop_duplicates(subset=['date', 'user_name'])

    for date, group in user_data.groupby('date'):
        followers_histogram, _ = np.histogram(group[metric], bins=bins)
        hist_data.append(followers_histogram)
        labels.append(date)

    hist_data = np.vstack(hist_data)

    # Create a stacked bar chart
    bar_width = 0.35
    index = range(len(labels))

    for i in range(len(bins) - 1):
        ax.bar(index, hist_data[:, i], width=bar_width, label=f'Bin {bins[i]}-{bins[i+1]}', alpha=0.7)

    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Accounts')
    ax.set_title(f'Stacked Histogram of Binned {metric} for Each Day')
    ax.legend(title=f'{metric} Bins', loc='upper right')
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    plt.title(f"Count of Users in different {metric} buckets in time")

    n = 15  # Display every n-th label
    xlabels = [label for i, label in enumerate(labels) if i % n == 0]
    plt.xticks(range(0, len(labels), n), xlabels)
    plt.tight_layout()
    plt.show()


def make_lineplot_with_bins_by_date(user_data, metric, bin_count = None, bot_removed = False, include_bitcoin = False):
    
    if bin_count is not None:
        # create percentile bins
        percentiles = np.linspace(0, 100, bin_count+1)
        bins = np.percentile(user_data[metric], percentiles)
    else:
        percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
        bins = np.percentile(user_data[metric], percentiles)
    
    fig, ax = plt.subplots(figsize=(50, 15))

    hist_data = []
    labels = []

    user_data['date'] = pd.to_datetime(user_data['date']).dt.date
    unique_users_per_date = user_data.drop_duplicates(subset=['date', 'user_name'])

    for date, group in unique_users_per_date.groupby('date'):
        followers_histogram, _ = np.histogram(group[metric], bins=bins)
        hist_data.append(followers_histogram)
        labels.append(date)

    hist_data = np.vstack(hist_data)

    index = range(len(labels))

    for i in range(len(bins) - 1):
        ax.fill_between(index, hist_data[:, i], label=f'Bin {bins[i]}-{bins[i+1]}, percentile = {percentiles[i]}', alpha=0.7)

    if include_bitcoin:
        bitcoin_prices = pd.read_csv('/mnt/bitcoin_price_minutely_data_till_2023_03_30.csv', lineterminator = '\n')
        bitcoin_prices['date'] = bitcoin_prices['time'].apply(datetime.utcfromtimestamp)
        labels_df = pd.DataFrame({'date': labels})
        labels_df['date'] = pd.to_datetime(labels_df['date'])
        merged_df = pd.merge(labels_df, bitcoin_prices, on='date', how='left')
        ax2 = ax.twinx()  # Create a secondary y-axis
        ax2.plot(labels, merged_df['close'], color='red', label='Bitcoin Price')
        ax2.set_ylabel('Bitcoin Price')
        ax2.legend(loc='upper left')

    # Adding labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Accounts')
    ax.set_title('Stacked Histogram of Binned Followers for Each Day')

    # Add legend
    ax.legend(title=f'{metric} Bins', loc='upper right')
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    if bot_removed:
        plt.title(f"Count of Users in different {metric} buckets in time after removing Bot Tweets")
    else:
        plt.title(f"Count of Users in different {metric} buckets in time without removing Bot Tweets")

    n = 15  # Display every n-th label
    xlabels = [label for i, label in enumerate(labels) if i % n == 0]
    plt.xticks(range(0, len(labels), n), xlabels)
    plt.tight_layout()
    # Display the plot
    plt.show()
    
    
def make_fractional_lineplot_with_bins_by_date(user_data_fn, metric, bin_count = None, bot_removed = False):
    
    if bin_count is not None:
        # create percentile bins
        percentiles = np.linspace(0, 100, bin_count+1)
        bins = np.percentile(user_data_fn[metric], percentiles)
    else:
        percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
        bins = np.percentile(user_data_fn[metric], percentiles)
    
    fig, ax = plt.subplots(figsize=(50, 15))

    hist_data = []
    labels = []

    user_data_fn['date'] = pd.to_datetime(user_data_fn['date']).dt.date
    unique_users_per_date = user_data_fn.drop_duplicates(subset=['date', 'user_name'])

    for date, group in unique_users_per_date.groupby('date'):
        followers_histogram, _ = np.histogram(group[metric], bins=bins)
        hist_data.append(followers_histogram)
        labels.append(date)

    hist_data = np.vstack(hist_data)
    row_sum = hist_data.sum(axis=1, keepdims=True)
    hist_data_fraction = hist_data/row_sum
    
    cumulative_fractions = np.cumsum(hist_data_fraction, axis=1)

    index = range(len(labels))
    baseline = np.zeros(len(index))
    for i in range(len(bins) - 2, 0, -1):
        ax.fill_between(index, baseline, cumulative_fractions[:, i], label=f'Bin {bins[i]}-{bins[i+1]}, percentile = {percentiles[i]}', alpha=0.7)

    # Adding labels and title
    ax.set_xlabel('Date', fontsize=40)
    ax.set_ylabel('Number of Accounts', fontsize=40)
    ax.set_title(f'Stacked Histogram of {metric} for Each Day', fontsize=40)

    # Add legend
    ax.legend(title=f'{metric} Bins', loc='upper right', fontsize = 25)
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=30)
    if bot_removed:
        plt.title(f"Count of Users in different {metric} buckets in time after removing Bot Tweets", fontsize=50)
    else:
        plt.title(f"Count of Users in different {metric} buckets in time without removing Bot Tweets", fontsize=50)

    n = 15  # Display every n-th label
    xlabels = [label for i, label in enumerate(labels) if i % n == 0]
    plt.xticks(range(0, len(labels), n), xlabels, fontsize = 20)
    plt.tight_layout()
    # Display the plot
    plt.savefig(f'Count_Users_{metric}_buckets_bot_removed_{bot_removed}.pdf', bbox_inches='tight', facecolor='white')
    plt.show()


def create_stacked_boolean_barplots(user_data, metric):

    fig, ax = plt.subplots(figsize=(15, 8))

    unique_users_per_date = user_data.drop_duplicates(subset=['date', 'user_name'])
    bar_width = 0.8
    unique_dates = unique_users_per_date['date'].unique()
    index = range(len(unique_users_per_date['date'].unique()))
    unique_values = unique_users_per_date[metric].unique()

    bottom = np.zeros(len(index))

    for value in unique_values:
        group = unique_users_per_date[unique_users_per_date[metric] == value]
        reindexed_group = group.groupby('date').size().reindex(unique_dates, fill_value=0)
        ax.bar(index, reindexed_group, width=bar_width, bottom=bottom, label=f'{metric}: {value}', alpha=0.7)
        bottom += reindexed_group

    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.set_title(f'Stacked Bar Plot of {metric} Over Time')
    ax.legend(title=f'{metric} Bins', loc='upper right')
    ax.set_xticks(index)
    ax.set_xticklabels(unique_dates, rotation=45, ha='right')
    plt.title(f"Count of {metric} buckets in time")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def create_boolean_lineplots(user_data, metric):

    fig, ax = plt.subplots(figsize=(15, 8))

    unique_users_per_date = user_data.drop_duplicates(subset=['date', 'user_name'])
    unique_dates = unique_users_per_date['date'].unique()
    index = range(len(unique_users_per_date['date'].unique()))
    unique_values = unique_users_per_date[metric].unique()

    bottom = np.zeros(len(index))

    for value in unique_values:
        group = unique_users_per_date[unique_users_per_date[metric] == value]
        reindexed_group = group.groupby('date').size().reindex(unique_dates, fill_value=0)
        ax.plot(index, reindexed_group, label=f'{metric}: {value}', alpha=0.7)
        bottom += reindexed_group

    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.set_title(f'Stacked Line Plot of {metric} Over Time')
    ax.legend(title=f'{metric} Bins', loc='upper right')
    ax.set_xticks(index)
    ax.set_xticklabels(unique_dates, rotation=45, ha='right')
    plt.title(f"Count of {metric} buckets in time")
    plt.xticks(rotation=45, ha='right')
    n = 15  # Display every n-th label
    xlabels = [label for i, label in enumerate(unique_dates) if i % n == 0]
    plt.xticks(range(0, len(unique_dates), n), xlabels)
    plt.tight_layout()
    plt.show()


def create_sentiment_weighted_plot(data):
    data['date'] = pd.to_datetime(data['date'])
    plt.figure(figsize=(100, 50))
    plt.plot(data['date'], data['spacytextblob_nonprocessed_data_polarity_score'], label = 'spacytextblob_nonprocessed_data_polarity_score')
    plt.plot(data['date'], data['spacytextblob_processed_data_polarity_score'], label = 'spacytextblob_processed_data_polarity_score')
    plt.plot(data['date'], data['spacytextblob_nonprocessed_data_subjectivity_score'], label = 'spacytextblob_nonprocessed_data_subjectivity_score')
    plt.plot(data['date'], data['spacytextblob_processed_data_subjectivity_score'], label = 'spacytextblob_processed_data_subjectivity_score')
    plt.plot(data['date'], data['textblob_nonprocessed_data_polarity_score'], label = 'textblob_nonprocessed_data_polarity_score')
    plt.plot(data['date'], data['textblob_processed_data_polarity_score'], label = 'textblob_processed_data_polarity_score')
    plt.plot(data['date'], data['textblob_nonprocessed_data_subjectivity_score'], label = 'textblob_nonprocessed_data_subjectivity_score')
    plt.plot(data['date'], data['textblob_processed_data_subjectivity_score'], label = 'textblob_processed_data_subjectivity_score')
    plt.plot(data['date'], data['vader_nonprocessed_data_score'], label = 'vader_nonprocessed_data_score')
    plt.plot(data['date'], data['vader_processed_data_score'], label = 'vader_processed_data_score')
    plt.plot(data['date'], data['bert_nonprocessed_data_sentiment_score'], label = 'bert_nonprocessed_data_sentiment_score')
    plt.plot(data['date'], data['roberta_nonprocessed_data_sentiment_score'], label = 'roberta_nonprocessed_data_sentiment_score')
    
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Sentiment Weight plots for different sentiment algorithms')
    plt.legend()
    plt.show()
    
    
    
def plot_sentiment_scores_with_prices(data_normal, data_influencer, bitcoin_data, metric, frequency, processed):
    # convert to datetime
    data_normal['date'] = pd.to_datetime(data_normal['date'])
    data_influencer['date'] = pd.to_datetime(data_influencer['date'])
    
    # introduce percentage change in bitcoin data by averaging over frequency
    bitcoin_data_grouped = bitcoin_data.resample(frequency, on='date').mean()
    bitcoin_data_grouped['pct_change'] = bitcoin_data_grouped['close'].pct_change() * 10
    bitcoin_data_grouped = bitcoin_data_grouped.reset_index(drop=False)
    
    # sort values
    common_dates = sorted(pd.concat([data_normal['date'], data_influencer['date']]).unique())
    bitcoin_sorted = bitcoin_data_grouped[bitcoin_data_grouped['date'].isin(common_dates)]
    bitcoin_sorted = bitcoin_sorted.sort_values(by='date')
    data_normal_joined = data_normal.sort_values(by='date')
    data_influencer_joined = data_influencer.sort_values(by='date')
    
    # combine the normal and influencer data with the bitcoin data
    data_normal_joined = pd.merge(data_normal[['date', metric]], bitcoin_sorted[['date']], on='date', how='right').fillna(0)
    data_influencer_joined = pd.merge(data_influencer[['date', metric]], bitcoin_sorted[['date']], on='date', how='right').fillna(0)
    
    plt.figure(figsize=(30, 6))
    plt.plot(data_normal_joined['date'], data_normal_joined[metric], label = "normal_data", color = 'green')
    plt.plot(data_influencer_joined['date'], data_influencer_joined[metric], label = 'influencer_data', color = 'blue')
    plt.plot(bitcoin_sorted['date'], bitcoin_sorted['pct_change'], label = "bitcoin_data_pct_change", color = 'red', linestyle='dotted')
    
    plt.xlabel("date")
    plt.ylabel("Value")
    plt.title(f"Plot for the {metric} sentiment metric with bitcoin close prices for frequency of {frequency} and processed = {processed} data")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_regression_train_losses(val_mse_loss, mae, mape, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Plot Validation MAE
    axes[0].plot(mae, "g-", label="Validation MAE")
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('MAE')
    axes[0].legend()
    axes[0].set_title('Validation MAE')

    # Plot Validation MSE
    axes[1].plot(val_mse_loss, "b-", label="Validation MSE")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel('MSE')
    axes[1].legend()
    axes[1].set_title('Validation MSE')
    
    # Plot Validation MAPE
    axes[2].plot(mape, "o-", label="Validation MAPE")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel('MAPE')
    axes[2].legend()
    axes[2].set_title('Validation MAPE')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved at: {save_path}")
    else:
        plt.show()

def plot_classification_train_loss_and_acc(val_loss, val_acc, f1_score, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Plot Validation Loss
    axes[0].plot(val_loss, "g-", label="Validation Loss")
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Validation Loss')

    # Plot Validation Accuracy
    axes[1].plot(val_acc, "b-", label="Validation Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Validation Accuracy')

    # Plot Validation F1 Score
    axes[2].plot(f1_score, "r-", label="Validation F1 Score")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].set_title('Validation F1 Score')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved at: {save_path}")
    else:
        plt.show()
        
def select_continuous_values(x_test, y_test, forecasting_period, longest = True):
    if forecasting_period==7:
        n = 16*(forecasting_period)
    elif forecasting_period == 1:
        n = 200*(forecasting_period)
    elif forecasting_period==0:
        n = 200
    continuous_segments = []
    start_index = None
    end_index = None
    for i in range(len(x_test.index)):
        if start_index is None:
            start_index = i
        elif x_test.index[i] - x_test.index[i-1] == pd.Timedelta(minutes=30):
            end_index = i
            if end_index - start_index + 1 == n:
                continuous_segments.append((start_index, end_index))
                start_index = None
                end_index = None
        else:
            start_index = None
            end_index = None
            
    if start_index is not None and end_index is not None and end_index - start_index + 1 >= n:
        continuous_segments.append((start_index, end_index))
    
    if longest:
        longest_segment = max(continuous_segments, key=lambda segment: segment[1] - segment[0])
        (start_index, end_index) = longest_segment
    else:
        random_number = 5
        (start_index, end_index) = continuous_segments[random_number]
    
    X, y = x_test.iloc[start_index:end_index+1], y_test.iloc[start_index: end_index+1]
    return X, y

def plot_regression_network_output(network, X_test_df, y_test_df, x_scaler, y_scaler, forecasting = False, forecasting_period = 0):
    network_name = network.__class__.__name__
    
    device = torch.device("cuda")
    test_dates = None
    ticks_to_show = None
    
    X_test_df, y_test_df = select_continuous_values(X_test_df, y_test_df, forecasting_period)
    test_dates = y_test_df.index
    if forecasting:
        selected_indexes = list(range(0, len(X_test_df), forecasting_period+1))
        X_test_df = X_test_df.iloc[selected_indexes]
        ticks_to_show = X_test_df.index
        ticks_to_show = [date.strftime("%Y-%m-%d %H:%M:%S") for date in ticks_to_show]
    
    x = torch.tensor(x_scaler.transform(X_test_df), dtype = torch.float32)
    y = torch.tensor(y_scaler.transform(y_test_df), dtype = torch.float32)
    
    testloader = torch.utils.data.DataLoader(list(zip(x, y)), shuffle=False, batch_size=1)
    y_actual = np.concatenate(y_test_df.values)
        
    test_dates_str = [date.strftime("%Y-%m-%d %H:%M:%S") for date in test_dates]

    # normal plot
    y_pred = []
    for x, _ in testloader:
        x = x.view(x.shape[0], 1, x.shape[1])
        z = network(x.to(device))
        y_pred.extend(y_scaler.inverse_transform(z.cpu().detach().numpy()))
    if forecasting:
        y_pred = np.concatenate(y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(test_dates_str, y_actual, linestyle='-', linewidth = 1, label='Actual')
    plt.plot(test_dates_str, y_pred, linestyle='-', linewidth = 1, label='Predicted')
    
    # Adding labels and title
    plt.xlabel('Dates')
    plt.ylabel('Values')
    plt.yticks(range(int(min(y_actual))-1000, int(max(y_actual)) + 1000, 500))
    if ticks_to_show is not None:
        if forecasting_period == 1:
            every_nth_index = range(0, len(ticks_to_show), 5)
        elif forecasting_period == 7:
            every_nth_index = range(0, len(ticks_to_show), 1)
        plt.xticks([ticks_to_show[i] for i in every_nth_index])
    else:
        every_5th_index = range(0, len(test_dates_str), 15)
        plt.xticks([test_dates_str[i] for i in every_5th_index])
    
    plt.grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Actual vs. Predicted Values for {network_name} model for W={forecasting_period+1}')
    plt.legend()
    plt.savefig(f'/home/ubuntu/Masters_Thesis/results/result_plots/Predictions_{network_name}_W{forecasting_period+1}.pdf', bbox_inches='tight', facecolor='white')
    return plt


def plot_regression_network_output_from_testloader(network, testloader, y_scaler, forecasting = False, forecasting_period = 0):
    network_name = network.__class__.__name__
    
    device = torch.device("cuda")
    test_dates = None
    ticks_to_show = None
    

    X_test_df, y_test_df = select_continuous_values(X_test_df, y_test_df, forecasting_period)
    test_dates = y_test_df.index
    selected_indexes = list(range(0, len(X_test_df), forecasting_period+1))
    X_test_df = X_test_df.iloc[selected_indexes]
    y_test_df = y_test_df.iloc[selected_indexes]
    if forecasting:
        ticks_to_show = X_test_df.index
    
    if test_dates is None:
        test_dates = y_test_df.index
        y_actual = y_test_df.values.tolist()
    else:
        y_actual = np.concatenate(y_test_df.values)
        
    test_dates_str = [date.strftime("%Y-%m-%d %H:%M:%S") for date in test_dates]

    # normal plot
    y_pred = []
    for x, _ in testloader:
        x = x.view(x.shape[0], 1, x.shape[1])
        z = network(x.to(device))
        y_pred.extend(y_scaler.inverse_transform(z.cpu().detach().numpy()))
    if forecasting:
        y_pred = np.concatenate(y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(test_dates_str, y_actual, linestyle='-', color='blue', linewidth = 1, label='Actual')
    plt.plot(test_dates_str, y_pred, linestyle='-', color='red', linewidth = 1, label='Predicted')
    
    # Adding labels and title
    plt.xlabel('Dates')
    plt.ylabel('Values')
    if ticks_to_show is not None:
        plt.xticks(ticks_to_show, ticks_to_show.strftime('%Y-%m-%d %H:%M:%S'))
    else:
        # every_5th_index = range(0, len(test_dates), 5)
        # plt.xticks(test_dates[every_5th_index], test_dates[every_5th_index].strftime('%Y-%m-%d %H:%M:%S'))
        every_5th_index = range(0, len(test_dates_str), 25)
        plt.xticks([test_dates_str[i] for i in every_5th_index])
    
    plt.grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Actual vs. Predicted Values for {network_name} model')
    plt.legend()

    return plt


def make_prob_density_plot_scatter(data, metric, bot_removed=False):
    user_data = data.groupby(['user_name']).max()
    followers_counts = user_data[metric].value_counts()
    followers_counts = followers_counts.sort_index()
    total_users = len(user_data)  # Total number of users

    # Calculate bin width
    bins = followers_counts.index
    bin_width = np.diff(bins)
    bin_width = np.append(bin_width, bin_width[-1]) 

    # Calculate probability density by dividing counts by total number of users and bin width
    probability_density = followers_counts / (total_users * bin_width)

    # Plot probability density as single points
    plt.scatter(followers_counts.index, probability_density, color='blue', label='Probability Density', s=3)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f'{metric}')
    plt.ylabel('Probability Density')
    if bot_removed:
        plt.title(f'Probability Density of {metric} after removing bot tweets')
    else:
        plt.title(f'Probability Density of {metric} before removing bot tweets')
    plt.legend()
    plt.show()