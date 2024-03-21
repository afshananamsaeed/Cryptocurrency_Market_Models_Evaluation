import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def extract_vader_sentiment_scores(df):
    sid_obj = SentimentIntensityAnalyzer()
    df['vader_sentiment'] = df['text'].progress_apply(sid_obj.polarity_scores)
    df['vader_neg'] = df['vader_sentiment'].progress_apply(lambda x: x['neg'])
    df['vader_pos'] = df['vader_sentiment'].progress_apply(lambda x: x['pos'])
    df['vader_neu'] = df['vader_sentiment'].progress_apply(lambda x: x['neu'])
    df['vader_compound'] = df['vader_sentiment'].progress_apply(lambda x: x['compound'])
    return df

# def get_vader_sentiments_and_score(dataframe):
#     result_df = dataframe.apply(lambda row: pd.Series(extract_vader_sentiment_scores(row), index=['vader_positive_score', 'vader_negative_score', 'vader_neutral_score', 'vader_compound_score']), axis=1, result_type='expand')
#     data = pd.concat([dataframe, result_df], axis=1)
#     return data

### The compound score tells us about the sentiment of the text
### pos, neg, neu tells us the percentage of text that falls in the given category
### don't clean the data when using vader- preferably