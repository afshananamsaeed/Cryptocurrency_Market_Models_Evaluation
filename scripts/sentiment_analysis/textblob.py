from textblob import TextBlob

def evaluate_textblob_sentiments(df):
    df['textblob_subjectivity'] = df['text'].apply(lambda text: TextBlob(text).sentiment.subjectivity)
    df['textblob_polarity'] = df['text'].apply(lambda text: TextBlob(text).sentiment.polarity)
    return df

# def get_textblob_sentiment_and_score(dataframe):
#     sentiment_df = dataframe.apply(lambda row: pd.Series(evaluate_textblob_sentiments(row), index=['textblob_subjectivity', 'textblob_polarity']), axis=1, result_type='expand')
#     data = pd.concat([dataframe, sentiment_df], axis=1)
#     return data


