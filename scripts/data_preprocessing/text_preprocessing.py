import pandas as pd
import re
import string
import emoji
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
from textblob import TextBlob
from nltk.corpus import stopwords
exclude = string.punctuation
from nltk.stem import WordNetLemmatizer
from data_preprocessing.multiprocess import MultiProcessText
from tqdm import tqdm
tqdm.pandas()

class PreProcessText:
    '''
    For preprocessing the social media texts. 
    Main function is the 'run_preprocess'
    '''
    def __init__(self, dataframe):
        self.df = dataframe

    def run_preprocess(self):
        # self.perform_spell_checks()
        # self.remove_emojis()
        self.convert_emoji_to_text()
        self.lowercase_text()
        self.remove_html_tags()
        self.remove_usermentions()
        self.remove_urls()
        self.remove_tabs_and_enters()
        self.remove_special()
        self.remove_pattern()
        self.remove_extra_space()
        self.df = self.df[~self.df['text'].isnull() | ~self.df['text'].eq('')]
        self.df = self.df.reset_index(drop=True)
        return self.df

    def remove_stopwords(self):
        self.df['text'] = self.df['text'].apply(lambda x: " ".join(word for word in x.split(" ") if word not in stopwords.words('english')) if x.strip() else x)

    def remove_emojis(self):
        emoj = re.compile("["
            u"\U00002700-\U000027BF"  # Dingbats
            u"\U0001F600-\U0001F64F"  # Emoticons
            u"\U00002600-\U000026FF"  # Miscellaneous Symbols
            u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                        "]+", re.UNICODE)
        self.df['text'] = self.df['text'].apply(lambda x: emoj.sub('', x))

    def convert_emoji_to_text(self):
        self.df['text'] = self.df['text'].apply(emoji.demojize)

    def remove_html_tags(self):
        self.df['text'] = self.df['text'].str.replace(r"(?:\@|https?\://)\S+", "", regex=True)

    def remove_urls(self):
        self.df['text'] = self.df['text'].str.replace(r'https?://\S+|www\.\S+', '', regex=True)

    def lowercase_text(self):
        self.df['text'] = self.df['text'].str.lower()

    def remove_special(self):
        self.df['text'] = self.df['text'].str.replace(r'[^a-zA-Z0-9\s]+', ' ', regex=True)

    def remove_pattern(self):
        self.df['text'] = self.df['text'].str.replace("@[A-Za-z0-9_]+","", regex=True)
        self.df['text'] = self.df['text'].str.replace("#[A-Za-z0-9_]+","", regex=True)

    def remove_extra_space(self):
        self.df['text'] = self.df['text'].str.replace(r'\s+', ' ', regex=True)

    def perform_spell_checks(self):
        self.df['text'] = self.df['text'].apply(lambda x: str(TextBlob(x).correct()))

    def perform_lemmatization(self):
        self.df['text'] = self.df['text'].apply(lambda x: " ".join([w.lemmatize() for w in TextBlob(x).words]))

    def remove_hashtag_words(self):
        self.df['text'] = self.df['text'].str.replace(r'\s*#(\w+)\b', '', regex=True)

    def remove_usermentions(self):
        self.df['text'] = self.df['text'].str.replace(r'@\w+', '', regex=True)
        
    def remove_tabs_and_enters(self):
        self.df['text'] = self.df['text'].str.replace(r'\t|\n', '', regex=True)

class PreProcessTextInDataFrame:
    def __init__(self, data):
        self.data = data
    
    def preprocess_text_column_for_mp(self, data):
        data["text"] = data["text"].apply(self.preprocess_text)
        return data

    @staticmethod
    def preprocess_text_for_mp(data):
        preprocessor = PreProcessText(data)
        data_processed = preprocessor.run_preprocess()
        # return preprocessor.df
        return data_processed

    def preprocess_text_via_multiprocess(self):
        self.data = MultiProcessText().parallelize_dataframe(self.data, self.preprocess_text_for_mp)
        return self.data

    def preprocess_text_normally(self):
        self.data = PreProcessText(self.data).run_preprocess()
        return self.data
