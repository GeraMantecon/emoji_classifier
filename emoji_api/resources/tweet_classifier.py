import re
import string
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification

class TweetClassifier:
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('models/tf_bert_sequence_tok')
        self.model = TFDistilBertForSequenceClassification.from_pretrained('models/tf_bert_sequence',num_labels=6)
        self.emoji_dict = {0:"\U0001f622", #sadness
                           1:'\U0001f620', #anger
                           2:"\U0001f602", #joy
                           3:'\U0001f60d', #love
                           4:'\U0001f632', #surprised
                           5:'\U0001f631'} #fear
    
    def clean_tweet(self, tweet):
        tweet = tweet.lower()
        punctuations = string.punctuation
        punctuations = punctuations.replace("'", "")
        translator = str.maketrans('','',punctuations)
        tweet = tweet.translate(translator)
        tweet = re.sub(r"\bim\b", "i\'m", tweet)
        tweet = re.sub(r"\bive\b", "i\'ve", tweet)
        tweet = re.sub(r"\bill\b", "i\'ll", tweet)
        tweet = re.sub(r"\bcant\b", "can\'t", tweet)
        tweet = re.sub(r"\bwont\b", "won\'t", tweet)
        tweet = re.sub(r"\bdidnt\b", "didn\'t", tweet)
        tweet = re.sub(r"\bcouldnt\b", "couldn\'t", tweet)
        tweet = re.sub(r"\bwouldnt\b", "wouldn\'t", tweet)
        tweet = re.sub(r"\bshouldnt\b", "should\'t", tweet)
        tweet = re.sub(r"\bcouldve\b", "could\'ve", tweet)
        tweet = re.sub(r"\bwouldve\b", "would\'ve", tweet)
        tweet = re.sub(r"\bshould've\b", "should\'ve", tweet)
        tweet = re.sub(r"\bhavent\b", "haven\'t", tweet)
        tweet = re.sub(r"\bhasnt\b", "hasn\'t", tweet)
        tweet = re.sub(r"\bisnt\b", "isn\'t", tweet)
        tweet = ' '.join([word for word in tweet.split() if word not in ['http','https','www','href']])
        return tweet

    def get_encoding(self, tweet):
        tokens = self.tokenizer(text=tweet,
                                add_special_tokens=True,
                                max_length=64,
                                truncation=True,
                                padding=True,
                                return_tensors='tf',
                                return_token_type_ids=False,
                                return_attention_mask=True)
        return tokens
    
    def predict(self, tweet):
        encodings = self.get_encoding(tweet)
        prediction = self.model.predict(({'input_ids':encodings['input_ids'],'attention_mask':encodings['attention_mask']}))
        emoji = self.emoji_dict[np.argmax(prediction.logits, axis=1)[0]]
        confidence = np.max(softmax(prediction.logits), axis=1)[0]
        return {'emoji':emoji, 'confidence':confidence}