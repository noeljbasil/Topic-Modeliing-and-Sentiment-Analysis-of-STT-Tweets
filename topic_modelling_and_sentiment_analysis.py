# Importing Libraries
from   datetime import date
from   google_drive_downloader import GoogleDriveDownloader as gdd
import pandas as pd
import pygsheets
import re
import time
import tweepy
from   dateutil.relativedelta import relativedelta
import math
from bertopic import BERTopic
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

"""### **Tweet Scrapping**"""

#program start time
start_time = time.time()

#initiating tweepy client
bearer_token = os.environ['Bearer_token']
client = tweepy.Client(bearer_token=bearer_token)

datetime = []
text = []
hashtags = []
like_count=[]
retweet_count=[]
reply_count=[]
quote_count=[]
retweeted = []
quoted = []
media=[]
lang=[]
url=[]
quoted_tweet_url=[]

print("\n================================")
print("Tweets collection started")
print("================================\n")

for tweet in tweepy.Paginator(client.get_users_tweets,id=28075780,
                              tweet_fields=['public_metrics','created_at','entities','lang','referenced_tweets'], 
                              media_fields=['media_key','type'], expansions='attachments.media_keys',
                              max_results=100).flatten():
    
    datetime.append(tweet.created_at)
    text.append(tweet.text)
    
    try:
        htag_list=[]
        for htag in tweet.entities['hashtags']:
            htag_list.append(htag['tag'])
    except:
        htag_list=[]
    hashtags.append(htag_list)
    
    like_count.append(tweet.public_metrics['like_count'])
    retweet_count.append(tweet.public_metrics['retweet_count'])
    reply_count.append(tweet.public_metrics['reply_count'])
    quote_count.append(tweet.public_metrics['quote_count'])
       
    try:
        if tweet.referenced_tweets[0]['type']=='retweeted':
            retweeted.append(1)
            quoted.append(0)
            quoted_tweet_url.append("")
        elif tweet.referenced_tweets[0]['type']=='quoted':
            retweeted.append(0)
            quoted.append(1)
            for url_iterator in tweet.entities['urls']:
                url_of_interest = url_iterator['expanded_url'] #last element of the url list has the url of the original quoted tweet
            quoted_tweet_url.append(url_of_interest)
        else:
            retweeted.append(0)
            quoted.append(0)
            quoted_tweet_url.append("")
    except:
        retweeted.append(0)
        quoted.append(0)
        quoted_tweet_url.append("")
            
    try:
        media.append(len(tweet.attachments['media_keys']))
    except:
        media.append(0)
    lang.append(tweet.lang)
    url.append("https://twitter.com/STOPTHETRAFFIK/status/"+str(tweet.id))

tweets_df = pd.DataFrame({'Datetime':datetime, 'Text':text, 'Hashtags':hashtags, 'Like Count':like_count, 'Retweet Count':retweet_count,'Reply Count':reply_count,'Quote Count':quote_count, 'Retweeted':retweeted, 'Quoted Tweet':quoted,'Media':media,'Language':lang,'URL':url,'Quoted Tweet url':quoted_tweet_url})

topic_data = tweets_df[tweets_df.Language == 'en'][['Datetime','Text','URL']]

print("\n================================")
print("Ran Successfully...")
print("================================\n")

"""### **Topic Modelling**"""

def cleanTxt(text):
    """Function to clean tweets"""
    text = re.sub('RT @[A-Za-z0–9]+_?[A-Za-z0–9]+:','',text)
    text = re.sub('@[A-Za-z0–9]+_?[A-Za-z0–9]+','',text)
    text = re.sub('#','',text)
    text = re.sub('“','',text)
    text = re.sub('”','',text)
    text = re.sub(',','',text)
    text = re.sub('📢','',text)
    text = re.sub('🤔','',text)
    text = re.sub('💬','',text)
    text = re.sub('http[^\s]+','',text)
    text = re.sub('[(0-9/0-9)]','',text)
    text = re.sub(r'\n',' ',text)
    return text

topic_data.loc[:,'Text'] = topic_data.loc[:,'Text'].apply(cleanTxt) 
topic_data_trimmed       = topic_data[topic_data['Text']!=' '][topic_data['Text']!='']

docs = list(topic_data.Text)

topic_model = BERTopic(verbose=True, embedding_model="all-distilroberta-v1", min_topic_size=35,  calculate_probabilities=True)

topics, probs = topic_model.fit_transform(docs)

topic_model.get_representative_docs()

topic_model.get_topics()

topic_model.visualize_topics()

topic_model.visualize_barchart()

topic_model.visualize_heatmap()

timestamps = topic_data.Datetime.to_list()

topics_over_time = topic_model.topics_over_time(docs, topics, timestamps, nr_bins=20)

topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=6)

def topic_label(topic):
  """Function to label the topics"""

  if topic == -1:
    label = 'Not topic defined'
  elif topic == 0:
    label = 'General Awareness'
  elif topic == 1:
    label = 'Job vacancy'
  elif topic == 2:
    label= 'Fashion Industry'
  elif topic == 3:
    label = 'Food Industry'

  return label

topic_data['Topic'] = topics
topic_data['Topic Label'] = topic_data.loc[:,'Topic'].apply(topic_label)

"""### **Sentiment Analysis**"""

def sentiment(text):
    """Function to assess sentiment"""
    sid_obj = SentimentIntensityAnalyzer()
    score   = sid_obj.polarity_scores(text)
    score_text = []
    return score['compound']

topic_data.loc[:,'Sentiment Score'] = topic_data.loc[:,'Text'].apply(sentiment) 
topic_data.loc[:,'Update Date']     = date.today().strftime("%d %b %Y")

# remember to share the google sheet file with the service account email id before running below code
# downloading the service account key from google drive
gdrive_id = os.environ['Google_drive_id']

gdd.download_file_from_google_drive(file_id=gdrive_id,
                                    dest_path='./secret_key.json',
                                    unzip=True)

#authenticating with google sheets with pygsheets
client = pygsheets.authorize(service_account_file='secret_key.json')

#open google sheet
gsheet_key = os.environ['Google_sheet_key']
google_sheet = client.open_by_key(gsheet_key)

all_tweets_sentianalysis = google_sheet.worksheet_by_title('Sentiment Analysis')

#clearing existing values from the sheets
all_tweets_sentianalysis.clear(start='A1', end=None, fields='*')

#writing dataframes into the sheets
all_tweets_sentianalysis.set_dataframe(topic_data, start=(1,1))

#program end time
print(f"Program ran for {round((time.time() - start_time)/60,2)} minutes.")