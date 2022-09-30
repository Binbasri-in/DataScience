"""
Question: Make a python script which fetches the App reviews(500 latest ones) 
for any app that you like and do a basic sentiment analysis on those reviews?
"""

# Importing the libraries
import pandas as pd
import numpy as np
import requests
import json
from app_store_scraper import AppStore

def main():
    # fetch the App reviews(500 latest ones) for any app that you like
    app_name = "splitwise"
    app_id = '458023433'
    app = fetch_reviews(app_name, app_id)
    print("reviews_count is: ",app.reviews_count)

    # create the data frame for the reviews
    df = pd.DataFrame(np.array(app.reviews),columns=['review'])

    df2 = df.join(pd.DataFrame(df.pop('review').tolist()))

    print(df2.head(1))

    # create a list of the latest 500 reviews and svae it in a csv file
    latest500 = df2.sort_values(by=['date'],ascending=False).head(500)
    latest500.to_csv(f'App Store Review {app_name}.csv')

    # do a basic sentiment analysis on those reviews
    basic_sentiment_analysis(latest500, app_name)
    print(latest500.head())




def fetch_reviews(app_name, app_id):
    """
    Fetch the App reviews(500 latest ones) for any app that you like
    """
    app = AppStore(country='in', app_name=app_name, app_id = app_id)
    app.review(how_many=2000)
    return app

def basic_sentiment_analysis(df, app_name):
    """
    Do a basic sentiment analysis on those reviews
    """
    # Importing the libraries
    import nltk
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # Creating a new column 'Compound' which contains the polarity scores
    df['Compound'] = df['review'].apply(lambda x: sid.polarity_scores(x)['compound'])

    # Creating a new column 'Sentiment' which classifies the polarity scores
    df['Sentiment'] = df['Compound'].apply(lambda x: 'Positive' if x >=0 else 'Negative')

    # Printing the first 10 rows of the data
    print(df.head(10))

    # Printing the percentage of positive and negative reviews
    print(df['Sentiment'].value_counts(normalize=True) * 100)

    df.to_csv(f'App Store Review {app_name} with Sentiment.csv')





if __name__ == "__main__":
    main()