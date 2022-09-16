import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import math
from sklearn import metrics

# Text Preprocess
def text_preprocess(data, new_col_name="cleaned_text"):
    """
    Cleans the content of Tweets by removing URLs, user handles, and retweets, then appends a new column which contains
    the cleaned version of the text
    """
    data[new_col_name] = data.replace(r"https?://[a-zA-Z0-9./]*", "", regex=True)
    data[new_col_name] = data[new_col_name].replace(r"\s?RT\s*@[a-zA-z0-9]*:\s", "", regex=True) # retweets
    data[new_col_name] = data[new_col_name].replace(r"(\s|^|\W)@[\w]*\)?", "", regex=True) # user handles/mentions
    data[new_col_name] = data[new_col_name].replace(r"([0-9]),([0-9])", r"\1\2", regex=True) # remove commas b/w numbers
    data[new_col_name] = data[new_col_name].replace(r"^\s\-\s", "", regex=True)
    data[new_col_name] = data[new_col_name].replace(r"\s+", " ", regex=True)
    data[new_col_name] = data[new_col_name].str.strip()

def vader_score(data):
    """
    Get negative, positive, neutral scoring
    """
    vader_score = SentimentIntensityAnalyzer()

    scores = []
    for index in range(data.shape[0]):
        compound = vader_score.polarity_scores(data[index])["compound"]
        pos = vader_score.polarity_scores(data[index])["pos"]
        neu = vader_score.polarity_scores(data[index])["neu"]
        neg = vader_score.polarity_scores(data[index])["neg"]

        scores.append({"Compound": compound,
                       "Positive": pos,
                       "Negative": neg,
                       "Neutral": neu
                       })

    sentiments_score = pd.DataFrame.from_dict(scores)

    return sentiments_score

def get_sentiment(series):
    """
    classify VADER polarity scores into positive, neutral, and negative moods
    """
    sentiment = []
    for score in series:
        if score >= 0.05:
            sentiment.append("Positive")
        elif score <= -0.05:
            sentiment.append("Negative")
        else:
            sentiment.append("Neutral")

    series["Sentiment"] = sentiment

    return series

# Regression
def regression_metrics(y_true, yhat):
    """
    return R^2, MSE, MAE, and RMSE metrics
    """

    print(f"R^2: {metrics.r2_score(y_true, yhat)}")
    print(f"Mean Squared Error: {metrics.mean_squared_error(y_true, yhat)}")
    print(f"Mean Absolute Error: {metrics.mean_absolute_error(y_true, yhat)}")
    print(f"Root MSE: {metrics.mean_squared_error(y_true, yhat, squared=False)}")
    print(f"% MAE: {metrics.mean_absolute_percentage_error(y_true, yhat)}")

# Modeling and Predictions
def time_series_split(data, train_size=0.8):
    """
    splits data while preserving order
    """
    split = math.ceil(len(data) * train_size)

    train = data[:split]
    test = data[split:]

    return train, test

