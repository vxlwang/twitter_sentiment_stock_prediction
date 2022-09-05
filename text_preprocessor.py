# imports
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def text_preprocess(data, new_col_name="cleaned_content"):
    """
    Cleans the content of Tweets by removing URLs, user handles, and retweets, then appends a new column which contains
    the cleaned version of the text

    :param data: Series or column containing the text to clean
    :param new_col_name: name of the new column
    """

    data[new_col_name] = data.replace(r"https?://[a-zA-Z0-9./]*", "", regex=True)
    data[new_col_name] = data[new_col_name].replace(r"\s?RT\s*@[a-zA-z0-9]*:\s", "", regex=True) # retweets
    data[new_col_name] = data[new_col_name].replace(r"(\s|^|\W)@[\w]*\)?", "", regex=True) # user handles/mentions
    data[new_col_name] = data[new_col_name].replace(r"([0-9]),([0-9])", r"\1\2", regex=True) # remove commas b/w numbers
    data[new_col_name] = data[new_col_name].replace(r"^\s\-\s", "", regex=True)
    data[new_col_name] = data[new_col_name].replace(r"\s+", " ", regex=True)
    data[new_col_name] = data[new_col_name].str.strip()

def vader_sentiment(data):
    """
    Get negative, positive, neutral scoring ranging from -1 (extreme negative) to +1 (extreme positive)
    """
    vader_score = SentimentIntensityAnalyzer()

    scores = []
    for index in range(data.shape[0]):
        # print(analyser.polarity_scores(sentiments_pd['text'][i]))
        compound = vader_score.polarity_scores(data[index])["compound"]
        pos = vader_score.polarity_scores(data[index])["pos"]
        neu = vader_score.polarity_scores(data[index])["neu"]
        neg = vader_score.polarity_scores(data[index])["neg"]

        scores.append({"Compound": compound,
                       "Positive": pos,
                       "Negative": neg,
                       "Neutral": neu
                       })

    # sentiments_score = pd.DataFrame.from_dict(scores)
    # cleaned_text = tweets.join(sentiments_score)

    return scores