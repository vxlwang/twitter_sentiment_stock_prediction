def text_preprocess(data, content_col="Tweet content"):
    """
    Cleans the content of Tweets by removing URLs, user handles, and retweets

    Parameters:
    -----------
    content_col: the column containing the text to clean
    """
    data["cleaned"] = data[content_col].replace(r"https?://[a-zA-Z0-9./]*", "", regex=True)
    data["cleaned"] = data["cleaned"].replace(r"\s?RT\s*@[a-zA-z0-9]*:\s", "", regex=True)
    data["cleaned"] = data["cleaned"].replace(r"(\s|^|\W)@[\w]*\)?", "", regex=True)
    data["cleaned"] = data["cleaned"].replace(r"^\s\-\s", "", regex=True)
    data["cleaned"] = data["cleaned"].replace(r"\s+", " ", regex=True)
    data["cleaned"] = data["cleaned"].str.strip()

def vader_sentiment:
    pass