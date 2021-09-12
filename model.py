import pandas as pd
import joblib
from typing import List

def get_product_recommendation(username: str) -> (str, List):

    recommender = pd.read_pickle(r"pickle_files/recommender.pkl")
    lr_model = joblib.load(r"pickle_files/lr_model.pkl")
    df = pd.read_pickle(r"pickle_files/cleaned_df.pkl")
    product_positivity_dict = {}

    username = username.strip()

    if not username:
        error_message = "Kindly enter a valid username"
        print(error_message)
        return None, error_message

    original_username = username
    username = original_username.lower()

    if username not in recommender.index:
        error_message = "User does not exist in the database. Kindly try another username"
        print(error_message)
        return None, error_message

    # Get 20 products from recommender
    product_list = recommender.loc[username].sort_values(ascending=False)[0:20]
    for product in product_list.keys():
        # print(product)
        reviews_list = df[df["name"] == product][["reviews"]]
        # print(type(reviews_list))
        # Get sentiment of vectors from sentiment model
        sentiment_list = lr_model.predict(reviews_list)
        # print(sentiment_list)
        # print(lr_model.predict_proba(reviews_list))
        # positivity_rate = sum(sentiment_list*len(sentiment_list)) / len(sentiment_list)
        positivity_rate = sum(sentiment_list) / len(sentiment_list)
        # print(len(sentiment_list), positivity_rate)
        product_positivity_dict[product] = positivity_rate

    # print(product_positivity_dict)
    top_5_items = sorted(product_positivity_dict.keys(), key=lambda product: product_positivity_dict[product], reverse=True)[:5]
    return original_username, top_5_items

