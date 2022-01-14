# Import required libraries
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Reading all the required pickle files
pipeline = pickle.load(open('pickle/Item_Recommendation.pkl', 'rb'))
vector_pipeline = pickle.load(open('pickle/word_vectorizer.pkl', 'rb'))
sentiment_pipeline = pickle.load(open('pickle/xgboost_sentiment_model.pkl', 'rb'))

def fetch_top_products(user_name):
    # Reading existing csv data - this csv contains clean review text column data
    df_data = pd.read_csv('dataset/updated_sample30.csv')

    # Reading recommendation model & getting top 20 recommended products for a user
    user_top_products_data = pipeline.loc[user_name].sort_values(ascending=False)[0:20]
    user_top_products_data = user_top_products_data.to_frame()
    user_top_products_data = user_top_products_data.reset_index()

    # Computation for top 5 products based on sentiment analysis
    final_df = pd.DataFrame()
    top_products_df = pd.DataFrame()

    for prod_name in user_top_products_data['name']:
        row_data = df_data[df_data['name'] == prod_name]
        # Reading vectorize model
        vector_sentance = vector_pipeline.transform(row_data['clean_text'])
        # Reading sentiment model
        sentiment_prediction = sentiment_pipeline.predict(vector_sentance)
        # Computing percentage of positive reviews of top 20 products 
        percentage = (sentiment_prediction.tolist().count('Positive') / len(sentiment_prediction.tolist())) * 100
        data = {'positive_percentage':percentage, 'product_name': prod_name}
        top_products_df = top_products_df.append(data, ignore_index=True)
        top_products_df = top_products_df.sort_values(by=['positive_percentage'], ascending=False)

        final_df = top_products_df['product_name'][:5]
        final_df = final_df.to_frame()

    return final_df
