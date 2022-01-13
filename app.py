# This is main file for flask application
from flask import Flask, render_template, request, redirect, url_for, abort
from scipy import sparse
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

app = Flask(__name__)  # intitialize the flaks app 

# Reading all the required pickle files
pipeline = pickle.load(open('pickle/Item_Recommendation.pkl', 'rb'))
vector_pipeline = pickle.load(open('pickle/word_vectorizer.pkl', 'rb'))
sentiment_pipeline = pickle.load(open('pickle/xgboost_sentiment_model.pkl', 'rb'))

# Error handling if any exceptions occurs
@app.errorhandler(404)
def error_handling(error):
    return render_template('error.html'), 404

# Base API
@app.route('/')
def index():
    return render_template('index.html')

# API for building product recommendation based on user sentiments
@app.route('/view')
def view():
    try:
        user_name = str(request.args.get('user_name')).lower()

        # Reading existing csv data - this csv contains clean review text column data
        df_data = pd.read_csv('dataset/updated_sample30.csv')
    
        # Getting top 20 recommended products for a user
        user_top_products_data = pipeline.loc[user_name].sort_values(ascending=False)[0:20]
        user_top_products_data = user_top_products_data.to_frame()
        user_top_products_data = user_top_products_data.reset_index()

        # Computation for top 5 products based on sentiment analysis
        final_df = pd.DataFrame()
        top_products_df = pd.DataFrame()

        for prod_name in user_top_products_data['name']:
            row_data = df_data[df_data['name'] == prod_name]
            vector_sentance = vector_pipeline.transform(row_data['clean_text'])
            sentiment_prediction = sentiment_pipeline.predict(vector_sentance)
            # Computing percentage of positive reviews of products 
            percentage = (sentiment_prediction.tolist().count('Positive') / len(sentiment_prediction.tolist())) * 100
            data = {'positive_percentage':percentage, 'product_name': prod_name}
            top_products_df = top_products_df.append(data, ignore_index=True)
            top_products_df = top_products_df.sort_values(by=['positive_percentage'], ascending=False)

            final_df = top_products_df['product_name'][:5]
            final_df = final_df.to_frame()

        return render_template('index.html',tables=[final_df.to_html(classes='product')], titles = ['NAN','Recommended Products for the given user'])
    except:
        abort(404)

if __name__ == '__main__' :
    app.run(debug=True )  # this command will enable the run of flask app or api






