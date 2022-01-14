# This is main file for flask application
from flask import Flask, render_template, request, abort
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from model import *

app = Flask(__name__)  # intitialize the flaks app 

# Error handling if any exceptions occurs
@app.errorhandler(404)
def error_handling(error):
    return render_template('index.html', error=error)

# Base API
@app.route('/')
def index():
    return render_template('index.html')

# API for building product recommendation based on user sentiments
@app.route('/view')
def view():
    try:
        user_name = str(request.args.get('user_name')).lower()

        # Invoke function having Recommendation & Sentiment models logic to fetch top 5 products
        final_df = fetch_top_products(user_name)

        return render_template('index.html',tables=[final_df.to_html(classes='product')], titles = ['NAN','Top Recommended Products for the given user'])
    except:
        abort(404)

if __name__ == '__main__' :
    app.run(debug=True )  # this command will enable the run of flask app or api



