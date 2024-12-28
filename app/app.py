# from flask import Flask, render_template, request
# import pandas as pd
# import pickle

# app = Flask(__name__)

# # Load data and model
# games = pd.read_csv('../datasets/output.csv')['game_name'].dropna().unique()
# with open('../model/sentiment_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# @app.route('/')
# def index():
#     return render_template('index.html', games=games)

# @app.route('/classify', methods=['POST'])
# def classify():
#     selected_game = request.form['game']
#     user_review = request.form['review']
#     prediction = model.predict([user_review])[0]
#     sentiment = 'Positive' if prediction == 'positive' else 'Negative'
#     return render_template('index.html', games=games, sentiment=sentiment, review=user_review, selected_game=selected_game)

# if __name__ == '__main__':
#     app.run(debug=True)

###----------------------------------------------------------->
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load data and model
# Load unique game names from the merged and cleaned dataset
games = pd.read_csv('../datasets/cleaned_data.csv')['name'].dropna().unique()

# Load the sentiment model
with open('../model/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html', games=games)

@app.route('/classify', methods=['POST'])
def classify():
    # Retrieve form data
    selected_game = request.form['game']
    user_review = request.form['review']
    
    # Predict sentiment
    prediction = model.predict([user_review])[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    
    # Pass data to the template
    return render_template('index.html', games=games, sentiment=sentiment, review=user_review, selected_game=selected_game)

if __name__ == '__main__':
    app.run(debug=True)

