# import pandas as pd
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import classification_report
# import pickle

# # Load datasets
# game_data = pd.read_csv('../datasets/output.csv')
# review_data = pd.read_csv('../datasets/output_steamspy.csv')


# # Combine and clean datasets
# data = pd.concat([game_data, review_data], ignore_index=True)
# data = data[['name', 'content', 'is_positive']]  # Ensure these columns exist

# # Text preprocessing function
# def clean_text(text):
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
#     return text

# data['cleaned_review'] = data['content'].apply(clean_text)

# # Split data
# X = data['cleaned_review']
# y = data['sentiment']  # Assume sentiment is labeled as 'positive' or 'negative'
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Build and train model
# pipeline = make_pipeline(CountVectorizer(), MultinomialNB())
# pipeline.fit(X_train, y_train)

# # Save the model
# with open('../model/sentiment_model.pkl', 'wb') as f:
#     pickle.dump(pipeline, f)

# # Evaluate
# y_pred = pipeline.predict(X_test)
# print(classification_report(y_test, y_pred))


##---------------------------------------->>>
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import pickle

# Load datasets
game_data = pd.read_csv('../datasets/output.csv')
review_data = pd.read_csv('../datasets/output_steamspy.csv')

# Merge datasets to include the game name
game_data = game_data.merge(review_data[['appid', 'name']], left_on='app_id', right_on='appid', how='left')

# Combine and clean datasets
game_data = game_data.dropna(subset=['content'])  # Remove rows with missing content
data = game_data[['name', 'content', 'is_positive']]  # Keep the game name, review, and sentiment columns

# Text preprocessing function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

data['cleaned_review'] = data['content'].apply(clean_text)

# Encode sentiment labels
data['sentiment'] = data['is_positive'].apply(lambda x: 1 if x == 'Positive' else 0)

# Split data
X = data['cleaned_review']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train model
pipeline = make_pipeline(CountVectorizer(), MultinomialNB())
pipeline.fit(X_train, y_train)

# Save the model
with open('../model/sentiment_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Optional: Save cleaned dataset with game names for further use
data[['name', 'cleaned_review', 'sentiment']].to_csv('../datasets/cleaned_data.csv', index=False)

