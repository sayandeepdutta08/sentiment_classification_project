# Sentiment Classification for Game Reviews

This project is a web application that classifies user reviews for video games as **Positive** or **Negative** using a machine learning model. The project utilizes a Naive Bayes classifier trained on game review data and provides an interactive interface for users to classify their reviews.

---

## Features

- **Sentiment Classification**: Predict whether a given review expresses positive or negative sentiment.
- **Game Selection**: Choose a game from a preloaded list to provide context for reviews.
- **User-Friendly Interface**: A Flask-based web application with an intuitive UI.

---

## Technologies Used

### Backend
- **Python**
  - `pandas`: For data preprocessing and manipulation.
  - `scikit-learn`: For building and training the sentiment classification model.
  - `Flask`: For creating the web application.
  - `pickle`: For saving and loading the trained model.

### Frontend
- **HTML** and **CSS**: For creating the user interface.

### Dataset
- **Game Reviews Dataset**:
  - Combined from `output.csv` and `output_steamspy.csv`.
  - Includes game names, reviews, and sentiment labels.

---

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries:
  ```bash
  pip install pandas scikit-learn flask
