## Movie Review Sentiment Analysis

This is a simple Python-based Movie Review Sentiment Analysis project that uses the Naive Bayes classifier and TF-IDF vectorization to predict the sentiment (positive or negative) of movie reviews. The dataset is assumed to be in CSV format with columns "Review" and "Sentiment."

## Table of Contents

- [Requirements](#requirements)
- [How it Works](#how-it-works)
- [Usage](#usage)

## Requirements

To run this project, you need the following dependencies:

- Python 3.x
- pandas
- scikit-learn
- 
You can install the required packages using the following command:

pip install pandas scikit-learn

## How it Works

The Movie Review Sentiment Analysis works as follows:

1- The dataset (CSV format) is loaded, containing the movie reviews and their corresponding sentiments (positive or negative).

2- The text data is preprocessed (optional) by converting it to lowercase.

3- The TF-IDF vectorizer is initialized and fit to the review text, transforming the text data into numerical vectors.

4- The data is split into training and validation sets using a 80-20 split.

5- A Naive Bayes classifier is trained on the training set.

6- The model is evaluated on the validation set, and accuracy along with a classification report is printed.

7- A function is defined to predict the sentiment of a new review using the trained model and TF-IDF vectorizer.

8- The first 10 test values along with their predicted sentiment and actual sentiment are printed.

9- Finally, the model is tested with a new review to predict its sentiment.

## Usage

1- Clone the repository or download the moviereviewsentimentanalysis.py and Dataset.csv files.

2- Make sure you have Python 3.x installed on your system.

3- Install the required dependencies by running pip install pandas scikit-learn.

4- Run the moviereviewsentimentanalysis.py script.

The script will load the dataset, preprocess the text data (if chosen), train the model, and evaluate its performance. Additionally, it will demonstrate sentiment prediction for a new review.
