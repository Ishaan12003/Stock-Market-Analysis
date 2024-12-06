# Stock-Market-Analysis
This project predicts stock market movements by analyzing sentiment from Reddit posts. The model uses Natural Language Processing (NLP) techniques to process and analyze text data, followed by machine learning algorithms like Random Forest and Neural Networks to make stock movement predictions based on sentiment trends in the financial community.

The dataset is scraped from Reddit using the PRAW (Python Reddit API Wrapper) library, specifically targeting subreddits related to stock trading (e.g., ETFs). The data includes post titles, scores, upvote ratios, total comments, and other relevant metrics, which are used to derive insights into the market sentiment surrounding various stocks.

Features
Sentiment Analysis: Using NLP techniques to extract sentiment polarity from Reddit posts, which indicates whether the market sentiment is positive, negative, or neutral.
Stock Movement Prediction: Predicting stock price movements based on sentiment analysis using machine learning models such as Random Forest and Neural Networks.
Data Scraping: Reddit posts are scraped using the PRAW API for various financial subreddits.
Model Evaluation: The models are evaluated using accuracy, precision, and recall metrics to assess their performance.
Installation
To run this project, you'll need to set up a Python environment and install the necessary dependencies.

Prerequisites
Python 3.6 or higher
pip (Python package installer)


