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
Steps
Clone the repository:

bash
Copy code
git clone https://github.com/Ishaan12003/Stock-Market-Analysis.git
cd Stock-Market-Analysis
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows
Install the required libraries:

bash
Copy code
pip install -r requirements.txt
The requirements.txt file includes the following:

pandas: For data manipulation and analysis.
numpy: For numerical computations.
scikit-learn: For machine learning algorithms.
tensorflow: For building and training neural network models.
nltk: For natural language processing tasks.
praw: For scraping Reddit data using the Reddit API.
You also need to set up the Reddit API credentials:

Create an application on Reddit (visit Reddit's developer page).
Add your client_id, client_secret, and user_agent to the code where indicated.
Usage
Scraping Data from Reddit
The scrape_reddit.py script allows you to scrape Reddit posts from a specified subreddit:

bash
Copy code
python scrape_reddit.py --subreddit "ETFs" --limit 100
This will collect the top 100 posts from the specified subreddit and save the data to a CSV file.

Data Preprocessing
The preprocessing step includes:

Text cleaning (removing special characters, links, etc.)
Sentiment analysis using NLP techniques (using libraries such as TextBlob or VADER).
Feature engineering to extract meaningful data for training the model.
Training the Model
To train the machine learning models, run:

bash
Copy code
python train_model.py
This will:

Load the cleaned and processed data.
Split the data into training and testing sets.
Train both a Random Forest classifier and a Neural Network model.
Save the trained models for future use.
Model Evaluation
After training the models, their performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. You can visualize the performance comparison between the two models using the plot_model_comparison.py script.

bash
Copy code
python plot_model_comparison.py
Predictions
To make stock movement predictions based on new data:

bash
Copy code
python predict.py --new_data "path_to_new_data.csv"
This will load the trained models, process the new data, and predict stock movements.

File Structure
bash
Copy code
Stock-Market-Analysis/
├── data/
│   ├── raw_data.csv           # Raw data scraped from Reddit
│   └── processed_data.csv     # Preprocessed data used for training
├── models/
│   ├── random_forest_model.pkl # Trained Random Forest model
│   └── neural_network_model.h5 # Trained Neural Network model
├── scrape_reddit.py           # Script to scrape data from Reddit
├── train_model.py             # Script to train machine learning models
├── predict.py                 # Script to make predictions
├── plot_model_comparison.py   # Script to visualize model performance
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation (this file)
└── .gitignore                 # Git ignore file
Model Evaluation Metrics
Accuracy: Percentage of correct predictions out of the total predictions.
Precision: Ratio of true positives to the sum of true positives and false positives.
Recall: Ratio of true positives to the sum of true positives and false negatives.
F1-score: Harmonic mean of precision and recall.
Results
The Random Forest model provided an accuracy of X%, while the Neural Network model achieved Y% accuracy.
Precision and recall were evaluated, and the results showed that one model performed better for certain types of predictions.
Future Work
Multiple Data Sources: Integrating additional data sources, such as stock price data, news articles, or financial reports, to improve prediction accuracy.
Hyperparameter Tuning: Experimenting with different hyperparameters for both Random Forest and Neural Network models to improve performance.
Advanced NLP Techniques: Using more advanced NLP techniques such as BERT or GPT-3 for better sentiment analysis and context understanding.
Model Deployment: Deploying the model as a web service using platforms like Flask or FastAPI for real-time stock predictions.
