📌 Project Overview
This project performs sentiment analysis on IMDb movie reviews using machine learning. The goal is to classify movie reviews as either positive or negative based on their textual content.

📊 Dataset Information
Source: Kaggle - IMDb Dataset (50,000 movie reviews)

Download Link: IMDB Dataset on Kaggle

Size: 50,000 rows

Columns:

review: Text of the movie review

sentiment: Sentiment label (positive/negative)

🎯 Project Goal
To build and train a machine learning model that can accurately classify movie review sentiments.

🔧 Technical Implementation
1. Data Loading & Exploration
Loaded the CSV dataset with 50,000 reviews

Explored data distribution and class balance

Conducted initial exploratory data analysis (EDA)

2. Data Preprocessing
Key preprocessing steps performed:

Text cleaning (removing HTML tags, special characters)

Lowercasing all text

Stopword removal

Handling missing values

3. Feature Engineering
TF-IDF vectorization for text-to-numeric conversion

Mapped 0 to negative and 1 to positive sentiment

4. Model Building
Model Used: Logistic Regression

Simple yet effective for binary classification

Good baseline model for text classification tasks

5. Model Performance
Test Accuracy: 0.89 (89%)

The model successfully classifies movie reviews with high accuracy

📈 Key Results
Achieved 89% accuracy on test data

Model demonstrates strong predictive capabilities

Logistic regression proved effective for this sentiment analysis task

🏗️ Project Structure
text
project/
│
├── data/
│   └── IMDB Dataset.csv
│
├── notebooks/
│   └── sentiment_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── model.py
│
├── models/
│   └── logistic_regression_model.pkl
│
└── README.md
🚀 How to Run
Clone the repository

Install requirements: pip install -r requirements.txt

Download the dataset from Kaggle link above

Run the preprocessing and modeling scripts

Evaluate the model performance

📝 Requirements
Python 3.7+

pandas

numpy

scikit-learn

matplotlib

seaborn

🔍 Key Insights
The dataset was well-balanced with equal positive and negative reviews

Text preprocessing significantly improved model performance

TF-IDF features worked really well

Logistic regression provided a good balance of performance and interpretability

🎯 Future Improvements
Experiment with more complex models (RNN, LSTM, BERT)

Try different feature extraction methods (Word2Vec, GloVe)

Implement hyperparameter tuning

Create a web interface for real-time predictions

Deploy as an API service

📚 References
Original Kaggle notebook: Sentiment Analysis of IMDB Movie Reviews

Scikit-learn documentation for Logistic Regression

👤 Author
Nabin Pun
