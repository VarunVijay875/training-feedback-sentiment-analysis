## Training Feedback Sentiment Analysis

A Natural Language Processing (NLP) project to analyze and summarize open-ended feedback from participants of **Nest Digital Academy** training programs. This project uses sentiment analysis, text preprocessing, machine learning, and data visualization to uncover insights and highlight areas of improvement.

---

## Objective

The goal of this project is to automatically analyze qualitative feedback from course participants to:
- Identify the sentiment (positive, neutral, negative) behind each response
- Summarize frequently mentioned themes using word clouds
- Train a machine learning model to classify feedback based on sentiment
- Help Nest Digital Academy enhance the quality of its training programs

---

## Methodology

🔹 Data Collection
- Source: Excel file with multiple sheets
- Target columns:
  - *What do you like most about the training?*
  - *Any suggestions for the training?*

🔹 Preprocessing
- Lowercasing, punctuation and stopword removal
- Tokenization and cleaning of text responses
- Labeling empty responses as `"nil"`

🔹 Sentiment Analysis
- Tool: **VADER** from NLTK
- Compound sentiment score determines:
  - Positive (≥ 0.05)
  - Neutral (between -0.05 and 0.05)
  - Negative (≤ -0.05)

🔹 Machine Learning
- TF-IDF vectorization of cleaned text
- Logistic Regression classifier for sentiment prediction
- Evaluation: Accuracy, Precision, Recall, F1-Score

🔹 Visualizations
- Word clouds for both feedback questions
- Sentiment distribution summary
- Exported CSV with sentiment labels and scores

---

## Results

- Sentiment classification performed well on responses with diverse sentiment
- Word clouds clearly highlighted key terms and concerns
- Overall sentiment scores were calculated for each question

---

## Files in This Repository

| File                                                       | Description                                                             |
|------------------------------------------------------------|-------------------------------------------------------------------------|
| `sentiment_analysis.py`                                    | Main Python script for data processing, analysis, and visualization     |
| `Test Sheet.xlsx`                                          | Sample Excel file with feedback data (not uploaded here for privacy)    |
| `sentiment_results_what_do_you_like_most_about_the_training.csv` | Output CSV with cleaned text and sentiment labels for positive feedback |
| `sentiment_results_any_suggestions_for_the_training.csv`   | Output CSV with cleaned text and sentiment labels for suggestions       |
| `wordcloud_what_do_you_like_most_about_the_training.png`   | Word cloud generated from "What do you like most" feedback              |
| `wordcloud_any_suggestions_for_the_training.png`           | Word cloud generated from "Any suggestions" feedback                    |

---

## Requirements

Make sure to install the required Python packages:

pip install pandas nltk scikit-learn matplotlib seaborn wordcloud

Also, download required NLTK resources:

import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')

---

## How to Run

python sentiment_analysis.py

Make sure the `Test Sheet.xlsx` file is in the same directory.

---

## Future Improvements

- Add support for multilingual feedback
- Integrate deep learning models (e.g., BERT) for richer analysis
- Build a simple dashboard to visualize live feedback summaries

---

## Author

Varun Vijay

Built as part of my internship project at Nest Digital.
