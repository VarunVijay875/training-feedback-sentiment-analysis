import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import os

nltk.download('stopwords')
nltk.download('vader_lexicon')

stop_words = set(stopwords.words("english"))
sid = SentimentIntensityAnalyzer()

file = "Test Sheet.xlsx"
xls = pd.ExcelFile(file)

target_cols = [
    "What do you like most about the training?",
    "Any suggestions for the training?"
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    cleaned = " ".join(tokens)
    return cleaned if cleaned.strip() else "nil"

def get_sentiment_label_and_score(text):
    score = sid.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive', score
    elif score <= -0.05:
        return 'negative', score
    else:
        return 'neutral', score

col_name = "What do you like most about the training?"
print(f"\n--- Processing column: {col_name} ---")
all_text = []

for sheet in xls.sheet_names:
    df_sheet = xls.parse(sheet)
    if col_name in df_sheet.columns:
        df_col = df_sheet[[col_name]].copy()
        df_col.columns = ['text']
        df_col['text'] = df_col['text'].fillna('Nil').astype(str)
        all_text.append(df_col)

df = pd.concat(all_text, ignore_index=True)
df['cleaned_text'] = df['text'].apply(clean_text)
df[['sentiment', 'compound_score']] = df['text'].apply(lambda t: pd.Series(get_sentiment_label_and_score(t)))
print("Sentiment distribution:\n", df['sentiment'].value_counts())

col_key = col_name.replace(" ", "_").replace("?", "").lower()

if df['sentiment'].nunique() >= 2:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['sentiment'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
else:
    print("Skipping ML model training — only one sentiment class detected.")

filtered_words = " ".join(df[df['cleaned_text'] != "nil"]['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout()

wordcloud_filename = f"wordcloud_{col_key}.png"
plt.savefig(wordcloud_filename)
plt.show()

avg_score = df['compound_score'].mean()
overall_sentiment = 'positive' if avg_score >= 0.05 else 'negative' if avg_score <= -0.05 else 'neutral'
print(f"\nOverall Sentiment Score: {avg_score:.3f}")
print(f"Overall Sentiment: {overall_sentiment}")

results_filename = f"sentiment_results_{col_key}.csv"
df[['text', 'cleaned_text', 'sentiment', 'compound_score']].to_csv(results_filename, index=False)
print(f"Saved to:\n - {results_filename}\n - {wordcloud_filename}")

col_name = "Any suggestions for the training?"
print(f"\n--- Processing column: {col_name} ---")
all_text = []

for sheet in xls.sheet_names:
    df_sheet = xls.parse(sheet)
    if col_name in df_sheet.columns:
        df_col = df_sheet[[col_name]].copy()
        df_col.columns = ['text']
        df_col['text'] = df_col['text'].fillna('Nil').astype(str)
        all_text.append(df_col)

df = pd.concat(all_text, ignore_index=True)
df['cleaned_text'] = df['text'].apply(clean_text)
df[['sentiment', 'compound_score']] = df['text'].apply(lambda t: pd.Series(get_sentiment_label_and_score(t)))
print("Sentiment distribution:\n", df['sentiment'].value_counts())

col_key = col_name.replace(" ", "_").replace("?", "").lower()

if df['sentiment'].nunique() >= 2:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['sentiment'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
else:
    print("Skipping ML model training — only one sentiment class detected.")

filtered_words = " ".join(df[df['cleaned_text'] != "nil"]['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout()

wordcloud_filename = f"wordcloud_{col_key}.png"
plt.savefig(wordcloud_filename)
plt.show()

avg_score = df['compound_score'].mean()
overall_sentiment = 'positive' if avg_score >= 0.05 else 'negative' if avg_score <= -0.05 else 'neutral'
print(f"\nOverall Sentiment Score: {avg_score:.3f}")
print(f"Overall Sentiment: {overall_sentiment}")

results_filename = f"sentiment_results_{col_key}.csv"
df[['text', 'cleaned_text', 'sentiment', 'compound_score']].to_csv(results_filename, index=False)
print(f"Saved to:\n - {results_filename}\n - {wordcloud_filename}")
