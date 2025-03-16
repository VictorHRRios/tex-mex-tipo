from preprocessing import dataFrameParser
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from modeling import create_pipeline
import numpy as np

def main():
    url = "https://raw.githubusercontent.com/gmauricio-toledo/NLP-LCC/main/Rest-Mex/Rest-Mex_2025_train.csv"
    df = pd.read_csv(url)

    features = df.drop("Type", axis=1)
    target = df["Type"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, stratify=target, random_state=2025
    )

    X_train, y_train = dataFrameParser(X_train, y_train)

    X_test, y_test = dataFrameParser(X_test, y_test)

    models = []
    for vectorizer in (TfidfVectorizer, CountVectorizer):
        for form in (LogisticRegression, ):
            models.append(create_pipeline(vectorizer(), form(), True))
            models.append(create_pipeline(vectorizer(), form(), False))

    for vectorizer in (TfidfVectorizer, CountVectorizer):
        for form in (MultinomialNB, ):
            models.append(create_pipeline(vectorizer(), form(), False))

    results = []
    for model in models:
        model.fit(X_train, y_train)

        train_score = cross_val_score(model, X_train, y_train, cv=3, n_jobs=-1)

        test_score = cross_val_score(model, X_test, y_test, cv=3, n_jobs=-1)

        results.append({
        'Model': str(model),
        'Train Mean': np.mean(train_score),
        'Train Std': np.std(train_score),
        'Test Mean': np.mean(test_score),
        'Test Std': np.std(test_score)
    })
        
    df_results = pd.DataFrame(results)

    df_results.to_csv('model_performance.csv', index=False)

if __name__ == "__main__":
    main()