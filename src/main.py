from preprocessing import dataFrameParser
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from modeling import create_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle
from datetime import datetime


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

    pipe = create_pipeline(TfidfVectorizer(), LogisticRegression(), False)

    param_grid = {
        "vectorizer__max_features": [500, 1000, 10000, 50000],
        "vectorizer__ngram_range": [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3)],
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__max_iter": [100, 500, 1000],
        "classifier__tol": [1e-4, 1e-3, 1e-2],
    }

    pipe_tuning = RandomizedSearchCV(
        pipe,
        param_distributions=param_grid,
        cv=10,
        n_iter=100,
        n_jobs=-1,
        verbose=2,
        scoring="f1_macro",
    )

    pipe_tuning.fit(X_train, y_train)  # Cambiar esto si estas en el ocotillo

    results = pd.DataFrame(pipe_tuning.cv_results_)
    results.to_csv("results.csv", index=False)

    print("Best Parameters:", pipe_tuning.best_params_)
    print("Best Score:", pipe_tuning.best_score_)

    y_pred = pipe_tuning.predict(X_test)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        linewidths=1,
        linecolor="black",
        xticklabels=pipe_tuning.classes_,
        yticklabels=pipe_tuning.classes_,
    )

    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Confusion Matrix")
    plt.savefig("Confusion_Matrix.png")

    time = datetime.now().strftime("%Y-%m-%d, %H")
    path = "type-classifier-{}".format(time)

    with open(path, "wb") as f:
        pickle.dump(pipe_tuning, f)


if __name__ == "__main__":
    main()
