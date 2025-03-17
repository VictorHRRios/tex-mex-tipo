from preprocessing import TextNormalizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD


def create_pipeline(vectorizer, estimator, reduction=False):
    steps = [
        ("normalize", TextNormalizer()),
        ("vectorizer", vectorizer),
    ]

    if reduction:
        steps.append(("reduction", TruncatedSVD(n_components=1000)))

    # Add the estimator
    steps.append(("classifier", estimator))
    return Pipeline(steps)
