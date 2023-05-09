from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def create_pipeline(X_train, y_train):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2, ngram_range=(1, 2), token_pattern=r'(?u)\b[\w\#.-]+\b')),
        ('clf', LogisticRegression(max_iter=1000, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline
