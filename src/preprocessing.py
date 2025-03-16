import spacy
from spacy.lang.es import stop_words
from sklearn.base import BaseEstimator, TransformerMixin

class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = stop_words.STOP_WORDS
        self.stop_words.update(
            ['\n', '\b', 'si']
        )
        self.nlp = spacy.blank(name='es')
    
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.apply(codificar_decodificar_latin)

        tokenized_X = [[t.text for t in tok_doc if
                not t.is_punct and \
                not t.is_space and \
                t.is_alpha and \
                not t.is_stop] for tok_doc in self.nlp.pipe(X)]

        docs_train = [" ".join(doc) for doc in tokenized_X]

        return docs_train

def dataFrameParser(X, y=None):
    X = X.copy()
    X = X.map(str)

    X.loc[:, 'Title'] = X['Title'].fillna(' ')
    X.loc[:, 'Review'] = X['Review'].drop_duplicates(keep='last')
    X = X.dropna()

    X.insert(loc=2, column='Text', value=X['Title'] + ' ' + X['Review'])

    X['Text'] = X['Text'].apply(lambda entry: entry.lower())

    if y is not None:
        y = y.copy()
        y = y[X.index]
        return X['Text'], y

    return X['Text']



def codificar_decodificar_latin(string):
    try:
        new_string = string.encode('latin-1').decode('utf-8')
        return new_string
    except UnicodeDecodeError as e: # Si no se pudo decodificar del latin, es decir que no esta mal formateado para este encoding
        return string