import pandas as pd
import numpy as np
from string import punctuation

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB


def get_category_data(df, cat_name):
    X = df[df[cat_name] ==1]['description'].values
    y = df[df[cat_name] ==1]['outcome'].values
    return X,y


def tokenizing(doc):
    return [snowball.stem(word) for word in word_tokenize(doc.lower())
            if word not in punctuation]


text_clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', tokenizer=tokenizing)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())])


parameters = {'vect__ngram_range': [(2, 2), (2,3)],
            'vect__min_df': (0.0, 0.05, 0.1),
            'vect__max_features': (None, 5000, 10000),
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3)
             }


def get_best_params(gs_clf):
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

def pickle_best_estimator(gs_clf, cat_name):
    filename = cat_name+'_best_est'+'.pkl'
    with open(filename, 'wb') as fin:
        pickle.dump(gs_clf.best_estimator_, fin)

if __name__ == '__main__':
    merged_df = pd.read_pickle('merged_df.pkl')
    snowball = SnowballStemmer('english')
    punctuation = set(punctuation)
    X, y = get_category_data(merged_df, 'cat_publishing')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)
    text_clf = text_clf.fit(X_train, y_train)
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(X_train, y_train)
    gs_clf.predict(X_test)
    gs_clf.best_score_
    pickle_best_estimator(gs_clf, cat_name)

##
#fashion 0.76803497692494538
#technology 0.78351078351078352
# publishing 0.67

# fashion
# clf__alpha: 0.01
# tfidf__use_idf: False
# vect__ngram_range: (1, 3)

#techonology
# clf__alpha: 0.01
# tfidf__use_idf: True
# vect__ngram_range: (1, 3)

############



#
# def pickle_vec(vectorizer, sm, cat):
#     '''
#     Pickle the vectorizer instance and sparse matrix
#     '''
#     v = open('{}_vectorizer.pickle'.format(cat), 'wb')
#     pickle.dump(vectorizer, v)
#     v.close()
#
#     f = open('{}_sparse.pickle'.format(cat), 'wb')
#     pickle.dump(sm, f)
#     f.close()
