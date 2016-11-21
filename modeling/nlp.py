import pandas as pd
import numpy as np
from string import punctuation
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB


def get_category_data(df, cat_name):
    X = df[df[cat_name] ==1]['description'].values
    y = df[df[cat_name] ==1]['outcome'].values
    return X,y


def tokenizing(doc):
    return [snowball.stem(word) for word in word_tokenize(doc.lower())
            if word not in punctuation]


text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', tokenizer=tokenizing)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())])


parameters = {'vect__ngram_range': [(2, 2), (2,3)],
            'vect__min_df': (0.0, 0.05),
            'vect__max_features': (None, 10000),
            # 'tfidf__use_idf': (True, False),
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
    X, y = get_category_data(merged_df, 'cat_design')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)

    #vect = TfidfVectorizer(stop_words='english', tokenizer=tokenizing)
    #vect.train(X_train)


    #X_train = vect.transform(X_train)

    text_clf = text_clf.fit(X_train, y_train)
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(X_train, y_train)
    gs_clf.predict(X_test)
    gs_clf.best_score_
    pickle_best_estimator(gs_clf, cat_name)


def extract_pipeline_steps():
    design_best = gs_clf.best_estimator_
    vect = design_best.get_params()['vect']
    clf = design_best.get_params()['clf']
    tfidf = design_best.get_params()['tfidf']
    feature_names = vect.get_feature_names()
    tfidf_X_train = vect.transform(X_train)
    tfidf_X_train = tfidf.transform(tfidf_X_train)
    tfidf_X_train = tfidf_X_train.todense()
    return tfidf_X_train

# use -coef
topn = sorted(zip(-clf.coef_[0], feature_names))[:80]
for coef, feat in topn:
    print (feat, coef)


tfidf_X_train = vect.transform(X_train)
tfidf_X_train = tfidf.transform(tfidf_X_train)

sorted(zip(-tfidf_x_train[:,i], range(len(tfidf_x_train))))[:20]

def debug(name):
    i = feature_names.index(name)
    tfidf_x_train[:,i]
    df = pd.DataFrame()
    df['y'] = y_train
    df['mask'] = np.array( tfidf_x_train[:,i] > 0 )
    return df.groupby(['y','mask']).size()

debug('stretch goal')
debug('play sound')
debug('-- --')
debug('everi day')


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
