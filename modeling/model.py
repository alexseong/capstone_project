import pandas as pd
import numpy as np
from string import punctuation
import seaborn as sns
import re
# import cPickle as pickle
import dill as pickle
from time import time
from datetime import datetime
from scipy.sparse import csr_matrix
from collections import Counter
from unidecode import unidecode

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import string
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


'''
The following functions:
- prepare data and perform modeling techniques
- GridSearch, cross-validate and evaluate models
'''
############### metadata modeling ###############

#split project outcome from features
def split_data(df):
    y = df.pop('outcome')
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)
    return X_train, X_test, y_train, y_test


#get the number of successful campaigns in previous week
def get_prev_wk_success(df):
    a = df.groupby('week_num')['outcome'].sum()
    b = pd.DataFrame(a)
    b['week_num'] = b.index
    b.loc[0, 'prevweek'] = b.loc[0, 'week_num']
    for i in range(1, len(b)):
        b.loc[i, 'prevweek_success'] = b.loc[i-1, 'outcome']
    df = df.join(b['prevweek_success'], on='week_num', rsuffix='_prev')
    return df


def filter_df(df, col_selection, category=None):
    df = df[df.country=='US']
    df = df.drop('country', axis=1)
    df.fillna(0,inplace=True)
    df = df.join(df.groupby('week_num')['outcome'].sum(), on='week_num', rsuffix='_success_pw')
    df = df[col_selection]
    if category:
        df = df[df[category]==1]
        project_id = df.pop('project_id')
        df = df
    else:
        project_id = df.pop('project_id')
        df = df
    return project_id, df


def get_train_x_metadata(X_train, X_test, meta_cols, cat, category = None):
    if category:
        selected_meta_cols = [feature for feature in meta_cols if feature not in cat]
        return X_train[selected_meta_cols], X_test[selected_meta_cols], selected_meta_cols
    else:
        return X_train[meta_cols], X_test[meta_cols], meta_cols


def pickle_best_estimator(best_model, model_name,cat_name='all'):
    with open('{}_{}.pkl'.format(model_name,cat_name), 'wb') as f:
        pickle.dump(best_model, f)


def plot_feature_importance( rf):
    """
    Input:
        X_test: 2-d numpy array
        y_test: 1-d numpy array
        rf: random forest object
        df: DataFrame
    """
    feature_importances = np.argsort(rf.feature_importances_)
    print "top twenty:", [meta_cols[i-1] for i in  feature_importances[-1:-21:-1]]

    # Calculate the standard deviation for feature importances across all trees
    n = 10 # top 10 features

    importances = rf.feature_importances_[:n]
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    axisnames =  [meta_cols[i-1] for i in  feature_importances[::-1]]
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(n):
        print("%d. %s (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Film & Video category: Feature importances")
    plt.bar(range(10), importances[indices], yerr=std[indices], color="c", align="center")
    plt.xticks(range(10), axisnames, rotation=70)
    plt.xlim([-1, 10])

    plt.show()


def view_classification_report(model, X_test, y_test):
    """
    Args
        model (sklearn classification model): this model from sklearn that
        will has already been fit
        X_test (2d numpy array): this is the feature matrix
        y_test (1d numpy array): this is the array of targets
    Returns
        nothing, this is just a wrapper for the classification report
    """
    print(classification_report(y_test, model.predict(X_test)))


def gridsearch(param_grid, model, X_train_m, y_train):

    gridsearch = GridSearchCV(model,
                              param_grid,
                              n_jobs=-1,
                              cv=10)
    gridsearch.fit(X_train_m, y_train)
    best_model = gridsearch.best_estimator_
    print('Parameters of best model')
    print(best_model)
    print('Best score')
    print(gridsearch.best_score_)
    return best_model, gridsearch


############### topic exploration ###############
def describe_nmf_results(full_tf, H, W,y_train, n_top_words = 10):
    feature_words = full_tf.get_feature_names()
    # top_topic_words = []
    # for topic_num, topic in enumerate(H):
    #     print("Topic %d:" % topic_num)
    #     print(" ".join([feature_words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    top_topic_words = []
    for topic_num, topic in enumerate(H):
        top_words = [feature_words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        top_topic_words.append([topic_num,top_words])


    nmf_df = pd.DataFrame(top_topic_words, columns=['topic','top_words'])
    group = get_success_pct_per_topic(W_train,y_train)

    mdf = group.merge(nmf_df, left_on='index1', right_on='topic',how='outer')
    return mdf


def get_success_pct_per_topic(W,y):
    topic_label = np.apply_along_axis(func1d=np.argmax,axis=1, arr=W)
    dtf = pd.DataFrame()
    dtf['topic'] = topic_label
    dtf['outcome'] = map(lambda x:1 if x else 0, y)
    group = dtf.groupby('topic').sum()
    group['total'] = dtf.groupby('topic').count()
    group['pct'] = group.outcome/group.total
    return group


################ text classification ################
def get_tf(documents):
    documents = [unidecode(document).lower().translate(None, string.punctuation) for document in documents]
    documents = [re.sub(r'\d+', '', doc) for doc in documents]
    countvec = CountVectorizer(stop_words='english', tokenizer=word_tokenize, min_df=5, max_df =0.95 ,ngram_range=(1, 2),preprocessor=PorterStemmer().stem)
    word_counts_matrix = countvec.fit_transform(documents)
    return countvec, word_counts_matrix

def get_nmf_results(tfidf_matrix, n_topics=20):
    nmf = NMF(n_components=n_topics)
    W = nmf.fit_transform(tfidf_matrix)
    H = nmf.components_
    return W, H, nmf

def get_Wtest(full_tf, X_test_d, nmf):
    full_tf_matrix_test = full_tf.transform(X_test_d)
    W_test = nmf.transform(full_tf_matrix_test)
    return W_test, full_tf_matrix_test


def get_ensemble_features(gs_model_m, gs_model_d,tf_matrix_train,tf_matrix_test,W_train,W_test):
    gs_model_m.fit(X_train_m, y_train)
    y_pred_m_train = gs_model_m.predict_proba(X_train_m)[:,1].reshape(-1,1)
    y_pred_d_train = gs_model_d.predict_proba(tf_matrix_train)[:, 1].reshape(-1,1)
    y_pred_m_test = gs_model_m.predict_proba(X_test_m)[:,1].reshape(-1,1)
    y_pred_d_test = gs_model_d.predict_proba(tf_matrix_test)[:, 1].reshape(-1,1)
    X_ensemble_train = np.hstack((X_train_m, y_pred_m_train, W_train, y_pred_d_train))
    X_ensemble_test = np.hstack((X_test_m, y_pred_m_test, W_test, y_pred_d_test))
    return X_ensemble_train, X_ensemble_test



if __name__ == '__main__':
    merged_df = pd.read_pickle('merged_df.pkl')
    df = get_prev_wk_success(merged_df)
    project_id, full_df = filter_df(df, selected_cols, category='cat_film & video')
    X_train, X_test, y_train, y_test = split_data(full_df)
    X_train_m, X_test_m, selected_meta_cols = get_train_x_metadata(X_train, X_test, meta_cols, cat, category = None)
    X_train_d, X_test_d = X_train['description'], X_test['description']

# metadata #
    param_grid_m = {
        'max_depth': [3,5,None],
        'max_features': ['auto'],
        'criterion': ['gini','entropy'],
        'bootstrap': [True, False]
        }

    rfc_m = RandomForestClassifier(n_jobs=-1,max_features= 'auto' ,n_estimators=200)
    start = time.time()
    gs_model_m, gs_m = gridsearch(param_grid_m, rfc_m, X_train_m, y_train)
    print( time.time()-start)
    # view_classification_report(best_model, X_test_m, y_test)
    # print(confusion_matrix(y_test, best_model.predict(X_test_m)))

# text classification #
## for all categories
    full_tf, full_tf_matrix_train = get_tf(X_train_d)
    param_grid_d = {'alpha': (1e-2, 1e-3)}
    nb_d = MultinomialNB()
    gs_model_d, gs_d = gridsearch(param_grid_d, nb_d, full_tf_matrix_train, y_train)

    W_train, H, nmf = get_nmf_results(full_tf_matrix_train, n_topics=20)
    W_test, full_tf_matrix_test = get_Wtest(full_tf, X_test_d, nmf)

X_ensemble_train, X_ensemble_test = get_ensemble_features(gs_model_m, gs_model_d,full_tf_matrix_train,full_tf_matrix_test,W_train,W_test)

## category film
    tf, tf_matrix_train = get_tf(X_train_d)
    param_grid_d = {'alpha': (1e-2, 1e-3)}
    nb_d = MultinomialNB()
    gs_model_d, gs_d = gridsearch(param_grid_d, nb_d, tf_matrix_train, y_train)

    W_train, H, nmf = get_nmf_results(tf_matrix_train, n_topics=20)
    W_test, tf_matrix_test = get_Wtest(tf, X_test_d, nmf)

X_ensemble_train, X_ensemble_test = get_ensemble_features(gs_model_m, gs_model_d,tf_matrix_train,tf_matrix_test,W_train,W_test)


# ensemble classification #
    ensemble_param_grid = {
        'max_depth': [5,7,10],
        'min_samples_leaf': [3,5,7],
        'min_samples_split':[4,6,8],
        }

    ensemble_rfc = RandomForestClassifier(n_jobs=-1,max_features= 'auto' ,n_estimators=200)

    ensemble_gs = GridSearchCV(ensemble_rfc, ensemble_param_grid, n_jobs=-1, cv=10)

    start = time.time()
    ensemble_gs.fit(X_ensemble_train, y_train)
    print( time.time()-start)


# feature_selection #
selected_cols = [
 'emb_video_count',
 'founder_backed',
 'founder_comments',
 'founder_created',
#  'founder_id',
 'description',
 'image_count',
 'main_video',
 'project_id',
 'goal',
#  'pledged',
#  'country',
#  'spotlight',
 'staff_pick',
#  'usd_pledged',
#  'backers_count',
#  'static_usd_rate',
#  'disable_communication',
 'days_to_launch',
 'proj_live_days',
#  'subscription_rate',
#  'oversubscribed',
 'launched_dow',
 'deadline_dow',
 'cat_art',
 'cat_comics',
 'cat_crafts',
 'cat_dance',
 'cat_design',
 'cat_fashion',
 'cat_film & video',
 'cat_food',
 'cat_games',
 'cat_journalism',
 'cat_music',
 'cat_photography',
 'cat_publishing',
 'cat_technology',
 'cat_theater',
#  'week_num',
 'outcome',
 'pledge_avg',
 'pledge_max',
 'pledge_min',
 'pledge_lvl',
 'pledge_std',
 'avg_backers_required',
#  'avg_backers_per_pledge',
 'desc_word_count',
#  'outcome_successpw',
 'prevweek_success',
 'sentence_count']


meta_cols = [
 'emb_video_count',
 'founder_backed',
 'founder_comments',
 'founder_created',
 'image_count',
 'main_video',
 # 'project_id',
 'goal',
 'staff_pick',
 'days_to_launch',
 'proj_live_days',
 'launched_dow',
 'deadline_dow',
 'cat_art',
 'cat_comics',
 'cat_crafts',
 'cat_dance',
 'cat_design',
 'cat_fashion',
 'cat_film & video',
 'cat_food',
 'cat_games',
 'cat_journalism',
 'cat_music',
 'cat_photography',
 'cat_publishing',
 'cat_technology',
 'cat_theater',
 'pledge_avg',
 'pledge_max',
 'pledge_min',
 'pledge_lvl',
 'pledge_std',
 'avg_backers_required',
 'desc_word_count',
 'prevweek_success',
 'sentence_count']




cat = [ 'cat_art',
 'cat_comics',
 'cat_crafts',
 'cat_dance',
 'cat_design',
 'cat_fashion',
 'cat_film & video',
 'cat_food',
 'cat_games',
 'cat_journalism',
 'cat_music',
 'cat_photography',
 'cat_publishing',
 'cat_technology',
 'cat_theater']

 
