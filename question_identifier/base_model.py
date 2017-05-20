from utility import print_model_details

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

import cPickle
import os

def train(train_x, train_y,  model_param,model_file_name, vectorizer_file_name):
    print '\nTRAING THE MODEL'

    # CREATING TF-IDF
    vectorizer = TfidfVectorizer(max_df=model_param['max_df'], min_df = 2)
    tfidf_x = vectorizer.fit_transform(train_x)

    # DEFINING MODELS
    clf = ''
    if model_param['name'] == 'random_forest':

        clf = RandomForestClassifier(n_estimators=model_param['n_estimator'], n_jobs= model_param['n_jobs'],
                                     random_state=model_param['state'])

    if model_param['name'] == 'ada_boost':
        # clf_rf = RandomForestClassifier(n_estimators=model_param['n_estimator'], n_jobs=model_param['n_jobs'],
        #                                  random_state=model_param['state'])
        clf_rf = MultinomialNB(alpha=0.01)
        clf = AdaBoostClassifier(n_estimators=model_param['n_estimator'],base_estimator=clf_rf)
    if model_param['name'] == 'naive_bayes':
        clf = MultinomialNB(alpha=0.001)
    if clf =='':
        print 'Please put proper model name in run_pipeline.'

    # TRAINING MODEL
    clf.fit(tfidf_x,train_y)
    predict_train = clf.predict(tfidf_x)
    print 'saving tf-idf model %s' % (vectorizer_file_name)
    cPickle.dump(vectorizer, open(vectorizer_file_name, 'w'))
    print 'saving classifier model %s' % (model_file_name)
    cPickle.dump(clf,open(model_file_name,'w'))


    # PRINTING MODEL DETAILS
    print_model_details(train_x, train_y, predict_train, clf)



def test(test_x, test_y,model_file_name, vectorizer_file_name):
    if not os.path.exists(vectorizer_file_name):
        print 'Vectorizer File Name: %s' % ( vectorizer_file_name)
        print 'TF-IDF matrix model is not cerated with this setting of max_df. Please run it with train_test=True'
        return None
    else:
        vectorizer = cPickle.load(open(vectorizer_file_name))
    if not os.path.exists(model_file_name):
        print 'Model File Name: %s' % (model_file_name)
        print 'Classifier model is not cerated with these setting. Please run it with train_test=True'
        return None
    else:
        model = cPickle.load(open(model_file_name))
    print '\nTESTING THE MODEL'

    tfidf_x = vectorizer.transform(test_x)

    predict_test = model.predict(tfidf_x)

    print_model_details(test_x, test_y, predict_test, model)
