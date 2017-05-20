from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from utility import print_model_details

def train(train_x, train_y,  model_param,model_file_name=''):
    print '\nTRAING THE MODEL'

    # CREATING TF-IDF
    vectorizer = TfidfVectorizer(max_df=model_param['max_df'])
    tfidf_x = vectorizer.fit_transform(train_x)

    # DEFINING MODELS
    clf = ''
    if model_param['name'] == 'random_forest':

        clf = RandomForestClassifier(n_estimators=model_param['n_estimator'], n_jobs= model_param['n_jobs'],
                                     random_state=model_param['state'])

    if model_param['name'] == 'ada_boost':
        clf_rf = RandomForestClassifier(n_estimators=model_param['n_estimator'], n_jobs=model_param['n_jobs'],
                                         random_state=model_param['state'])
        clf = AdaBoostClassifier(n_estimators=model_param['n_estimator'],base_estimator=clf_rf)

    if clf =='':
        print 'Please put proper model name in run_pipeline.'

    # TRAINING MODEL
    clf.fit(tfidf_x,train_y)
    predict_train = clf.predict(tfidf_x)

    if model_file_name !='':
        pass

    # PRINTING MODEL DETAILS
    print_model_details(train_x, train_y, predict_train, clf)

    return clf, vectorizer


def test(test_x, test_y,model, vectorizer):
    print '\nTESTING THE MODEL'

    tfidf_x = vectorizer.transform(test_x)

    predict_test = model.predict(tfidf_x)

    print_model_details(test_x, test_y, predict_test, model)
