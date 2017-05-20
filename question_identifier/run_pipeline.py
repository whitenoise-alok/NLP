from read import read_text_file
from utility import get_histogram
import base_model as bsmdl

import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

import argparse
import os


class TextDocument:

    def __init__(self, text, cat):
        self.raw_text = text
        self.category = cat

        self.no_alien_char_text = ''
        self.no_stop_word_text = ''
    # REMOVING ALIEN CHARACTER OUTSIDE ASCII VALUES

    def remove_alien_char(self):
        clean_word_list = []

        for w in self.raw_text:
            if ord(w) > 128:
                print 'Alien Character: %s with decimal value %s, so removing this character' % (w, ord(w))
                clean_word_list.append(' ')
            else:
                clean_word_list.append(w)

        self.no_alien_char_text = ''.join(clean_word_list)

    def remove_stop_words(self):
        words = nltk.word_tokenize(self.no_alien_char_text)
        words = [word for word in words if word not in stopwords.words('english')]
        self.no_stop_word_text = ' '.join(words)


def main(input_file_path, model_param):
    # READING DATA
    text_list, cat_list = read_text_file(input_file_path)
    cat_hist = get_histogram(cat_list)

    print 'Category Histogram'
    print cat_hist, '\n'

    # LOADING DATA AND CLEANING IT
    doc_list = []
    for text, cat in zip(text_list, cat_list):
        doc = TextDocument(text, cat)
        doc.remove_alien_char()
        doc_list.append(doc)

    # CREATING TRAINING AND TEST DATASET
    train_data, test_data = train_test_split(doc_list, test_size=model_param['test_size'])
    print ''
    print 'length of training data set is %s' % (len(train_data))
    print 'length of testing data set is %s' % (len(test_data))

    train_x = [t.no_alien_char_text for t in train_data]
    train_y = [t.category for t in train_data]

    test_x = [t.no_alien_char_text for t in test_data]
    test_y = [t.category for t in test_data]

    model_file_name = 'model/'+'_'.join([model_param['name'], 'tr', str(model_param['test_size']),
                                'tree', str(model_param['n_estimator']), 'treeab', str(model_param['n_estimator_ab'])])+'.pickle'
    vectorizer_file_name = 'model/vectorizer_'+str(model_param['max_df'])+'.pickle'
    if model_param['train_test']:
        bsmdl.train(train_x, train_y, model_param,model_file_name, vectorizer_file_name)
        bsmdl.test(test_x, test_y, model_file_name, vectorizer_file_name)
    else:
        new_x = train_x + test_x
        new_y = train_y + test_y
        bsmdl.test(new_x, new_y, model_file_name, vectorizer_file_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This method is for training base  model for Question-Identifier')
    parser.add_argument('--input', dest='input_file_path', help='Please input file path', type=str)
    parser.add_argument('--model', dest='model_name', help='Please the model to train', type=str,
                        default='random_forest')
    parser.add_argument('--max_df', dest='max_df', help='This will remove words whose freq is more in corpus',
                        type=float, default=0.8)
    parser.add_argument('--tree', dest='n_estimator', help='no of trees to create', type=str, default=100)
    parser.add_argument('--jobs', dest='n_jobs', help='no of jobs to run parallel', type=int, default=2)
    parser.add_argument('--tree_adaboost', dest='n_estimator_ab', help='no of iteration of boosting', type=int,
                        default=5)
    parser.add_argument('--test_ratio', dest='test_ratio', help='faction of data to validate model', type=float,
                        default=0.3)
    parser.add_argument('--train_test', dest='train_test', help='if true, model will train and test otherwise only test',
                        type=bool, default=True)

    result = parser.parse_args()
    print result.input_file_path
    if not os.path.exists(result.input_file_path):
        print 'input path does not exists, so exiting now'
        exit()
    if result.model_name not in ['ada_boost', 'random_forest']:
        print 'currently program take only two models as input 1. ada_boost 2.random_forest. model name is different ' \
              'so exiting'

    param = {
        'name': result.model_name,
        'test_size': result.test_ratio,
        'max_df': result.max_df,
        'n_estimator': result.n_estimator,
        'n_jobs': result.n_jobs,
        'state': 1,
        'n_estimator_ab': result.n_estimator_ab,
        'train_test':result.train_test
    }

    main(result.input_file_path, param)
