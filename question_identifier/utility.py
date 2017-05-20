from sklearn.metrics import confusion_matrix, accuracy_score
import nltk
import numpy as np

def get_histogram(cat_list):
    cat_dict = {}
    for cat in cat_list:
        if cat in cat_dict:
            cat_dict[cat] +=1
        else:
            cat_dict[cat] = 1
    return cat_dict

def print_model_details(var_x, var_y, predicted, clf,model_name = ''):
    accuracy = accuracy_score(var_y, predicted)
    cf_matrix = confusion_matrix(var_y, predicted)
    if model_name == '':
        print 'Accuracy: %s'%(accuracy)
        print 'Category Labels: %s'%(clf.classes_)
        print 'Confusion Matrix'
        print cf_matrix
        # print clf.oob_score_
        print '\n<<Wrong Predicted Cases>>\n'
        for t_x, t_y, p_y in zip(var_x, var_y, predicted):
            if t_y != p_y:
                print 'Text: %s'%( t_x)
                print 'Actual Category: %s'%(t_y)
                print 'Predicted Category: %s'%(p_y)
                print '#' * 25

def get_vocab(text_list):
    vocab = set()
    for text in text_list:
        text_words = nltk.word_tokenize(text)
        _ = [vocab.add(w) for w in text_words]

    return list(vocab)

def get_location(master_list, param, val):
    for m_ind, m in enumerate(master_list):
        if m[param] == val:
            return m_ind
    return -1

def get_key_pmi_feature(train_x, train_y, vocab, no_feature=-1):
    categ_list = list(set(train_y))
    vocab_list = list(vocab)
    pmi_list = []

    for categ in categ_list:
        categ_bool = [t == categ for t in train_y]
        prob_y = float(np.sum(categ_bool))/len(categ_bool)

        for w_ind, word in enumerate(vocab_list):
            if w_ind %100 ==0:
                print word, w_ind

            word_bool = [word in text for text in train_x]
            prob_x = float(np.sum(word_bool))/len(word_bool)
            categ_word_bool = [w and c for w, c in zip(word_bool, categ_bool)]
            prob_xy = np.sum(categ_word_bool)/len(categ_word_bool)
            pmi = np.log((prob_xy+0.00001)/(prob_x * prob_y+0.00001))
            p_loc = get_location(pmi_list,'word', word)
            if p_loc == -1:
                pmi_list.append({
                    'word':word,
                    categ:pmi
                })
            else:
                pmi_list[p_loc][categ] = pmi


    key_word_dict = {}
    for categ in categ_list:
        sorted_pmi_list = sorted(pmi_list,key=lambda x:-x[categ])
        temp_word_list = [s['word']for s in sorted_pmi_list]
        if no_feature == -1:
            key_word_dict[categ] = temp_word_list
        else:
            key_word_dict[categ] = temp_word_list[:no_feature]

    return key_word_dict



if __name__ == '__main__':
    train_x = ['Alok lives in bangalore', 'Alok wants to live to pune', 'Alok is machine learning engineer',
               'Alok loves machine learning algorithm']
    train_y = ['0', '0', '1', '1']
    get_key_pmi_feature(train_x,train_y)



