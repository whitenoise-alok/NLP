from sklearn.metrics import confusion_matrix, accuracy_score

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