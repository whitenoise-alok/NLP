
def read_text_file(text_file_path):
    txt_file = open(text_file_path,'r')
    text_list, cat_list = [], []
    for line in txt_file.readlines():
        line = line[:-1]
        cat = line.rsplit(',')[-1].strip()
        text_l = line.split(',')[:-3]
        text = ', '.join(text_l).strip()
        text_list.append(text)
        cat_list.append(cat)
    return text_list, cat_list













