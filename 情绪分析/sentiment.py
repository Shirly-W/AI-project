import sys
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import swadesh
from nltk.stem import PorterStemmer,WordNetLemmatizer
import string




def predict_and_test(model, X_test_bag_of_words):
    '''

    predict and test model and data
    model:decision tree
    X_test_bag_of_words: testing data
    :return: testing results
    '''
    predicted_y = model.predict(X_test_bag_of_words)
    model.predict_proba(X_test_bag_of_words)
    return predicted_y






# create training and testing dictionary
train_dict={"num_of_train":[],"x_train":[],"y_train":[]}
test_dict={"num_of_test":[],"x_test":[],"y_test":[]}
data=sys.argv[1]
test=sys.argv[2]


def r_file():
    '''
    read training and testing files
    '''
    with open(data, 'r', encoding='utf-8') as data_list:
        for d_line in data_list:
            d_parts = d_line.split("\t")
            train_dict["num_of_train"].append(d_parts[0])
            train_dict["x_train"].append(d_parts[1])
            train_dict["y_train"].append(d_parts[-1])
    with open(test, 'r', encoding='utf-8') as test_list:
        for t_line in test_list:
            t_parts = t_line.split("\t")
            test_dict["num_of_test"].append(t_parts[0])
            test_dict["x_test"].append(t_parts[1])
            test_dict["y_test"].append(t_parts[-1])






def del_url():
    '''
    finding the form of URLs of tweets and replacing URLs to a space
    '''
    for num_train in range(len(train_dict["x_train"])):
        train_dict["x_train"][num_train]=re.sub(r'(http|https)?:\/\/(\w|\d|\.|\/|\?|\=|\&|\%)*\b',' ',train_dict["x_train"][num_train],flags=re.MULTILINE)
    for num_test in range(len(test_dict["x_test"])):
        test_dict["x_test"][num_test]=re.sub(r'(http|https)?:\/\/(\w|\d|\.|\/|\?|\=|\&|\%)*\b',' ',test_dict["x_test"][num_test],flags=re.MULTILINE)





def del_junk_words():
    '''
    find the form of invalid words
    delete junk words
    '''
    for n_train in range(len(train_dict["x_train"])):
        train_lines = train_dict["x_train"][n_train].split(" ")
        for line_num in range(len(train_lines)):
            train_lines[line_num]= re.sub(r'[^a-zA-Z#@_$%\d]', '', train_lines[line_num])
        train_dict["x_train"][n_train]=" ".join(train_lines)
    for n_test in range(len(test_dict["x_test"])):
        test_lines = test_dict["x_test"][n_test].split(" ")
        for line_num in range(len(test_lines)):
            test_lines[line_num] = re.sub(r'[^a-zA-Z#@_$%\d]', '', test_lines[line_num])
        test_dict["x_test"][n_test] = " ".join(test_lines)




def stemwords():
    """
    Stemming and Lemmatizing words in the training and testing text
    """
    stemwords=PorterStemmer()
    wordnet=WordNetLemmatizer()
    for stem_w in range(len(train_dict["x_train"])):
        train_words=train_dict["x_train"][stem_w].split(" ")
        for train_stemwords in range(len(train_words)):
            train_words[train_stemwords]=stemwords.stem(train_words[train_stemwords])
            train_words[train_stemwords]=wordnet.lemmatize(train_words[train_stemwords])
        train_dict["x_train"][stem_w]=" ".join(train_words)
    for stem_t in range(len(test_dict["x_test"])):
        test_words=test_dict["x_test"][stem_t].split(" ")
        for test_stemwords in range(len(test_words)):
            test_words[test_stemwords]=stemwords.stem(test_words[test_stemwords])
            test_words[test_stemwords]=wordnet.lemmatize(test_words[test_stemwords])
        test_dict["x_test"][stem_t]=" ".join(test_words)




if __name__ == '__main__':
   r_file()
   del_url()
   del_junk_words()
   stemwords()
   count = CountVectorizer(token_pattern='[\d\w#@_$%]{2,}', lowercase=True, encoding="utf-8",
                           stop_words=set(list(string.punctuation) + list(swadesh.fileids())), max_features=2000,
                           min_df=2)                                                                 # create count vectorizer and change values for some variables
   X_train_bag_of_words = count.fit_transform(train_dict["x_train"])                                 #fit it with training data
   X_test_bag_of_words = count.transform(test_dict["x_test"])                                        #transform testing data into bag of words creaed with fit_transform
   clf = MultinomialNB(alpha=0.8)                                                                    #construct MNB model
   for train_num in range(len(train_dict["y_train"])):
       train_dict["y_train"][train_num] = train_dict["y_train"][train_num].replace("\n", "")
   for test_num in range(len(test_dict["y_test"])):
       test_dict["y_test"][test_num] = test_dict["y_test"][test_num].replace("\n", "")
   model = clf.fit(X_train_bag_of_words, train_dict["y_train"])
   test_predict=predict_and_test(model, X_test_bag_of_words)                                          #obtain predict results
   for num in range(len(test_dict["num_of_test"])):                                                   # print testing results
       print(test_dict["num_of_test"][num], test_predict[num])
