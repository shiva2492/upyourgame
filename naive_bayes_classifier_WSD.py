from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import sys
import codecs
import re
from nltk import word_tokenize
stop = set(stopwords.words('english'))

def text_features(text):

    stemmer = PorterStemmer()

    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(text)

    result = [stemmer.stem(x.lower()) for x in tokens if x not in stopwords.words('english') and len(x) > 1]
    return result

def get_feature(word):
    return dict([(word, True)])

# Returns a dict given a list as input
def bag_of_words(words):
    return dict([(word, True) for word in words])

# Breaks input string to make seperate substrings each containing one disambiguation word
def breakString(string):
    sub = '(\w*)\W*(\w*)\W*(apple)\W*(\w*)\W*(\w*)'
    str1 = string
    list_string = []
    for i in re.findall(sub, str1, re.I):
        list_string.append(" ".join([x for x in i if x != ""]))
        #print" ".join([x for x in i if x != ""])
    return list_string

def run_classifier():
    # create our dict of training data
    texts = {}
    texts['fruit'] = 'apple-fruit.txt'
    texts['company'] = 'apple-company.txt'

    #holds a dict of features for training our classifier
    train_set = []
    test_string = []
    result_list = []

    # loop through each item, grab the text, tokenize it and create a training feature with it
    for sense, file in texts.iteritems():
        text = codecs.open(file, 'r', encoding='utf-8').read()
        features = text_features(text)
        train_set = train_set + [(get_feature(word), sense) for word in features]

    classifier = NaiveBayesClassifier.train(train_set)

    input_string = raw_input("Enter the test String: ") 
    count = [];
    split_string = [x for x in map(str.strip, input_string.lower().split('.')) if x];
    for string in split_string:
        split_subString = string.split(" ");
        indiv_count = split_subString.count('apple')
        if indiv_count >1:
            test_string = breakString(string) + test_string
        else:
            test_string.append(string);
        count.append(indiv_count);  
    for line in test_string:
        tokens = bag_of_words(text_features(line))
        decision = classifier.classify(tokens)
        result = "%s - %s" % (decision, line)
        result_list.append(decision)
    print result_list

if __name__ == '__main__':
    run_classifier()

