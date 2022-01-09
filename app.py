from flask import Flask, render_template, request, url_for
from flask_bootstrap import Bootstrap

import os
import csv
import re
import math
import string
import io
from flask import Response
import numpy as np
import pandas as pd
from flask import make_response
import joblib
from flask import Flask

from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
basepath = os.path.abspath(".")
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html', data='',prediction='')


# @app.route('/Download', methods=['GET', 'POST'])
# def Download():
#     prediction = request.form['prediction']
#     prediction.replace(" ", "")
#     prediction = prediction[1:]
#     prediction = prediction[:-1]
#     prediction_list = list(map(str, prediction.split(',')))
#     prediction_list = [str(i) for i in prediction_list] 

#     output = pd.DataFrame({'Prediction': prediction_list})

#     out = io.StringIO()
#     output.to_csv(out)
#     response = make_response(out.getvalue())
#     response.headers["Content-Disposition"] = "attachment; filename=prediction.csv"
#     response.headers["Content-type"] = "text/csv"
#     return response


@app.route('/iRNA5hmC_PS_FASTA', methods=['GET', 'POST'])
def iRNA5hmC_PS_FASTA():
    if request.method == 'POST':
        
        data = request.form['text_box']

        df = fasta_to_csv_1(data)
        
        df = data_construction(df)

        test_data = df.iloc[:, 5]
        
        tf_idf_vectorizer = joblib.load(basepath+'/models/tf_idf.object')

        test_data_transformed = tf_idf_vectorizer.transform(test_data)

        classifier = joblib.load(basepath+'/models/text_categorization.pkl')

        predictions = classifier.predict(test_data_transformed)
        # predictions = np.array(predictions).tolist()

        # 4 = State
        # 3 = Sports
        # 2 = International
        # 1 = Entertainment
        # 0 = Economy

        if predictions == 4:
            predictions = 'State'
        elif predictions == 3:
            predictions = 'Sports'
        elif predictions == 2:
            predictions = 'International'
        elif predictions == 1:
            predictions = 'Entertainment'
        elif predictions == 0:
            predictions = 'Economy'

        return render_template('prediction.html', data=data,prediction=predictions)


def fasta_to_csv_1(data):

    test_data = data.split('\n')
    test_data = [x for x in test_data if not x.startswith('>')]

    for i in range(len(test_data)):
        test_data[i] = test_data[i].upper()    

    df = pd.DataFrame(test_data, columns = ['Text'])

    return df

# def fasta_to_csv_2(data):

#     test_data = data.split('\n')
#     test_data = [x for x in test_data if not x.startswith('>')]
    
#     for i in range(len(test_data)):
#         test_data[i] = test_data[i].upper() 

#     test_data = [x.rstrip("\r") for x in test_data]

#     df = pd.DataFrame(test_data, columns = ['Text'])

#     return df


stopwords = ["অবশ্য","অনেক","অনেকে","অনেকেই","অন্তত","অথবা","অথচ","অর্থাত","অন্য","আজ","আছে","আপনার","আপনি","আবার","আমরা","আমাকে","আমাদের"
             ,"আমার","আমি","আরও","আর","আগে","আগেই","আই","অতএব","আগামী","অবধি","অনুযায়ী","আদ্যভাগে","এই","একই","একে","একটি","এখন","এখনও"
             ,"এখানে","এখানেই","এটি","এটা","এটাই","এতটাই","এবং","একবার","এবার","এদের","এঁদের","এমন","এমনকী","এল","এর","এরা","এঁরা","এস","এত"
             ,"এতে","এসে","একে","এ","ঐ"," ই","ইহা","ইত্যাদি","উনি","উপর","উপরে","উচিত","ও","ওই","ওর","ওরা","ওঁর","ওঁরা","ওকে","ওদের","ওঁদের",
             "ওখানে","কত","কবে","করতে","কয়েক","কয়েকটি","করবে","করলেন","করার","কারও","করা","করি","করিয়ে","করার","করাই","করলে","করলেন",
             "করিতে","করিয়া","করেছিলেন","করছে","করছেন","করেছেন","করেছে","করেন","করবেন","করায়","করে","করেই","কাছ","কাছে","কাজে","কারণ","কিছু",
             "কিছুই","কিন্তু","কিংবা","কি","কী","কেউ","কেউই","কাউকে","কেন","কে","কোনও","কোনো","কোন","কখনও","ক্ষেত্রে","খুব	গুলি","গিয়ে","গিয়েছে",
             "গেছে","গেল","গেলে","গোটা","চলে","ছাড়া","ছাড়াও","ছিলেন","ছিল",'ছিলো',"জন্য","জানা","ঠিক","তিনি","তিনঐ","তিনিও","তখন","তবে","তবু","তাঁদের",
             "তাঁাহারা","তাঁরা","তাঁর","তাঁকে","তাই","তেমন","তাকে","তাহা","তাহাতে","তাহার","তাদের","তারপর","তারা","তারৈ","তার","তাহলে","তিনি","তা",
             "তাও","তাতে","তো","তত","তুমি","তোমার","তথা","থাকে","থাকা","থাকায়","থেকে","থেকেও","থাকবে","থাকেন","থাকবেন","থেকেই","দিকে","দিতে",
             "দিয়ে","দিয়েছে","দিয়েছেন","দিলেন","দু","দুটি","দুটো","দেয়","দেওয়া","দেওয়ার","দেখা","দেখে","দেখতে","দ্বারা","ধরে","ধরা","নয়","নানা","না",
             "নাকি","নাগাদ","নিতে","নিজে","নিজেই","নিজের","নিজেদের","নিয়ে","নেওয়া","নেওয়ার","নেই","নাই","পক্ষে","পর্যন্ত","পাওয়া","পারেন","পারি","পারে",
             "পরে","পরেই","পরেও","পর","পেয়ে","প্রতি","প্রভৃতি","প্রায়","ফের","ফলে","ফিরে","ব্যবহার","বলতে","বললেন","বলেছেন","বলল","বলা","বলেন","বলে",
             "বহু","বসে","বার","বা","বিনা","বরং","বদলে","বাদে","বার","বিশেষ","বিভিন্ন","বিষয়টি","ব্যবহার","ব্যাপারে","ভাবে","ভাবেই","মধ্যে","মধ্যেই","মধ্যেও",
             "মধ্যভাগে","মাধ্যমে","মাত্র","মতো","মতোই","মোটেই","যখন","যদি","যদিও","যাবে","যায়","যাকে","যাওয়া","যাওয়ার","যত","যতটা","যা","যার","যারা",
             "যাঁর","যাঁরা","যাদের","যান","যাচ্ছে","যেতে","যাতে","যেন","যেমন","যেখানে","যিনি","যে","রেখে","রাখা","রয়েছে","রকম","শুধু","সঙ্গে","সঙ্গেও",
             "সমস্ত","সব","সবার","সহ","সুতরাং","সহিত","সেই","সেটা","সেটি","সেটাই","সেটাও","সম্প্রতি","সেখান","সেখানে","সে","স্পষ্ট","স্বয়ং","হইতে","হইবে",
             "হৈলে","হইয়া","হচ্ছে","হত","হতে","হতেই","হবে","হবেন","হয়েছিল","হয়েছে","হয়েছেন","হয়ে","হয়নি","হয়","হয়েই","হয়তো","হল","হলে","হলেই","হলেও",
             "হলো","হিসাবে","হওয়া","হওয়ার","হওয়ায়","হন","হোক","জন","জনকে","জনের","জানতে","জানায়","জানিয়ে","জানানো","জানিয়েছে","জন্য","জন্যওজে",
             "জে","বেশ","দেন","তুলে","ছিলেন","চান","চায়","চেয়ে","মোট","যথেষ্ট","টি"]


# Function for removing special characters from a sentence/review/data
special_characters = string.punctuation + ',?!।‘’'
numeric_characters = '0123456789০১২৩৪৫৬৭৮৯'
def remove_special_characters(sentence):
    sentence = str(sentence)
    char_list = []
    for char in sentence:
        if char not in special_characters and char not in numeric_characters:
            char_list.append(char)
    updated_sentence = "".join(char_list)
    return updated_sentence

# Function for tokenizing the sentence 
def tokenize(sentence):
    tokens = sentence.split(" ")
    return tokens

# Function for removing stopwords
def remove_stopwords(tokenized_list):
    updated_tokenized_list = [word for word in tokenized_list if word not in stopwords]
    new_word_list = []
    newline_status = 0
    special_character_status = 0
    for word in updated_tokenized_list:
        if "\n" in word:
            word = re.sub('\n', ' ', word)
            if "\xa0" in word:
                word = re.sub('\xa0', ' ', word)
            if ' ' in word:
                collection = word.split(" ")
                for c in collection:
                    newline_status = 1
                    new_word_list.append(c)
                 
        if "\xa0" in word:
            word = re.sub('\xa0', ' ', word)
            if "\n" in word:
                word = re.sub('\xa0', ' ', word)
            if ' ' in word:
                collection = word.split(" ")
                for c in collection:
                    special_character_status = 1
                    new_word_list.append(c)
                
        if newline_status == 1:
            pass
        elif special_character_status == 1:
            pass
        else:
            new_word_list.append(word)    
    return list(set(new_word_list))

# Function for lemmatizing (Keeps the sentence's context)
def lemmatizing(tokenized_list):
    lemmatizing_list = Noun_Stemmer(' '.join(tokenized_list)) ###### change noun or verbal
    return lemmatizing_list

# Function for merging words to create a sentence/review/data
def words_merging(lemmatized_list):
    sentence = ' '.join(str(x) for x in lemmatized_list)
    return sentence

#Function for Noun Stemming
def Noun_Stemmer_Step_1(sentence):
    word_list     = sentence.split(" ")
    new_word_list = []
    independant_inflections_1 = 'তে,কে,রা,দে,কা,রা'.split(",")
    independant_inflections_2 = 'গুলি,গুলো,দের,গুলোতে'.split(",")
    single_inflections_character = 'া,ো,ে,ি,ী'.split(",")
    for word in word_list:
        flag = 0
        for i_inflections_1 in independant_inflections_1:
            if i_inflections_1 in word:
                word = re.sub(i_inflections_1+'$', '', word)
                flag = 1
                break
        
        status = 0
        characters = list(word)
        clean_list = [x for x in characters if x not in single_inflections_character]
        length = len(clean_list)
        if length > 3:
            status = 1
        
        if status == 1:
            for i_inflections_2 in independant_inflections_2:
                if i_inflections_2 in word:
                    word = re.sub(i_inflections_2+'$', '', word)
                    break
                
        new_word_list.append(word)

    return ' '.join(new_word_list)      
    
def Noun_Stemmer_Step_2(sentence):
    word_list     = sentence.split(" ")
    new_word_list = []
    independant_inflections = ['য়ের']
    single_inflections_character = 'া,ো,ে,ি,ী'.split(",")
    vowels = 'অ,আ,ই,ঈ,উ,ঊ,ঋ,এ,ঐ,ও,ঔ'.split(",")
    for word in word_list:
        for i_inflections in independant_inflections:
                if i_inflections in word:
                    temp = re.sub(i_inflections+'$', '', word)
                    characters = list(temp)
                    clean_list = [x for x in characters if x not in single_inflections_character]
                    length = len(clean_list)
                    if length < 2:
                        word = re.sub(i_inflections+'$', '', word)
                        break
                    elif clean_list[-1] in vowels:
                        word = re.sub(i_inflections+'$', '', word)
                        break
                    else:
                        word = word[:-2]
                        break
        for single_inflection in single_inflections_character:
            if  len(word) > 1 and single_inflection in word[-2] and word[-1] == 'র':
                word = word[:-1]
                if word[-1] == 'ে':
                    word = word[:-1]
                    break
                
        new_word_list.append(word)
    
    return ' '.join(new_word_list) 

def Noun_Stemmer_Step_3(sentence):
    word_list     = sentence.split(" ")
    new_word_list = []
    for word in word_list:
        if word != "":
            if word[-1] == 'র':
                index = word.find('ষ')
                if index != -1:
                    if index < len(word) - 2: 
                        if word[index+2]  == 'ট':
                            word = re.sub('র'+'$', '', word)
                        elif 'টির' in word:
                            word = re.sub('টির'+'$', '', word)
            elif 'টি' in word:
                word = re.sub('টি'+'$', '', word)
            elif 'টা' in word:
                word = re.sub('টা'+'$', '', word)
            elif 'টির' in word:
                word = re.sub('টির'+'$', '', word)
            elif 'জন' in word:
                word = re.sub('জন'+'$', '', word)
            elif 'খানা' in word:
                word = re.sub('খানা'+'$', '', word)
            new_word_list.append(word)
    
    return ' '.join(new_word_list)   

def Noun_Stemmer(sentence):
    stemmed_sentence = Noun_Stemmer_Step_1(sentence)
    stemmed_sentence = Noun_Stemmer_Step_2(stemmed_sentence)
    stemmed_sentence = Noun_Stemmer_Step_3(stemmed_sentence)
    return stemmed_sentence.split(" ")

# Function for Bangla Stemming
def Verbal_Stemmer_Step_1(sentence):
    word_list = sentence.split(" ")
    new_word_list = []
    independant_inflections_1 = 'ই,ছ,ত,ব,ল,ন,ক,স,ম'.split(",")
    independant_inflections_2 = 'লা,লো,তো,লে,লে,তা,তি,ছি,ছে,ছো,তে,লি,বে'.split(",")
    combined_inflections = 'ছিলাম,ছিলেন,ছেন,লাম,লেন,তেন,তাম,বেন'.split(",")
    single_inflections_character = 'া,ো,ে,ি,ী'.split(",")
    for word in word_list:
        if word == 'আন':
            word = 'আনা'
            new_word_list.append(word)
        elif word == 'আস':
            word = 'আসা'
            new_word_list.append(word)
        elif word == 'আস':
            word = 'আসা'
            new_word_list.append(word)
        elif word == 'আয়':
            word = 'আসা'
            new_word_list.append(word)
        elif word == 'এলেন':
            word = 'আসা'
            new_word_list.append(word)
        elif word == 'এসেছিলাম':
            word = 'আসা'
            new_word_list.append(word)
        elif word == 'গিয়েছিলাম':
            word = 'যাওয়া'
            new_word_list.append(word)
        elif word == 'খাচ্ছিলাম':
            word = 'খাওয়া'
            new_word_list.append(word)
        else:
            status = 0
            characters = list(word)
            length1 = len(characters)
            clean_list = [x for x in characters if x not in single_inflections_character]
            length2 = len(clean_list)
            if length1 == length2 and length1 < 3:
                status = 1

            if status == 1:
                new_word_list.append(word)
            elif word == 'দে':
                new_word_list.append(word)
            elif word == 'খা':
                new_word_list.append(word)
            elif  word[-1] == 'া':
                new_word_list.append(word)
            else:
                status = 0
                for c_inflections in combined_inflections:
                    if c_inflections in word:
                        status = 1
                        word = re.sub(c_inflections+'$', '', word)
                if status == 0:
                    for i_inflections_1 in independant_inflections_1:
                        if len(list(word)) > 2:
                            if word[-1] == i_inflections_1:
                                word = word[:-1]
                                if word[-1] in single_inflections_character:
                                    for i_inflections_2 in independant_inflections_2:
                                        word_part1 = word[-1]
                                        word_part2 = word[-2]
                                        inflection_list = list(i_inflections_2)
                                        inflection_part1 = inflection_list[1]
                                        inflection_part2 = inflection_list[0]
                                        if (word_part1 == inflection_part1) and (word_part2 == inflection_part2):
                                            word = word[:-2]
                                            break
                            else:
                                if len(list(word)) == 3:
                                    if word[-1] == 'ে':
                                        word = word[:-1]
                                elif word[-1] in single_inflections_character:
                                    for i_inflections_2 in independant_inflections_2:
                                        word_part1 = word[-1]
                                        word_part2 = word[-2]
                                        inflection_list = list(i_inflections_2)
                                        inflection_part1 = inflection_list[1]
                                        inflection_part2 = inflection_list[0]
                                        if (word_part1 == inflection_part1) and (word_part2 == inflection_part2):
                                            word = word[:-2]
                                            break
                new_word_list.append(word)
        
    return ' '.join(new_word_list)
    
def Verbal_Stemmer_Step_2(sentence):
    word_list = sentence.split(" ")
    new_word_list = []
    single_inflections_character = 'া,ো,ে,ি,ী'.split(",")
    for word in word_list:
        status = 0
        characters = list(word)
        length1 = len(characters)
        clean_list = [x for x in characters if x not in single_inflections_character]
        length2 = len(clean_list)
        if length1 == length2 and length1 < 3:
            status = 1
        
        if status == 1:
            new_word_list.append(word)
        elif word == 'দে':
            new_word_list.append(word)
        elif word == 'খা':
            new_word_list.append(word)
        elif  word[-1] == 'া':
            new_word_list.append(word)
        else:        
            length = 0
            characters = list(word)
            clean_list = [x for x in characters if x not in single_inflections_character]
            length = len(clean_list)
            if length < 3:
                if word[-1] == 'য়' or word[-1] == 'ও':
                    word = word[:-1]
                    word += 'ওয়া'
                    new_word_list.append(word)
                elif word == 'বল':
                    word = 'বলা'
                    new_word_list.append(word)
                else:
                    for s_i_inflections in single_inflections_character:
                        if word[-1] == s_i_inflections:
                            word = word[:-1]
                            if word[-1] == 'য়' or word[-1] == 'ও':
                                word = word[:-1]
                                word += 'ওয়া'
                                new_word_list.append(word)
    return ' '.join(new_word_list)

def Verbal_Stemmer_Step_3(sentence):
    word_list = sentence.split(" ")
    new_word_list = []
    single_inflections_character = 'া,ো,ে,ি,ী'.split(",")
    for word in word_list:
        words = list(word)
        if len(words) > 2 and words[1] in single_inflections_character:
            if words[1] == 'ু':
                words[1] = 'ো'
            elif words[1] == 'ি':
                words[1] = 'ে'
            elif words[1] == 'ে':
                words[1] = 'া'
        new_word_list.append(''.join(words))        

    
    return ' '.join(new_word_list)
    
def Verbal_Stemmer(sentence):
    stemmed_sentence = Verbal_Stemmer_Step_1(sentence)
    stemmed_sentence = Verbal_Stemmer_Step_2(stemmed_sentence)
    stemmed_sentence = Verbal_Stemmer_Step_3(stemmed_sentence)
    return stemmed_sentence.split(" ")



def data_construction(df):
    df['Text_removed_special_characters']  = df['Text'].apply(lambda sentence: remove_special_characters(sentence))
    df['Text_tokens']                      = df['Text_removed_special_characters'].apply(lambda sentence: tokenize(sentence))
    df['Text_no_stopwords']                = df['Text_tokens'].apply(lambda sentence: remove_stopwords(sentence))
    df['Text_lemmatized']                  = df['Text_no_stopwords'].apply(lambda sentence: lemmatizing(sentence))
    df['Text_lemmatized_sentence']         = df['Text_lemmatized'].apply(lambda sentence: words_merging(sentence))
    return df