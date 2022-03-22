from fastapi import FastAPI
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import re
import emoji
import keras
from keras.preprocessing import sequence
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
app = FastAPI()

@app.post('/CNN')
def DialectClassifier_DL(Text: str):
        dic = { "0": 'IQ',
                "1": 'LY',
                "2": 'QA',
                "3": 'PL',
                "4": 'SY',
                "5": 'TN',
                "6": 'JO',
                "7": 'MA',
                "8": 'SA',
                "9": 'YE',
                "10": 'DZ',
                "11": 'EG',
                "12": 'LB',
                "13": 'KW',
                "14": 'OM',
                "15": 'SD',
                "16": 'AE',
                "17": 'BH'}
        filename = 'CNN_MODEL.h5'
        new_model = keras.models.load_model(filename)
        loaded_model = keras.models.load_model(filename)
        tok = pickle.load(open("tok_DL.pkl", 'rb'))

        Text = re.sub('@[\w]+','',Text) #Username
        Text = emoji.get_emoji_regexp().sub(u'', Text) #emmojis
        Text = re.sub("http\S+",'',Text) #URL
        Text = re.sub('[^\w\s]','',Text) #punctuation
        stop = stopwords.words('arabic')
        text_tokens = word_tokenize(Text)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        Text = (" ").join(tokens_without_sw)##stop words

        MAX_SEQUENCE_LENGTH = 100
        x = tok.texts_to_sequences([Text])
        x = sequence.pad_sequences(x,padding='post' ,maxlen=MAX_SEQUENCE_LENGTH)

        out = loaded_model.predict(x)
        my_list = out.tolist()
        max_index = my_list[0].index(max(my_list[0]))
        w = dic.get(str(max_index)) 
        print(my_list)
        return ("The dialect is from: " + str(w))



@app.post('/SVC')
def DialectClassifier(Text: str):
    dic = { "0": 'IQ',
            "1": 'LY',
            "2": 'QA',
            "3": 'PL',
            "4": 'SY',
            "5": 'TN',
            "6": 'JO',
            "7": 'MA',
            "8": 'SA',
            "9": 'YE',
            "10": 'DZ',
            "11": 'EG',
            "12": 'LB',
            "13": 'KW',
            "14": 'OM',
            "15": 'SD',
            "16": 'AE',
            "17": 'BH'}
        
    filename = 'LinearSVC_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    tf1 = pickle.load(open("tfidf1.pkl", 'rb'))

    Text = re.sub('@[\w]+','',Text) #Username
    Text = emoji.get_emoji_regexp().sub(u'', Text) #emmojis
    Text = re.sub("http\S+",'',Text) #URL
    Text = re.sub('[^\w\s]','',Text) #punctuation
    stop = stopwords.words('arabic')
    text_tokens = word_tokenize(Text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    Text = (" ").join(tokens_without_sw) ##stop words

    x = tf1.transform([Text])
    out = loaded_model.predict(x)
    print(np.int64(out))
    o = str(out[0])
    w = dic.get(o) 
    print(w)
    return ("The dialect is from: " + w)
