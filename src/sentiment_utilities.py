# imports
import csv
import pandas as pd
import numpy as np
import uuid

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
porter = PorterStemmer()
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA

import matplotlib.pyplot as plt


def load_combine_split(path):
    '''Loads in corpi from specified path; path contains labelled product review files from UCI ML Repo:
    https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
    
    Args
        - path of .txt files
    
    Returns
        - df of training data (70% randomly sampled)
        - df of validation data (30% randomly sampled)
    '''
    
    #read in and combine reviews into single df
    with open(path + 'amazon_cells_labelled.txt', 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        amz = list(reader)
    
    with open(path + 'imdb_labelled.txt', 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        imdb = list(reader)
    
    with open(path + 'yelp_labelled.txt', 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        ylp = list(reader)
    
    df = pd.concat([
            pd.DataFrame(amz, columns = ['text', 'label']),
            pd.DataFrame(imdb, columns = ['text', 'label']),
            pd.DataFrame(ylp, columns = ['text', 'label'])
           ], axis=0).reset_index()
    
    #create uid and combine with reviews
    uid = []
    for i in range(0, len(amz)+len(imdb)+len(ylp)):
        uid.append(str(uuid.uuid1()))
    dfid = pd.DataFrame(uid, columns = ['uid'])
    
    out = pd.concat([df[['text', 'label']], dfid], axis=1, join='inner')
    
    #split into train/validate
    validate = out.sample(frac=0.3)
    train = out.loc[out['uid'].isin(validate['uid'].tolist())==False]
    if train.shape[0]+validate.shape[0] == out.shape[0]:
        print("Train and test appropriately split")
        print("Train shape is "+str(train.shape[0]) + " rows and "+str(train.shape[1]) + " columns.")
        print("Validate shape is "+str(validate.shape[0]) + " rows and "+str(validate.shape[1]) + " columns.")
    
    return train, validate



def tokenize(text):
    '''Tokenizes and removes stop words, lower cases, removed punct
    
    Args:
        - string of text raw
    
    Returns:
        - string of processed text
    '''
    tokenizer = nltk.RegexpTokenizer(r"\w+") 
    text_tokens = tokenizer.tokenize(text)
    tokens_without_sw = [word.lower() for word in text_tokens if not word in stopwords.words()]
    stemmed = " ".join([porter.stem(word) for word in tokens_without_sw])
    return stemmed

def process_data(train, validate, tfidf = False):
    '''creates and tokenizes text for downstream modelling (TFIDF or embedding)
    
    Args: 
        train: dataframe of training product reviews
        validate: dataframe of training product reviews

    Returns:
        - train_x: list of tokenized strings 
        - train_y: list of labels (1,0)
        - val_X: list of tokenized strings
        - val_y: list of labels (1,0)
        
    '''
    #create train_x, train_y, val_X, val_y
    ltrain = train.values.tolist()
    if tfidf == True:
        train_X = [tokenize(i[0]) for i in ltrain]
    else:
        train_X = [i[0] for i in ltrain]
    train_y = [i[1] for i in ltrain]
    
    lval = validate.values.tolist()
    if tfidf == True:
        val_X = [tokenize(i[0]) for i in lval]
    else:
        val_X = [i[0] for i in lval]
    val_y = [i[1] for i in lval]

    return train_X, train_y, val_X, val_y


def vector_agg(ft, text, punct_remove):
    '''Removes punctuation and embeds words from a given document (review) using pretrained FastText model
       and then aggregates embeddings into single 1x300 vector by taking mean of each element across embedded
    
    Args: 
        - ft: pretrained FastText model
        - text: document (string)
        - punct_remove: list of punctuation to remove from text prior to embedding
    
    Returns:
        - 1 x 300 np.array of aggregated single-word embeddings
    '''
    #remove punctuation
    t = text.translate(str.maketrans('', '', punct_remove)).split()
    count = 0
    for i in t:
        if count == 0:
            vec = ft.get_word_vector(i).reshape(1,300)
        else:
            vec = np.concatenate(  ( 
                                      vec, 
                                      ft.get_word_vector(i).reshape(1,300)
                                    ), axis = 0)
        count+=1
    
    return np.mean(vec, axis=0)



def embed_prep(ft, corpus, punct_remove):
    '''Embeds a corpus  using pretrained FastText model
       returns a single 1x300 vector for each document
     
    Args:
         - corpus of text (np.array)
    
    Returns
        - n x 1 x 300 array, where n is length of corpus
    '''
    count = 0
    for t in corpus:
        if count == 0:
            vec = vector_agg(ft, t, punct_remove).reshape(1,300)
        else:
            vec = np.concatenate(  ( 
                                      vec, 
                                      vector_agg(ft, t, punct_remove).reshape(1,300)
                                    ), axis = 0)
        count+=1
    print(vec.shape)
    return vec