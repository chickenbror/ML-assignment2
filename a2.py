import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random

# To lemmatise the words from dataset
from nltk.stem import WordNetLemmatizer


# Module file with functions that you fill in so that they can be
# called by the notebook.  This file should be in the same
# directory/folder as the notebook when you test with the notebook.

# You can modify anything here, add "helper" functions, more imports,
# and so on, as long as the notebook runs and produces a meaningful
# result (though not necessarily a good classifier).  Document your
# code with reasonable comments.

# Function for Part 1
def preprocess(inputfile):
    """
    Returns the GMB dataset-obj as a dataframe, 
    the words in which are lowercased and lemmatised. 
    """
    columns = ['sentenceNr', 'word', 'POS', 'NEtag']
    rows = []
    lemmatise=WordNetLemmatizer().lemmatize
    # inputfile already opened as an obj
    # each line: Int<tab>Float<tab>Word<tab>POS<tab>NEtag<newline>;  eg, '19\t1.0\ttroops\tNNS\tO\n'
    for line in inputfile.readlines()[1:]:
        line=line.strip('\n')
        items = line.split('\t')  # [indexNr, sentenceNr, Word,	POS, NEtag]
        items[1] = int( float(items[1]) )  # Turn sentenceNr "NNN.0" into integer
        items[2] = lemmatise( items[2].lower() )  # Turn word to lowercase and citation form
        # filter words? if items[2] not in AvoidList=>append to rows
        rows.append(items[1:]) # All items but indexNr

    return pd.DataFrame(rows, columns=columns)


# Code for part 2
# Make each NE an obj, with:
#   the NE-type as .neclass property
#   the neighbouring words (5 before, 5 after) as .features property
# (6922 NE/instances from GMB dataset)

class Instance:
    def __init__(self, name, neclass, features):
        self.name = name
        self.neclass = neclass
        self.features = features

    def __str__(self):
        return "Class: {} Features: {}".format(self.neclass, self.features) # When an Instance obj is printed

    def __repr__(self):
        return str(self)

def create_instances(inputdata):
    """Converts the DF from Part1 to a list of Instances. """
    instances = []

    # Convert Part1's DF to list of sentences:
    # Group rows by sentenceNr and make each sentence a list of (word,tag) tuples
    words_only = inputdata.groupby('sentenceNr')['word'].apply(list)
    tags_only = inputdata.groupby('sentenceNr')['NEtag'].apply(list)
    sents_words_tags = list(zip(words_only, tags_only))
    sentences = []  # 2999 sublists, each = [(w1,tag),(w2,tag)...]
    for sen in sents_words_tags:
        sentences.append( list(zip(sen[0],sen[1])) )

    for sen in sentences:
        for duo in sen:
            word,tag = duo[:]
            
            # Find an NE led by a B-tag word
            if tag[0]=='B': 
                # Individual words of NE & its class-tag
                NEwords=[word] # first word of NE
                NEclass=tag[2:]
                i=sen.index(duo) # index of current word in the sentence

                # Recursively look for ensuing words with I- tag, 
                # as well as next 5 words led by an O-tag word
                
                try:
                    c=1
                    while sen[i+c][1]=='I': # rest of the NE words
                        nextword = sen[i+c][1]
                        NEwords.append(nextword)
                        c+=1
                        
                    if sen[i+c][1]=='O': # the first of next 5 words
                        next5=[]
                        for nxt in range(0,5): # includes current index & next4
                            try:
                                nextfeat=sen[i+c+nxt][0]
                                next5.append(nextfeat)
                            except IndexError:
                                pass           
                except IndexError:
                    pass

                # Previous 5 words before the word with B- tag
                prev5=[]
                for prev in range(5,0,-1): #prev-5~-1; excludes current index
                    try:
                        if i-prev>=0: # index should be positive to prevent indexing from the end
                            prevword=sen[i-prev][0]
                            prev5.append(prevword)
                    except IndexError:
                        pass
                
                # Instanciate an object and add to list
                NEname = ' '.join(NEwords)
                features = prev5 + next5 # The neighbouring 10 words
                instances.append(Instance(NEname, NEclass, features))
                
    return instances  

# Code for part 3
# generate vectors and split the instances into training and testing datasets at random
from sklearn.feature_extraction.text import TfidfVectorizer
def create_table(instances, top_freq=3000, stopwords="yes"):

    # .features and .neclass properties from each instance
    docs=[obj.features for obj in instances] # Treat each features list as a doc
    neclasses=[obj.neclass for obj in instances]

    if stopwords=="yes":
        stopwords_opt=None
    elif stopwords=="no":
        stopwords_opt = 'english'
    tfidf = TfidfVectorizer(preprocessor=' '.join, max_features=top_freq, stop_words=stopwords_opt)
    tdidf_vecs= tfidf.fit_transform(docs) # Vectors are in compressed sparse format

    # pd.DataFrame(tdidf_vecs[1], columns=["vector"])
    df=pd.DataFrame.sparse.from_spmatrix( tdidf_vecs )
    df.insert(0,'class',neclasses,) # Add to the leftest column
    return df

def ttsplit(bigdf):
    df_train = pd.DataFrame()
    df_train['class'] = [random.choice(['art','eve','geo','gpe','nat','org','per','tim']) for i in range(80)]
    for i in range(3000):
        df_train[i] = npr.random(80)

    df_test = pd.DataFrame()
    df_test['class'] = [random.choice(['art','eve','geo','gpe','nat','org','per','tim']) for i in range(20)]
    for i in range(3000):
        df_test[i] = npr.random(20)
        
    return df_train.drop('class', axis=1).to_numpy(), df_train['class'], df_test.drop('class', axis=1).to_numpy(), df_test['class']

# Code for part 5
def confusion_matrix(truth, predictions):
    print("I'm confusing.")
    return "I'm confused."

# Code for bonus part B
def bonusb(filename):
    pass
