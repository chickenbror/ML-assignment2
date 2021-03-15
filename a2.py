import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random

# To lemmatise the words from dataset
from nltk.corpus import wordnet
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
    POS={'N':wordnet.NOUN, 'V':wordnet.VERB, 'J':wordnet.ADJ, 'R':wordnet.ADV}

    # inputfile already opened as an obj
    # each line: Int<tab>Float<tab>Word<tab>POS<tab>NEtag<newline>;  eg, '19\t1.0\ttroops\tNNS\tO\n'
    for line in inputfile.readlines()[1:]:
        line=line.strip('\n')
        items = line.split('\t')  # [indexNr, sentenceNr, Word,	POS, NEtag]
        items[1] = int( float(items[1]) )  # Turn sentenceNr "NNN.0" into integer
        items[2] = lemmatise(items[2].lower(), POS[ items[3][0] ]) if items[3][0] in POS else items[2]   # Lowercase and lemmatise according to POS, if applicable
        
        if items[4][0]!='O' or items[2] not in ['""""', ',', '.','-','(',')'] : #Some puncts provide important context eg '%' or '-'
            rows.append(items[1:]) # All items but indexNr

    return pd.DataFrame(rows, columns=columns)


# Code for part 2
# Make each NE an obj, with:
#   the NE-type as .neclass property
#   the neighbouring words (5 before, 5 after) as .features property
# (6922 NE/instances from GMB dataset)

# English stopwords and punctuations to avoid
from nltk.corpus import stopwords
import string

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

    #Sandwich sentences with start/end tokens so the context window size will always be 10
    for sen in sents_words_tags:
        start = [ (t,'O') for t in ['<s1>','<s2>','<s3>','<s4>','<s5>']]
        end = [(t,'O') for t in ['</s1>','</s2>','</s3>','</s4>','</s5>']]
        sentences.append( start + list(zip(sen[0],sen[1])) + end )

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
                
    return instances  #6922 objs

# Code for part 3
# generate vectors and split the instances into training and testing datasets at random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
def create_table(instances):

    # .features and .neclass properties from each instance
    corpus=[obj.features for obj in instances] # list of features/doc 
    neclasses=[obj.neclass for obj in instances]# list of class names
    
    #Initialize & config the vectorizer for already-tokenized docs
    vectorizer = TfidfVectorizer(analyzer='word',tokenizer=lambda doc:doc,
                preprocessor=lambda doc:doc, token_pattern=None) 
    #Build corpus vocab & transform each doc to tfidf vector (which is in compressed sparse format)
    vectors=vectorizer.fit_transform(corpus)
    
    #Turn sparce format into array format (df shape = no-of-NEs x vocab-size, eg 6922x4959)
    vectors = pd.DataFrame.sparse.from_spmatrix(vectors)
    
    # Reduced the dims of tfidf vectors (eg to 3000-D)
    tsvd = TruncatedSVD(3000)
    reduced_vecs = tsvd.fit_transform(vectors)  # array of 3000-D vectors

    # Turn into DataFrame & add NE-class column to the left
    reduced_matrix = {'NE_class':neclasses, 'vector':list(reduced_vecs)}
    df = pd.DataFrame.from_dict(reduced_matrix)
    
    return df


def ttsplit(bigdf):

    #Sample ~80% for training:
    df_train = bigdf.sample(frac=0.80, replace=False)
    # train X & train Y
    train_vectors = df_train.vector # array of 5538 vector-arrays
    train_vectors = np.asarray( [list(arr) for arr in train_vectors] ) # convert to array of 5538 vector-lists
    train_neclasses = df_train.NE_class # array of 5538 NE classes
    
    # Excludes the sampled training-data, ie the remaining ~20%
    df_test = bigdf.drop(df_train.index) 
    # test X & test Y
    test_vectors = df_test.vector.to_numpy() # array of 1384 vector arrays
    test_vectors = np.asarray( [list(arr) for arr in test_vectors] ) # convert to array of 1384 vector-lists
    test_neclasses = df_test.NE_class # array of 1384 NE classes
        
    return train_vectors, train_neclasses, test_vectors, test_neclasses

# Code for part 5
def confusion_matrix(truth, predictions):
    
    #Initialize a 8*8 table:
    all_classes=list( set( list(truth)+list(predictions) ) )
    all_classes.sort() # Order classes alphabetically
    table={}
    for i in all_classes:
        table[i]={}
        for j in all_classes:
            table[i][j]=0
    
    #Fill in the counts
    for gold, pred in list(zip(truth, predictions)):
        table[gold][pred]+=1

    return pd.DataFrame.from_dict(table)  # X-axis: truth/gold ; Y-axis:predicted

# Code for bonus part B
def bonusb(filename):
    pass
