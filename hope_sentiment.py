#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 18:21:19 2022

@author: nihz415
"""

#pip install snscrape
#pip install git+https://github.com/JustAnotherArchivist/snscrape.git


import snscrape.modules.twitter as sntwitter
import pandas
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
import matplotlib
%matplotlib inline
import seaborn as sns
import string
import warnings 
from scipy.special                   import expit
from sklearn                         import linear_model
from sklearn.decomposition           import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors               import KNeighborsClassifier
from sklearn.metrics                 import confusion_matrix
from sklearn.metrics                 import classification_report
%pip install wordcloud
from wordcloud import WordCloud
%pip install matplotlib
import re
import nltk
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Creating list to append tweet data to
tweets=[]
# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('#hopeww lang:en ').get_items()):
    tweets.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    
# Creating a dataframe from the tweets list above
len(tweets)
tweets[-1]
tweets_df = DataFrame(tweets, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
tweets_df
outputpath='/Users/nihz415/Desktop/hopeww.csv'
tweets_df.to_csv(outputpath,index=True,header=True)


dta=pd.read_csv('hopeww.csv')
dta.duplicated()
dta=dta.drop_duplicates()
dta.duplicated(['Text'])
dta=dta.drop_duplicates(['Text'])
##clean
def clean_text(text):
    text=re.sub(r'@[A-Za-z0-9]+',' ' ,text)
    text = re.sub(r'https?:\/\/.*\/\w*',' ',text)
    text = re.sub(r'[^a-zA-Z#]',' ',text)
    text = re.sub(r'#',' ',text)
    text = re.sub(r'RT[\s]+',' ',text)
    return text

dta['tidy_tweet'] = dta['Text'].astype('U').apply(clean_text)  
dta['tidy_tweet']=dta['tidy_tweet'].str.lower()   
dta['tidy_tweet'] = dta['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("n't"," not")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("I'm","I am")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("'ll"," will")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("It's","It is")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("it's","It is")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("that's","that is")


stop = stopwords.words('english')
additional=['hopeww','hope','this','from','with','support','here']
stop+=additional
dta['tidy_tweet'] = dta['tidy_tweet'].str.split()
dta['tidy_tweet']=dta['tidy_tweet'].apply(lambda x:' '.join([item for item in x if item not in stop]))  

#CLASSIFICATION
def logistic_reg_classifier_mult_labels(X_train,y_train,X_valid,y_valid,X_test,y_test):
    
    ' . '
    categories         = pd.DataFrame(np.sort(np.unique(y_train))).reset_index()
    categories.columns = ['index','label']
    

    ' . '    
    ccp_train_list = []
    ccp_valid_list = []
    ccp_test_list  = []
    for cat in categories['label'].to_list():
        y_train_c = 1*(y_train==cat)
        clf       = linear_model.LogisticRegression(tol          = 0.0001,
                                                    max_iter     = 10000,
                                                    random_state = None).fit(X_train, y_train_c)
        ccp_train_list.append(  clf.predict_proba(X_train)[:,1])
        ccp_valid_list.append(  clf.predict_proba(X_valid)[:,1])
        ccp_test_list.append(   clf.predict_proba(X_test)[:,1])
    
    ' . Topic probability matrix'
    ccp_train = pd.DataFrame(ccp_train_list).transpose()
    ccp_valid = pd.DataFrame(ccp_valid_list).transpose()
    ccp_test  = pd.DataFrame(ccp_test_list).transpose()
    
    'reset column index'
    ccp_train.columns = categories['label'].to_list()
    ccp_valid.columns = categories['label'].to_list()
    ccp_test.columns = categories['label'].to_list()
    
    'Choosing your predictive category for the y '
    ccp_train['label_hat'] =  ccp_train.idxmax(axis=1)
    ccp_valid['label_hat'] =  ccp_valid.idxmax(axis=1)
    ccp_test['label_hat']  =  ccp_test.idxmax(axis=1)    
    
    'caculate'
    confusionmatrix=confusion_matrix(y_test,ccp_test['label_hat'] )
    correct =  np.trace(confusionmatrix)
    total = confusionmatrix.sum()
    percent_accuracy = correct/total
        
    return(confusionmatrix,percent_accuracy)

logistic_reg_classifier_mult_labels(X_train,y_train,X_valid,y_valid,X_test,y_test)
dta_test=dta
dta_test.columns

dta_test['ML_group']  = np.random.randint(10,size=dta.shape[0])
dta_test['ML_group']  = (dta_test['ML_group']<=7)*0 + (dta_test['ML_group']==8)*1 +(dta_test['ML_group']==9)*2
corpus= dta_test['tidy_tweet'].to_list()   
vectorizer_count     = CountVectorizer(lowercase   = True,ngram_range = (1,1),max_df      = 0.99,min_df      = 0.001);
X                    = vectorizer_count.fit_transform(dta_test['tidy_tweet'].values.astype('U'))
features_frequency   = pd.DataFrame({'feature'           : vectorizer_count.get_feature_names(),'feature_frequency' : X.toarray().sum(axis=0)})
X.shape

X_train = X[np.where(dta_test['ML_group']==0)[0],:]
X_valid = X[np.where(dta_test['ML_group']==1)[0],:]
X_test  = X[np.where(dta_test['ML_group']==2)[0],:]
y_train = dta_test.loc[dta_test['ML_group']==0,['rate']]['rate'].to_numpy()
y_valid = dta_test.loc[dta_test['ML_group']==1,['rate']]['rate'].to_numpy()
y_test  = dta_test.loc[dta_test['ML_group']==2,['rate']]['rate'].to_numpy()


corpus= dta_test['tidy_tweet'].to_list()   
vectorizer_count     = CountVectorizer(lowercase   = True,ngram_range = (1,1),max_df      = 0.99,min_df      = 0.001);
X                    = vectorizer_count.fit_transform(dta['tidy_tweet'].values.astype('U'))
features_frequency   = pd.DataFrame({'feature'           : vectorizer_count.get_feature_names(),'feature_frequency' : X.toarray().sum(axis=0)})
X.shape
dta_test['ML_group'] = dta_test['ML_group'].fillna(4)

X_train = X[np.where(dta_test['ML_group']==0)[0],:]
X_valid = X[np.where(dta_test['ML_group']==1)[0],:]
X_test  = X[np.where(dta_test['ML_group']==2)[0],:]
y_train = dta_test.loc[dta_test['ML_group']==0,['rate']]['rate'].to_numpy()
y_valid = dta_test.loc[dta_test['ML_group']==1,['rate']]['rate'].to_numpy()
y_test  = dta_test.loc[dta_test['ML_group']==2,['rate']]['rate'].to_numpy()
X_predict  =X


def logistic_prediction(X_train,y_train,X_valid,y_valid,X_test,y_test,X_predict):
    
    ' . '
    categories         = pd.DataFrame(np.sort(np.unique(y_train))).reset_index()
    categories.columns = ['index','label']
    

    ' . '    
    ccp_train_list = []
    ccp_valid_list = []
    ccp_test_list  = []
    predict_list=[]
    for cat in categories['label'].to_list():
        y_train_c = 1*(y_train==cat)
        clf       = linear_model.LogisticRegression(tol          = 0.0001,
                                                    max_iter     = 10000,
                                                    random_state = None).fit(X_train, y_train_c)
        ccp_train_list.append(  clf.predict_proba(X_train)[:,1])
        ccp_valid_list.append(  clf.predict_proba(X_valid)[:,1])
        ccp_test_list.append(   clf.predict_proba(X_test)[:,1])
        predict_list.append(   clf.predict_proba(X_predict)[:,1])
    
    ' . Topic probability matrix'
    ccp_train = pd.DataFrame(ccp_train_list).transpose()
    ccp_valid = pd.DataFrame(ccp_valid_list).transpose()
    ccp_test  = pd.DataFrame(ccp_test_list).transpose()
    predict  = pd.DataFrame(predict_list).transpose()
    
    
    'reset column index'
    ccp_train.columns = categories['label'].to_list()
    ccp_valid.columns = categories['label'].to_list()
    ccp_test.columns = categories['label'].to_list()
    predict.columns = categories['label'].to_list()
    
    'Choosing your predictive category for the y '
    ccp_train['label_hat'] =  ccp_train.idxmax(axis=1)
    ccp_valid['label_hat'] =  ccp_valid.idxmax(axis=1)
    ccp_test['label_hat']  =  ccp_test.idxmax(axis=1)
    predict['label_hat']  =  predict.idxmax(axis=1)
    
    'prediction'
    dta_test['predict']=predict['label_hat']
        
    return(dta_test)

dta_final=logistic_prediction(X_train,y_train,X_valid,y_valid,X_test,y_test,X_predict)



########plot
dta_final.columns
len(dta_final)
dta_final.head()
#part 2:Discription
##overall count
dta_final['date']=dta_final['Datetime'].str.split(' ',expand=True)[0]
dta_final['date'] = pd.to_datetime(dta_final['date'])
count_in_day = dta_final.groupby(['date'],as_index=False).count()
#plot
date_format = mpl.dates.DateFormatter("%y-%m")
ax = plt.gca()
ax.xaxis.set_major_formatter(date_format)
plt.plot(count_in_day['date'], count_in_day['rate_x'])

plt.axis('tight')

corpus= dta_final['tidy_tweet'].to_list()   
vectorizer_count     = CountVectorizer(lowercase   = True,ngram_range = (1,1),max_df      = 0.99,min_df      = 0.001);
X                    = vectorizer_count.fit_transform(dta_final['tidy_tweet'].values.astype('U'))
features_frequency   = pd.DataFrame({'feature'           : vectorizer_count.get_feature_names(),'feature_frequency' : X.toarray().sum(axis=0)})
X.shape
features_frequency_sorted=features_frequency.sort_values('feature_frequency',ascending=False)
features_frequency_sorted.head(50)
sns.barplot(x="feature", y="feature_frequency", data=features_frequency.sort_values(by='feature_frequency',ascending=False).head(10))
plt.xticks(rotation=270)
plt.show()
#wordcloud
def worldcloudplot(features_frequency):
    wc1=features_frequency['feature'].tolist()
    wc2=features_frequency['feature_frequency'].tolist()
    wc= dict(zip(wc1,wc2))
    wordcloud = WordCloud(background_color='WHITE', height = 600, width = 800)
    wordcloud.generate_from_frequencies(frequencies=wc)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt.show()
worldcloudplot(features_frequency)

count_attitude=dta1.groupby(['predict'],as_index=False).count()
#timeseries of negative and positive

pos = dta_final[dta_final['predict']==3]
avg = dta_final[dta_final['predict']==2]

count_in_day_avg = avg.groupby(['date'],as_index=False).count()
count_in_day_avg['date'] = pd.to_datetime(count_in_day_avg['date'])
count_in_day_pos = pos.groupby(['date'],as_index=False).count()
count_in_day_pos['date'] = pd.to_datetime(count_in_day_pos['date'])
date_format = mpl.dates.DateFormatter("%y-%m")
ax = plt.gca()
ax.xaxis.set_major_formatter(date_format)
plt.plot(count_in_day_avg['date'], count_in_day_avg['predict'],label=avg)
plt.plot(count_in_day_pos['date'], count_in_day_pos['predict'],label=pos)
plt.title('positive comment about #hopeww from twitter')
plt.axis('tight')