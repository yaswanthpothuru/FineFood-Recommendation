import findspark
findspark.init()
import pyspark
import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql.functions import *
import json
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer as cv
import pyspark.sql.functions as f
import numpy as np
import pandas
import re
import string
from wordcloud import WordCloud, STOPWORDS
import nltk
from pandas import DataFrame

#initialize spark session
spark = SparkSession \
    .builder \
    .appName("Phone Book - Country Look up") \
    .config("spark.some.config.option", "some-value") \
    .config("spark.sql.caseSensitive", "false")\
    .getOrCreate()
spark.conf.set('spark.sql.caseSensitive', False)

#read the data
pdf=pandas.read_excel("uncleaned_summary.xlsx")

#clean the data and use nltk to tokenize the words
cleanup_re = re.compile('[^a-z]+')
def cleanup(sentence):
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    sentence = " ".join(nltk.word_tokenize(sentence))
    return sentence

#Apply the cleanup function to the data
pdf['Summary_Clean'] = pdf['Summary'].apply(cleanup)
pdf=pdf.drop_duplicates(['Score'], keep='last')
pdf=pdf.reset_index()
df=spark.createDataFrame(pdf)

#Create theTokenizer and transform the data
tokenizer=Tokenizer(inputCol="Summary",outputCol="words")
words=tokenizer.transform(df)
words.show()

#Using the CountVectorizer to create the word count by fiting and transforming the data
count=CountVectorizer(inputCol="words",outputCol="rawFeatures")
model=count.fit(words)
result=model.transform(words)
result.show()

#Using the IDF to create the TF-IDF by fiting and transforming the data
idf=IDF(inputCol="rawFeatures",outputCol="features")
idfModel=idf.fit(result)
rescaledData=idfModel.transform(result)
rescaledData.show()
rescaledData=ps.DataFrame(rescaledData)
rescaledData.to_excel("summary_features.xlsx")


#Extracting Words and their frequencies
docs=pdf['Summary_Clean']
vect=cv(max_features=100, stop_words='english')
X=vect.fit_transform(docs)


#Exporting Countvectorizer data to pandas dataframe
df1 = DataFrame(X.A, columns=vect.get_feature_names())
df1=df1.astype(int)
#Exporting in Excel
df1.to_excel("WordScores.xlsx")








