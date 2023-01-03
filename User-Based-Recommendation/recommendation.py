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
import numpy as np
import pandas
import re
import string
from wordcloud import WordCloud, STOPWORDS
import nltk     
import csv
import math
from numpy.linalg import norm

import warnings
warnings.filterwarnings('ignore')

#Initialize the spark session
spark = SparkSession \
    .builder \
    .appName("Phone Book - Country Look up") \
    .config("spark.some.config.option", "some-value") \
    .config("spark.sql.caseSensitive", "false")\
    .getOrCreate()
spark.conf.set('spark.sql.caseSensitive', False)

#Create a own KNN model using euclidean distance
def kNearestNeighborClassifier(pdataset,point,k):
  results = {}
  for point_item in pdataset:  
    ecludian_distance=math.sqrt(np.sum(np.subtract(point_item,point)*np.subtract(point_item,point)))
    if len(results)<k:
      results[ecludian_distance] = point_item
    else:
      for max_key in sorted(results.keys(),reverse=True):
              if(max_key>=ecludian_distance):
                results[ecludian_distance]=point_item
                results.pop(max_key)
              break
  return results

  #Create a own KNN model using Cosine Similarity
def kNNCosine(pdataset,point,k):
  results = {}
  for point_item in pdataset:  
    cosine_distance=np.dot(point_item,point)/(norm(point_item)*norm(point))
    if len(results)<k:
      results[cosine_distance] = point_item
    else:
      for max_key in sorted(results.keys(),reverse=True):
              if(max_key>=cosine_distance):
                results[cosine_distance]=point_item
                results.pop(max_key)
              break
  return results

#Read the data
pdf=pandas.read_excel("WordScores.xlsx")
raw=pandas.read_excel("summary_features.xlsx")
rev=pandas.read_csv("Reviews.csv")
rev=rev.drop_duplicates(['Summary'], keep='last')
rev["Score"]=rev["Score"].astype(int)
rev=rev.reset_index()

pdflen=len(pdf)
revlen=len(rev)
print(revlen)
print(pdflen)

#Create a Excel file and add the headers in append mode
with open('User_Recommendation.csv', 'a') as file:
    writerObj = csv.writer(file)
    writerObj.writerow(['User','NearestUser', 'Products'])

#Apply the KNN model to the data Using Cosine Similarity  
for i in range(pdflen):
  point=pdf.iloc[i,0]
  results=kNNCosine(pdf.iloc[i],point,11)
  first_related_user=[item for item in results.values()] 
  #Initialize the list and append the related products to the list
  products_list=[]
  for j in first_related_user:
     products_list.append(raw.iloc[j,3])
  if len(products_list)>2:
    products_list=products_list[:1]+products_list[2:]
  products_list=products_list[0:2]


#Apply the KNN model to the data using Euclidean Distance
for i in range(pdflen):
  point=pdf.iloc[i,0]
  results=kNearestNeighborClassifier(pdf.iloc[i],point,11)
  first_related_user=[item for item in results.values()]
  #Initialize the list and append the related products to the list
  products_list=[]
  for j in first_related_user:
     products_list.append(raw.iloc[j,3])
  if len(products_list)>2:
    products_list=products_list[:1]+products_list[2:]
  products_list=products_list[0:2]
  for k in range(revlen):
    if (rev["UserId"][k]==products_list[1] and rev["Score"][k]>3):
      products_list.append(rev["ProductId"][k])
  #Open the file in append mode and write the data to the file
  with open('User_Recommendation.csv', 'a') as file:
    writerObj = csv.writer(file)
    writerObj.writerow(products_list)
  

      



  
  











