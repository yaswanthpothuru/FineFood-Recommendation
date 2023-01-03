import findspark
findspark.init()
import pyspark
import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql.functions import *
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
def kNN(pdataset,point,k):
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

pdflen=len(pdf)

#Create a Excel file and add the headers in append mode
with open('Product_Recommendation.csv', 'a') as file:
    writerObj = csv.writer(file)
    writerObj.writerow(['Product','Recommended Products'])

#Apply the KNN model to the data using Cosine Similarity
for i in range(pdflen):
  point=pdf.iloc[i,0]
  results=kNNCosine(pdf.iloc[i],point,11)
  first_related_product=[item for item in results.values()]
  #Initialize the list and append the related products to the list
  products_list=[]
  for j in first_related_product:
     products_list.append(raw.iloc[j,3])
  products_list=products_list[:1]+products_list[2:]

#Apply the KNN model to the data using Euclidean distance
for i in range(pdflen):
  point=pdf.iloc[i,0]
  results=kNN(pdf.iloc[i],point,11)
  first_related_product=[item for item in results.values()]
  #Initialize the list and append the related products to the list
  products_list=[]
  for j in first_related_product:
     products_list.append(raw.iloc[j,3])
  products_list=products_list[:1]+products_list[2:]
  #Open the Excel file and write the data in append mode
  with open('Product_Recommendation.csv', 'a') as file:
    writerObj = csv.writer(file)
    writerObj.writerow(products_list)
   











