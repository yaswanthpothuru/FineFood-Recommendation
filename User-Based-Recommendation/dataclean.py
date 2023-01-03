import findspark
findspark.init()
import pyspark
import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql.functions import *
from pandas import DataFrame

#initiate spark session
spark = SparkSession \
    .builder \
    .appName("Phone Book - Country Look up") \
    .config("spark.some.config.option", "some-value") \
    .config("spark.sql.caseSensitive", "false")\
    .getOrCreate()
spark.conf.set('spark.sql.caseSensitive', False)

#read in the csv file
df = ps.read_csv('reviews.csv')

#Conver Score to Integer
df['Score'] = df['Score'].astype('int64')

#Basic Information about the dataset
print(df.info())
print(df["Score"].describe())

#Drop Null Values
df = df.dropna()
print(df.info())


#compute the count and mean value as group by the products
count = df.groupby("UserId", as_index=False).count()
mean = df.groupby("UserId", as_index=False).mean()

#merge two dataset create df1
df1 = ps.merge(df, count, how='right', on=['UserId'])


#rename column
df1["Count"] = df1["ProductId_y"]
df1["Score"] = df1["Score_x"]
df1["Summary"] = df1["Summary_x"]

#Create New datafram with selected variables
df1 = df1[['UserId','Summary','Score',"Count"]]
df1 = df1.sort_values('Count', ascending=False)
df2 = df1[df1.Count >= 5]

#Insights from the dataset2 whohave count more than 5
print(df2.info())
print(df2["Score"].describe())

#create new dataframe as combining all summary with same product Id
df4 = df.groupby("UserId", as_index=False).mean()
combine_summary = df2.groupby("UserId")["Summary"].apply(list)
combine_summary = ps.DataFrame(combine_summary)
combine_summary.to_excel("combine_summary.xlsx")

#create with certain columns
df3 = ps.read_excel("combine_summary.xlsx")
df3 = ps.merge(df3, df4, on="UserId", how='inner')
df3.to_excel("uncleaned_summary.xlsx")

