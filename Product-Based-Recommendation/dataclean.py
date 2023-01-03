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
count = df.groupby("ProductId", as_index=False).count()
mean = df.groupby("ProductId", as_index=False).mean()

#merge two dataset create df1
df1 = ps.merge(df, count, how='right', on=['ProductId'])


#rename column
df1["Count"] = df1["UserId_y"]
df1["Score"] = df1["Score_x"]
df1["Summary"] = df1["Summary_x"]

#Create New datafram with selected variables
df1 = df1[['ProductId','Summary','Score',"Count"]]
df1 = df1.sort_values('Count', ascending=False)

#Less than 5 reviews
print(df1[df1.Count < 2].count())

#Get the data whose count is greater than 5
df2 = df1[df1.Count >= 2]



#create new dataframe as combining all summary with same product Id
df4 = df.groupby("ProductId", as_index=False).mean()
combine_summary = df2.groupby("ProductId")["Summary"].apply(list)
combine_summary = ps.DataFrame(combine_summary)
combine_summary.to_excel("combine_summary.xlsx")

#create with certain columns
df3 = ps.read_excel("combine_summary.xlsx")
df3 = ps.merge(df3, df4, on="ProductId", how='inner')
df3.to_excel("uncleaned_summary.xlsx")

