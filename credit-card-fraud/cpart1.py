# Add the data file to hdfs.
# !hdfs dfs -put resources/creditcard.csv /tmp/lufeng

### Part 1 Introduction
# In this part of the project:
# * `Using PySpark to load the data`
# * `Do some transformation`
# * `Simple SQL query, and then visulization`
# * `Save the data`


from __future__ import print_function
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd 
from matplotlib import pyplot as plt

## Create a Spark Session
spark = SparkSession.builder.appName("creditcard").getOrCreate()

## Load the CSV data
df = spark.read.csv("/tmp/lufeng/creditcard.csv", header=True, inferSchema=True)
df.cache()

## Print schema
df.printSchema()

## Print the sample data. 
df.show()

# We take 5 record and transform it into pandas DataFrame, 
# and then transpose
pd.DataFrame(df.take(5), columns=df.columns).T

## Summary Statistics
df.describe().toPandas().T

## check the number of fraud transaction
df.groupBy("Class").count().orderBy("Class").show()


## Add a new column 'Hour'= Time / 3600 % 24 
df = df.withColumn("Hour", (df.Time / 3600).cast("int") % 24 )
df.select("Time","Hour").show()
df.groupBy("Hour").count().orderBy("Hour").show(25)


## Add a new column 'AmountBin'
# divide the Amount into several group, just like cut in pandas
df = df.withColumn('AmountBin', F.when(df.Amount > 500, '>500') \
                               .when(df.Amount > 400, '400 - 500') \
                               .when(df.Amount > 300, '300 - 400') \
                               .when(df.Amount > 200, '200 - 300') \
                               .when(df.Amount > 100, '100 - 200') \
                               .when(df.Amount > 50, '50 - 100') \
                               .when(df.Amount > 10, '10 - 50') \
                               .otherwise('<10'))

# check the transaction volumns group by AmountBin
df.groupBy("AmountBin").count().sort("count", ascending=False).show()

# check the datatype of newly added columns
df.dtypes

## Execute sql query, and then visulize the result

# Register a temp table fraud
df.registerTempTable("fraud")

# check the transaction volumns grouped by Class and Hour
group = spark.sql("select Class, Hour, count(1) as Volumn from fraud group by Class, Hour order by Class, Hour").toPandas()
group

# generate the pivot table
pivot = group.pivot(index='Hour',columns='Class', values='Volumn')
pivot

# draw plot
pivot.iloc[:,0].plot(kind='line',style='o-',title='Nomal')
pivot.iloc[:,1].plot(kind='line',style='o-',title='Fraud')

# check how the fraud transaction distributed by AmountBin
spark.sql("select AmountBin, count(1) as volumn from fraud where Class = 1 group by AmountBin order by volumn").show(100)

## save file
df.write.csv(path="hdfs:///tmp/lufeng/cpart1",mode="overwrite",header=True)



#!hdfs dfs -cat /tmp/lufeng/cpart1/part-00000-40dd8340-2273-449f-adff-e479c8f764ea.csv | head






