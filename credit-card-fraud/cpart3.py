### Part 3 Introduction
# In this part of the project:
# * `Using pyspark to load the data`
# * `Imbalanced sample data`
# * `Feature selection and transformation`
# * `ML pipline and train the model`
# * `Model tuning`

from __future__ import print_function
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType
from pyspark.sql import functions as F

import pandas as pd

## Create the Spark Session
spark = SparkSession.builder.appName("creditcard").getOrCreate()

## load the data processed by the part 1 
df = spark.read.csv("/tmp/lufeng/cpart1", header=True, inferSchema=True)
df.cache()

## Describe the data
df.printSchema()
pd.DataFrame(df.take(5), columns=df.columns).transpose()
df.describe().toPandas().transpose()

## Imbalanced sample data
df.groupby('Class').count().toPandas()

# stratified sampling
stratified_df = df.sampleBy('Class', fractions={0: 492./284315, 1: 1.0})
stratified_df.groupby('Class').count().toPandas()

## Feature selection, remove `Time` and `AmountBin`
stratified_df = stratified_df.select("Class",\
                                     "Amount", "Hour", \
                                     "V1","V2","V3","V4","V5", \
                                     "V6","V7","V8","V9","V10", \
                                     "V11","V12","V13","V14","V15", \
                                     "V16","V17","V18","V19","V20", \
                                     "V21","V22","V23","V24","V24", \
                                     "V26","V26","V28")
stratified_df.printSchema()

## Feature Engineering
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StandardScaler, VectorAssembler, VectorIndexer
from pyspark.ml import Pipeline

# Define the `input_data` 
# `Amount` and `Hour` as `feature1`
# `V1` to `V28` as `feature2`
input_data = stratified_df.rdd.map(lambda x: (x[0], Vectors.dense(x[1:3]), Vectors.dense(x[3:])))

# Replace `df` with the new DataFrame
df = spark.createDataFrame(input_data, ["label", "feature1", "feature2"])
df.take(2)

## Build the pipeline using RandomForestClassifier
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# stages in our Pipeline
stages = [] 

# Define the stage to standardize `feature1`
standardScaler = StandardScaler(inputCol="feature1", outputCol="feature1_scaled")
stages += [standardScaler]

# Define the stage to assemble `feature1_scaled` and `feature2` as `features`
assemblerInputs = ["feature1_scaled", "feature2"]
vectorAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [vectorAssembler]

# labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel")

# Define the stage to automatically identify categorical features and index them
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4)
stages += [featureIndexer]

# Split the date set
train_data, test_data = df.randomSplit([.7,.3],seed=1234)

# RandomForestClassifier
rf = RandomForestClassifier(labelCol="label", featuresCol="indexedFeatures", numTrees=10)

# create the pipeline
pipeline = Pipeline(stages=(stages + [rf]))
# train the model
model = pipeline.fit(train_data)

# predict using the test data set
predictions = model.transform(test_data)

# predict result
predictions.select("prediction", "label","probability", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator( \
   labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

# print model summary
rfModel = model.stages[3]
print(rfModel)

## Build the pipeline using LogisticRegression

# LogisticRegression
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# create the pipeline
pipeline = Pipeline(stages=(stages + [lr]))
# train the model
model = pipeline.fit(train_data)

# predict using the test data set
predictions = model.transform(test_data)

# predict result
predictions.select("prediction", "label","probability", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator( \
   labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

# summary only
lrModel = model.stages[3]
print(lrModel)  

## model tuning 
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder

# Search through multiple parameter combination for best model
# total iteration = 3 * 3 * 3 = 27
paramGrid = ParamGridBuilder() \
             .addGrid(lr.regParam, [0.01, 0.5, 2.0]) \
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
             .build()

# Estimation
lr = LogisticRegression(maxIter=10)
# Pipline
pipeline = Pipeline(stages=(stages + [lr]))
# Evaluator
evaluator = MulticlassClassificationEvaluator( \
   labelCol="label", predictionCol="prediction", metricName="accuracy")

# CrossValidation + mutiple parameter combination
crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, numFolds=3)

# train the model
CV_model = crossval.fit(train_data)

# Fetch best model
lrModel = CV_model.bestModel.stages[3]
print(lrModel)

trainingSummary = lrModel.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
trainingSummary.roc.show()
roc_auc = trainingSummary.areaUnderROC
print("areaUnderROC: " + str(roc_auc))


# plot ROC
def plot_roc(roc):
  import matplotlib.pyplot as plt
  plt.title('Receiver Operating Characteristic')
  plt.plot(roc.FPR, roc.TPR, 'b',label='AUC = %0.5f'% roc_auc)
  plt.legend(loc='lower right')
  plt.plot([0,1],[0,1],'r--')
  plt.xlim([-0.1,1.0])
  plt.ylim([-0.1,1.01])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()

plot_roc(trainingSummary.roc.toPandas())
  
# Select (prediction, true label) and compute test error
predictions = CV_model.transform(test_data)
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))



