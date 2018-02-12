### Part 2 Introduction
# In this part of the project:
# * `Using pandas to load and analyze the data`
# * `Visulization with matplotlib and seaborn`
# * `feature engineering`
# * `inbalanced sample data`
# * `Using sklearn to train a Logistic Regression Classification model`
# * `Model tuning`

import numpy as np
import pandas as pd
from pandas import DataFrame

from matplotlib import pyplot as plt
import seaborn as sns

## Load data
data = pd.read_csv('resources/creditcard.csv')

## Describe data
# * the name of each column
# * the number of values in each column
# * the number of missing/NaN values in each column; 
# * the contents of the first 5 rows
# * the contents of the last 5 rows
# * the basic stats of each column
data.columns.values
data.info()
data.isnull().sum()
data.head()
data.tail()
data.describe().T

## Frequency of Fraud/Normal
fraud_count = data['Class'].value_counts()
fraud_count
fraud_count.plot.bar(color='r',title='Frequency of Fraud/Normal')

## Percentage of Fraud/Normal
fraud_count_n = data['Class'].value_counts(normalize=True)
fraud_count_n
fraud_count_n.plot.pie(title='Percentage of Fraud/Normal', \
                       labels = ['Normal', 'Fraud'], \
                       colors = ['g', 'r'], \
                       autopct='%.2f')


## Generate two new columns 'Day' and 'Hour'
# Since the column 'Time' is second, we need to  
# add new column 'Hour', calculate by Hour = Time / 3600 % 24
# data['Time'].apply(lambda x : int(x / 3600 % 24)).value_counts(sort=False)

# However, transaction data is from two days. We need to 
# calculate day and Hour  
cal_hour = lambda x : int(x / 3600 % 24)
cal_day  = lambda x : 'Day 1' if int(x/3600) > 23 else 'Day 2'

data['Day'] = data['Time'].apply(cal_day)
data['Hour'] = data['Time'].apply(cal_hour)

## Plot transaction freqence and amount
# what we can see here is that busy time is from 9am to 10pm
data['Hour'].value_counts(sort=False)
data['Hour'].value_counts(sort=False).plot.bar()

# However, we don't know if Day make any difference
# Plot transaction freqence by Day and Hour
grouped_count = data.groupby(['Hour','Day',])['Amount'].count().unstack()
grouped_count.head()
grouped_count.plot(title='Transaction Freq',style=['o-','v-'])
grouped_count.plot.bar(title='Transaction Freq',stacked=True)

# Plot transaction amount by Day and Hour
grouped_amount = data.groupby(['Hour','Day',])['Amount'].sum().unstack()
grouped_amount.head()
grouped_amount.plot(title='Transaction Amount',style=['o-','v-'])
grouped_amount.plot.bar(title='Transaction Amount',stacked=True)

# Now, we want to know fraud transaction freqence and amount grouped 
# by Day and Hour

# Get the fraud only data
fraud = data[data['Class'] == 1]
fraud[['Time','Amount','Class']].head()

# What we found here is that a lot of fraud happed in both 2am and 11am
# This is because people sleep on 2am, and spend lots of mony on 11am
# It is difficult for people to find out the fraud at that time since they
# are either in the dream or buy 
f_grouped_count = fraud.groupby(['Hour','Day',])['Amount'].count().unstack()
f_grouped_count.head()
f_grouped_count.plot(title='Fraud Freq',style=['o-','v-'])
f_grouped_count.plot.bar(title='Fraud Freq',stacked=True)

# If we calculte the fraud dollar amount, almost same as frenqucy
f_grouped_amount = fraud.groupby(['Hour','Day',])['Amount'].sum().unstack()
f_grouped_amount.head()
f_grouped_amount.plot(title='Fraud Amount',style=['o-','v-'])
f_grouped_amount.plot.bar(title='Fraud Amount',stacked=True)


# Plot Fraud distribution by time
# Fraud with more money intend to happed during the day time
def plot_fraud_dist(data):
  fig, axes = plt.subplots(2,1,sharex=True)
  fraud = data[data["Class"] == 1]
  normal = data[data["Class"] == 0]
  axes[0].scatter(fraud["Hour"], fraud["Amount"])
  axes[0].set_title('Fraud')

  axes[1].scatter(normal["Hour"], normal["Amount"])
  axes[1].set_title('Normal')

  plt.xlabel('Time (in Hours)')
  plt.ylabel('Amount')
  plt.show()

plot_fraud_dist(data)

## Divide amount into several group
# What we found here is that most fraud is small by dollor value
# People usually ignore the small money they spent
fraud = data[data['Class'] == 1].copy()
fraud.head()
fraud['Amount'].describe()

fraud['Cut'] = pd.cut(fraud['Amount'], bins=[-0.1,10,50,100,200,300,400,3000])
fraud.groupby('Cut')['Amount'].count().plot.bar()



## check correlation among variables
# Correlation Plot is not really important here 
# as input data is already a PCA tranformation
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 1]

corr_fraud = fraud.loc[:, ~fraud.columns.isin(['Class','Day'])].corr()
sns.heatmap(corr_fraud,cmap="BrBG",annot=False)

corr_normal = normal.loc[:, ~fraud.columns.isin(['Class','Day'])].corr()
sns.heatmap(corr_normal,cmap="BrBG",annot=False)

## distplot for V1 and V28
# What we oberse here is that the ditribution for varibles 
# V8, V13,V15,V20,V22,V23,V25,V26,V27,V28 has no big  difference 
# between fraud and normal data set.
# We may consider remove these variables during the feature selection phase
features = data.iloc[:,1:29].columns
plt.figure(figsize=(16,28*4))
for i, cn in enumerate(data[features]):
    ax = plt.subplot()
    sns.distplot(data[cn][data["Class"] == 1], bins=50, color='r')
    sns.distplot(data[cn][data["Class"] == 0], bins=100, color='g')
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
    plt.show()

## Feature selection
# We also need to remove Time, Day, Cut
# droplist = ['Time','Day','V8','V13','V15','V20','V22','V23','V25','V26','V27','V28']
droplist = ['Time', 'Day']
data2 = data.drop(droplist, axis=1)
data2.shape
data2.columns

## Feature preprocessing
# Two feature Amoount and Hour need to be standardized
from sklearn.preprocessing import StandardScaler
c = ['Amount','Hour']
data2[c] =StandardScaler().fit_transform(data2[c])
data2.head().T

## imbalanced sample data
data2['Class'].value_counts(normalize=True)

# Random under sampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()
X = data2.drop(['Class'], axis=1).values
y = data2['Class'].values
X_res, y_res = rus.fit_sample(X, y)

# check the undersampling result
X_res.shape
y_res.shape
import collections
collections.Counter(y_res)

X = X_res
y = y_res

## Train a classification model

# To solve a bug from Anaconda. 
# When running skilearn, get the following erro message
# Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so.
#
# Solution is:
# export LD_PRELOAD=/opt/cloudera/parcels/Anaconda/lib/libmkl_def.so:/opt/cloudera/parcels/Anaconda/lib/libmkl_avx.so:/opt/cloudera/parcels/Anaconda/lib/libmkl_core.so:/opt/cloudera/parcels/Anaconda/lib/libmkl_intel_lp64.so:/opt/cloudera/parcels/Anaconda/lib/libmkl_intel_thread.so:/opt/cloudera/parcels/Anaconda/lib/libiomp5.so

# LogisticRegression estimator
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# dateset split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=1234)
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

# better approach is cross validation
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(lr, X, y, cv=10)
scores

# ROC curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
y_pred1_prob = lr.predict_proba(X_test)[:, 1] 

fpr, tpr, thresholds = roc_curve(y_test,y_pred1_prob)
roc_auc = auc(fpr,tpr)

def plot_roc(fpr, tpr):
  plt.title('Receiver Operating Characteristic')
  plt.plot(fpr, tpr, 'b',label='AUC = %0.5f'% roc_auc)
  plt.legend(loc='lower right')
  plt.plot([0,1],[0,1],'r--')
  plt.xlim([-0.1,1.0])
  plt.ylim([-0.1,1.01])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()

plot_roc(fpr, tpr)

# Model tuning

# Change Hyper-Parameters C
from sklearn.cross_validation import KFold
lr = LogisticRegression(C=.0001)
kf = KFold(len(X),n_folds=5, shuffle=True, random_state=1234)
score = cross_val_score(lr, X, y, cv=kf)
print(score)
score.mean()

lr = LogisticRegression(C=1000)
kf = KFold(len(X),n_folds=5, shuffle=True, random_state=1234)
new_scores2 = cross_val_score(lr, X, y, cv=kf)
print(score)
score.mean()

lr = LogisticRegression(C=.5)
kf = KFold(len(X),n_folds=5, shuffle=True, random_state=1234)
new_scores2 = cross_val_score(lr, X, y, cv=kf)
print(score)
score.mean()

# Automate the tuning process
from sklearn.grid_search import GridSearchCV

param_grid = {'C':np.logspace(-5,2, num=100)}
lr = LogisticRegression()
kf = KFold(len(X),n_folds=5, shuffle=True, random_state=1234)
gscv = GridSearchCV(lr, param_grid=param_grid, cv=kf)
gscv.fit(X, y)
gscv.best_params_

lr = LogisticRegression(C=gscv.best_params_['C'])
kf = KFold(len(X),n_folds=5, shuffle=True, random_state=1234)
score = cross_val_score(lr, X, y, cv=kf)
print(score)
score.mean()




































