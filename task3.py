import pandas as pd 
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
#from task2 import df

df = pd.read_csv('task2.csv', low_memory=False)


# The remaining relevant features in the data set are as follows:
# SalePrice, MachineID, ModelID, datasource, saledate, fiModelDesc, fiBaseModel, fiProductClassDesc, state
# ProductGroup, ProductGroupDesc, Enclosure, Hydraulics

# Some of the features are columns that are of the categorical form: 
# E.g. state --> New York, Texas, Florida, etc...
# Intuitionally, if you can group data together in your head fairly easily and represent it with a string, 
# there’s a chance it’s part of a category.
print('Df types', df['state'].dtypes)
# So to train a machine learning model, we’ll need a way of converting these to numbers.
# So the procedure for handling categorical data is to convert them to discrete numerical form before training the model.

# For date time cases there is a separate approach: Transform the datetime feature to integer timestamp
df["saledate"] = pd.to_datetime(df["saledate"]).astype(int)/10**9

# Transform the categorical variables to numerical representation
for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes

# In this case saledate, fiModelDesc, fiBaseModel, fiProductClassDesc, state, ProductGroup, ProductGroupDesc,
# Enclosure, Hydraulics features are of categorical form
# For categorical features different method is applied to find correlation to the numeric value of SalePrice
# This applies for categorical featuers that are not point-biserial correlation data (https://www.researchgate.net/post/Can_I_use_Pearsons_correlation_coefficient_to_know_the_relation_between_perception_and_gender_age_income), 
# i.e. more than two categorical representation in a category feature (e.g. male and female) 
# Hence, Pearsons approach connot be applied here and Cramer’s V method must be used. 
# This is to decide whether to keep or discard the feauture based on the significance of the relationship 

# Source for code below: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

print('YearMade Correlation ratio: ', correlation_ratio(df['YearMade'], df['SalePrice']))
print('saledate Correlation ratio: ', correlation_ratio(df['saledate'], df['SalePrice']))
print('fiModelDesc Correlation ratio: ', correlation_ratio(df['fiModelDesc'], df['SalePrice']))
print('fiBaseModel Correlation ratio: ', correlation_ratio(df['fiBaseModel'], df['SalePrice']))
print('fiProductClassDesc Correlation ratio: ', correlation_ratio(df['fiProductClassDesc'], df['SalePrice']))
print('state Correlation ratio: ', correlation_ratio(df['state'], df['SalePrice']))
print('ProductGroup Correlation ratio: ', correlation_ratio(df['ProductGroup'], df['SalePrice']))
print('ProductGroupDesc Correlation ratio: ', correlation_ratio(df['ProductGroupDesc'], df['SalePrice']))
print('Enclosure Correlation ratio: ', correlation_ratio(df['Enclosure'], df['SalePrice']))
print('Hydraulics Correlation ratio: ', correlation_ratio(df['Hydraulics'], df['SalePrice']))
# Results:
# YearMade Correlation ratio:  0.2775356916687609
# saledate Correlation ratio:  0.2744035534280298
# fiModelDesc Correlation ratio:  0.8645253755564725
# fiBaseModel Correlation ratio:  0.6845477991228243
# fiProductClassDesc Correlation ratio:  0.6469584499524053
# state Correlation ratio:  0.12229622361195866
# ProductGroup Correlation ratio:  0.4343650312727657
# ProductGroupDesc Correlation ratio:  0.4343650312727657
# Enclosure Correlation ratio:  0.48920471534517196
# Hydraulics Correlation ratio:  0.24291670869918697


# The remaining data set yet to be assessed, which is all numerical features, 
# can be applied with Pearsons method.
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


# Setting up a correlation matrix wrt SalePrice
# to see which features can be removed as they wont impact sale price. 
# To do this, it is assumed in this case that the ML method used is based on simple linear classifier 
# (i.e. linear regression)
# https://datascience.stackexchange.com/questions/40602/can-we-remove-features-that-have-zero-correlation-with-the-target-label?fbclid=IwAR1dKS87ELtxcvA-ttIQE1ex8CCVvg3lL9lmkBrTyGNVVamqLZ-a5mctBgk
# Setting up the correlation matrix
#correlation_whole_set = df.corr(method="pearson")['SalePrice'][:]
correlation_whole_set = df.corr(method="pearson")['SalePrice'][['MachineID', 'ModelID', 'datasource']]
print(correlation_whole_set)
# Results from the correlation matrix with numerical features:
# MachineID    -0.216841
# ModelID      -0.038063
# datasource    0.021708

# Here it can be conculded that ModelID and datascource features are quite insignificant 
# and hence dropped from data set for Linear Regression model training
# In addition, the datasource feature does not add value for the end user as it is simply an unknown user
# hence it is left out 
print("Unique Datasource", df["datasource"].unique())
df = df.drop(columns=['datasource'])

# Also:
# Studying the corr matrix shows that the feautures ProductGroup and ProductGroupDesc represent the same variables
# Hence, one of them can be removed so that there is no unneccessary noice in the data set
# In this case the latter is removed
print([df.ProductGroup.unique(), df.ProductGroupDesc.unique()])

# Dropping the assessed features
#df = df.drop(columns=['ProductGroupDesc', 'ModelID', 'datasource', 'state'])


# Remaining features in this dataset are consequently: 
# SalePrice, MachineID, saledate, fiModelDesc, fiBaseModel, fiProductClassDesc, state
# ProductGroup, ProductGroupDesc, Enclosure, Hydraulics

# Write changed in df to a seperate csv file named task3.csv
df.to_csv('task3.csv')




