import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split
from multilabel import multilabel_train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('task3.csv', low_memory=False)

# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f : 
# There are two types of supervised machine learning algorithms: Regression and classification. 
# The former predicts continuous value outputs while the latter predicts discrete outputs. 
# For instance, predicting the price of a house in dollars is a regression problem 
# whereas predicting whether a tumor is malignant or benign is a classification problem.


# Quick assessment on the linear relation between two arbitrary chosen features 
df.plot(x='SalePrice', y='MachineID', style='o')  
plt.title('SalePrice vs MachineID')  
plt.xlabel('SalePrice')  
plt.ylabel('MachineID')  

df.plot(x='SalePrice', y='Enclosure', style='o')  
plt.title('Enclosure vs SalePrice')  
plt.xlabel('SalePrice')  
plt.ylabel('Enclosure')  

# Conclusion: No linear relation --> Linear regression is seemingly a poor choice

# Divide the data into “attributes” and “labels”. 
# X variable contains all the attributes/features and y variable contains labels.
X = df[['MachineID', 'saledate', 'fiModelDesc', 'fiBaseModel', 'fiProductClassDesc', 'ProductGroup', 'Enclosure', 'Hydraulics', "YearMade"]]
y = df['SalePrice'].values

# Check the average value of the SalePrice column
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(df['SalePrice'])

# Split 80% of the data to the training set, while 20% of the data to test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

# Inspect what coefficients the regression model has chosen
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
print(coeff_df)

# Prediction on test data
y_pred = regressor.predict(X_test)

# Check the difference between the actual value and predicted value
dataset = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
dataset1 = dataset.head(25)

dataset1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')


# Metrics: Evaluation of ML algorithm performance
# 1. Mean Absolute Error - 
#    The average of the difference between the Original Values and the Predicted Values. 
#    It gives us the measure of how far the predictions were from the actual output. 
#    However, they don’t gives us any idea of the direction of the error
#    i.e. whether we are under predicting the data or over predicting the data. 

# 2. Mean Squared Error - 
#    Quite similar to Mean Absolute Error, the only difference being that MSE takes the average of the square 
#    of the difference between the original values and the predicted values. 
#    The advantage of MSE being that it is easier to compute the gradient, 
#    whereas Mean Absolute Error requires complicated linear programming tools to compute the gradient. 
#    As, we take square of the error, the effect of larger errors become more pronounced then smaller error, 
#    hence the model can now focus more on the larger errors.


# Evaluate the performance of the algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Not so good results, however some cases show pretty good predictons

plt.show()

# Possible explanation: 
# 1. Bad assumptions: Made the assumption that this data has a linear relationship, 
#    but that might not be the case. Visualizing the data may help you determine that.
# 2. Poor features: The features we used may not have had a high enough correlation 
#    to the values we were trying to predict. 
# 3. Including categorical features in the training and test set



