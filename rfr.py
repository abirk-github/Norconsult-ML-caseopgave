import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split
from multilabel import multilabel_train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle

df = pd.read_csv('task3.csv', low_memory=False)

# Divide the data into "features" and “labels”. 
# X variable contains all the features and y variable contains labels.
X = df[['MachineID', 'saledate', 'fiModelDesc', 'fiBaseModel', 'fiProductClassDesc', 
'ProductGroup', 'Enclosure', 'Hydraulics', "YearMade", 'ProductGroupDesc', 'ModelID', 'state']]
y = df['SalePrice'].values

# List of features for later use
feature_list = list(X.columns)


# Split 80% of the data to the training set, while 20% of the data to test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
regressor = RandomForestRegressor()  
regressor.fit(X_train, y_train)

# Inspect what coefficients the regression model has chosen
#coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
#print(coeff_df)

# Prediction on test data
y_pred = regressor.predict(X_test)

# Evaluate error with cross-validation error
scores = cross_val_score(regressor, X, y, cv=10, scoring='neg_mean_absolute_error')
print('Cross-Validation scores: ', scores)

# Check the difference between the actual value and predicted value
dataset = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
dataset1 = dataset.head(25)

dataset1.plot(kind='bar',figsize=(10,8))  
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()

# Metrics: Evaluation of ML algorithm performance
# 1. Mean Absolute Error  
#    The average of the difference between the Original Values and the Predicted Values. 
#    It gives us the measure of how far the predictions were from the actual output. 
#    However, they don’t gives us any idea of the direction of the error
#    i.e. whether we are under predicting the data or over predicting the data. 

# 2. Mean Squared Error  
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

# Performance metrics
errors = abs(y_pred - y_test)
print('Metrics for Random Forest Trained on Expanded Data')
print('Average absolute error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
mape = np.mean(100 * (errors / y_test))

# Calculate and display accuracy
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(regressor.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# save the model to disk
filename = 'model.sav'
pickle.dump(regressor, open(filename, 'wb'))

# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance') 
plt.xlabel('Variable') 
plt.title('Variable Importances')

plt.show()
