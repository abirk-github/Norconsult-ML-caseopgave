import pandas as pd 
import missingno as msno
import matplotlib.pyplot as plt



df = pd.read_csv('TrainAndValid.csv', low_memory=False)

# Show columns for data overview 
print(df.columns)

# Create correlation matrix wrt SaleDate for checking compatibility with linear regression
correlation_whole_set = df.corr(method="pearson")['SalePrice'][:]
print("Correleation for whole set wrt to SalePrice", correlation_whole_set)
print("Correlation size", correlation_whole_set.shape)

# Finding the parameters with predominantly missing data, and discarded them
print(df.isna().sum())

# Visualizing the missing data. Hydraulics feature is seemingly be imputable. 
msno.matrix(df, labels=True)

# Removing incomplete features and SalesID. The latter is removed as it is irrelevant for sale price prediction.
df = df.drop(columns=['MachineHoursCurrentMeter', 'UsageBand', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor',
'ProductSize', 'Drive_System', 'Forks', 'Pad_Type', 'Ride_Control', 'Stick', 'Transmission',
'Turbocharged', 'Blade_Extension', 'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Pushblock', 
'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size', 'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow',
'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb', 'Pattern_Changer', 'Grouser_Type',
'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls', 'Differential_Type', 'Steering_Controls', 'auctioneerID', 
'SalesID'])

# Imputing the missing values (NaN) with the median value, i.e. 2 Valve. 
print(df['Hydraulics'].value_counts(dropna=False))
df['Hydraulics'] = df['Hydraulics'].fillna('2 Valve')
print(df['Hydraulics'].value_counts(dropna=False))
msno.matrix(df, labels=True)

# The remaining relevant features in the data set are as follows:
# SalePrice, MachineID, ModelID, datasource, saledate, fiModelDesc, fiBaseModel, fiProductClass, state
# ProductGroup, ProductGroupDesc, Enclosure, Hydraulics

# Write changed in df to a seperate csv file named task2.csv
df.to_csv('task2.csv')

plt.show()