import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('TrainAndValid.csv', low_memory=False)

# Print out column names (i.e. parameters)
print(df.columns)

# SalePrice vs YearMade
max_YearMade = df['YearMade'].max()
min_YearMade = df['YearMade'].min()
mean_YearMade = df['YearMade'].mean()
YearMade_array = [min_YearMade, mean_YearMade, max_YearMade]
print('min_YearMade, mean_YearMade, max_YearMade', YearMade_array)

data1 = df.groupby('YearMade')['SalePrice'].agg([('low', 'min'), ('avg', 'mean'), ('high', 'max')]).reset_index()

ax  = data1.plot(x='YearMade', y='avg', c='white')
plt.fill_between(x='YearMade',y1='low',y2='high', data=data1)
plt.title("YearMade vs SalePrice")
# Study unique data to check if there is a gap between year 1000 and 1942 which it is,
# hence indicates that year 1000 was filled in for lacking production year of the vehicle
print("Unique YearMade", df.YearMade.unique())

# SalePrice vs YearMade: Corrected df for assumed trustworthy YearMade-range data (i.e. from 1946 and onwards)
df_yearmade_corrected = df[df['YearMade'] >= 1946]
max_YearMade_corrected = df_yearmade_corrected['YearMade'].max()
min_YearMade_corrected = df_yearmade_corrected['YearMade'].min()
mean_YearMade_corrected = df_yearmade_corrected['YearMade'].mean()
print('min_YearMade_corrected, mean_YearMade_corrected, max_YearMade_corrected', min_YearMade_corrected, mean_YearMade_corrected, max_YearMade_corrected)
data2 = df_yearmade_corrected.groupby('YearMade')['SalePrice'].agg([('low', 'min'), ('avg', 'mean'), ('high', 'max')]).reset_index()

ax  = data2.plot(x='YearMade', y='avg', c='white')
plt.fill_between(x='YearMade',y1='low',y2='high', data=data2)
plt.title("YearMade vs SalePrice")

# ProductSize vs SalePrice
small_size = df[df['ProductSize'] == 'Small']
SalePrice_small_min = small_size['SalePrice'].min()
SalePrice_small_mean = small_size['SalePrice'].mean()
SalePrice_small_max = small_size['SalePrice'].max()
SalePrice_small_array = [SalePrice_small_min, SalePrice_small_mean, SalePrice_small_max]
print('SalePrice_small_min, SalePrice_small_mean, SalePrice_small_max', SalePrice_small_array)
#SalePrice_small = df[df['SalePrice'] & small_size]
#is_small_size = df['ProductSize'].isin(['Small'])
#SalePrice_small_min = df.groupby(SalePrice_small)['SalePrice'].mean()
data3 = df.groupby('ProductSize')['SalePrice'].agg([('low', 'min'), ('avg', 'mean'), ('high', 'max')]).reset_index()

ax = data3.plot(x='ProductSize', y='avg', c='white')
plt.fill_between(x='ProductSize',y1='low',y2='high', data=data3)
plt.title("ProductSize vs SalePrice")
#plt.show()

# AuctioneerID vs SalePrice
auctioneerID=df['auctioneerID']
print(auctioneerID)

auctioneerID_min = df['auctioneerID'].min()
auctioneerID_mean = df['auctioneerID'].mean()
auctioneerID_max = df['auctioneerID'].max()
print(auctioneerID_min, auctioneerID_mean, auctioneerID_max)
data4 = df.groupby('auctioneerID')['SalePrice'].agg([('low', 'min'), ('avg', 'mean'), ('high', 'max')]).reset_index()

ax = data4.plot(x='auctioneerID', y='avg', c='white')
plt.fill_between(x='auctioneerID',y1='low',y2='high', data=data4)
plt.title("auctioneerID vs SalePrice")
#plt.show()


# AuctioneerID vs SalePrice where ActioneerID ranges to 28.0 only
corrected_auctioneerID = df[df['auctioneerID']<=28.0]
corrected_auctioneerID_min = corrected_auctioneerID['auctioneerID'].min()
corrected_auctioneerID_mean = corrected_auctioneerID['auctioneerID'].mean()
corrected_auctioneerID_max = corrected_auctioneerID['auctioneerID'].max()
print(corrected_auctioneerID_min, corrected_auctioneerID_mean, corrected_auctioneerID_max)

data5 = corrected_auctioneerID.groupby('auctioneerID')['SalePrice'].agg([('low', 'min'), ('avg', 'mean'), ('high', 'max')]).reset_index()
ax = data5.plot(x='auctioneerID', y='avg', c='white')
plt.fill_between(x='auctioneerID',y1='low',y2='high', data=data5)
plt.title("auctioneerID vs SalePrice")
plt.show()


