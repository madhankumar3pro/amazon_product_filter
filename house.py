# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 19:32:06 2022

@author: Admin
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import pylab
##%matplotlib inline
import statsmodels.api as smf
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler
import scipy.stats as stats
import statsmodels

house = pd.read_excel(r"C:\Users\Admin\Desktop\shake labs\DS - Assignment Part 1 data set.xlsx")
house.columns


house.describe()

house = house.drop(['Transaction date','latitude', 'longitude'],axis=1)

# MODULE 1
### Load all the packages required
#null values detection
house.isnull().sum()
#duplicates detection
c=house.duplicated()
c.value_counts()#In this data there is no null values are there

# MODULE 2
# Exploratory Data Analysis
## Measures of Central Tendency / First moment business decision
################### 1. Mean #######################
house.mean()
#################### 2.Median #################"""
house.median()
# Measures of Dispersion / Second moment business decision
################# Variance #################
house.var()
#CWT feature has some much good information is there
################### Stdev ###################"""
house.std()
#CWT and OC features has there is some good information 
# Third moment business decision
################# Skew ####################
house.skew()
#Fourth Moment Business Decision
############## kurtosis #################
house.kurt()
# Visualizations
## Univariate analysis
house.hist()
### Finding ouliers"""
for i in house.iloc[:,:].columns:
  sns.boxplot(house[i])
  plt.show()     ##
ax=sns.boxplot(data= house,orient="h")
sns.boxplot(house["Distance from nearest Metro station (km)"])
# Module 3
### Data Preprocessing
duplicate = house.duplicated()
duplicate
sum(duplicate)
house.columns
### Outlier Treatment"""
# Detection of outliers (find limits for Distance from nearest Metro station (km) based on IQR)
IQR = house['Distance from nearest Metro station (km)'].quantile(0.75) - house['Distance from nearest Metro station (km)'].quantile(0.25)
lower_limit = house['Distance from nearest Metro station (km)'].quantile(0.25) - (IQR * 1.5)
upper_limit = house['Distance from nearest Metro station (km)'].quantile(0.75) + (IQR * 1.5)

outliers_house = np.where(house['Distance from nearest Metro station (km)'] > upper_limit, True, np.where(house['Distance from nearest Metro station (km)'] < lower_limit, True, False))
house_trimmed = house.loc[~(outliers_house), ]
house.shape, house_trimmed.shape

# let's explore outliers in the trimmed dataset
sns.boxplot(house_trimmed['House price of unit area'])
plt.title('Boxplot')
plt.show()

############### 2.Replace ###############
# Now let's replace the outliers by the maximum and minimum limit
house['Distance from nearest Metro station (km)'] = pd.DataFrame(np.where(house['Distance from nearest Metro station (km)'] > upper_limit, upper_limit, np.where(house['Distance from nearest Metro station (km)'] < lower_limit, lower_limit, house['Distance from nearest Metro station (km)'])))
sns.boxplot(house['Distance from nearest Metro station (km)'])
plt.title('Boxplot')
plt.show()
# Check for Missing Values
########### check for count of NA'sin each column
house.isna().sum()
# Visualizations
# Pair Plot
sns.pairplot(house)
# Heat map after Standardization"""
fig, ax = plt.subplots(figsize=(20,20))       
sns.heatmap(house.corr(),annot=True, linewidths=.5, ax=ax)
### Model #######

house['Number of bedrooms'].unique()
house['Number of bedrooms'].value_counts()
### Dropping allplant column"""

colnames = list(house.columns)
### FINDING CORRELATION COEFFICIENT BETWEEN DIFFERENT FEATURES"""
correlation = house.corr()
predictors = colnames[:5]
target = colnames[5]

house.to_csv()

predictors=house.iloc[:,:5]
target=house.iloc[:,5]

#Our predictors are continuos that's why transform data to particular range (0,1)
Scaler=MinMaxScaler()
predictors=Scaler.fit_transform(predictors)

#***********************************Model building********************************************
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(predictors,target,test_size=0.2)
#test accuracy
MODEL1=DT(criterion='mse',random_state=0)
MODEL1.fit(x_train,y_train)
predi=MODEL1.predict(x_test)
MODEL1.score(x_test, y_test)
predi

#train accuracy
MODEL1.fit(x_test,y_test)
predi1=MODEL1.predict(x_train)
tree = MODEL1.score(x_train, y_train)

#decision tree model is not overfitted and under fitted


************************Random forest model*********************************
from sklearn import ensemble
Random=ensemble.RandomForestRegressor(n_estimators=200, criterion='mse', max_depth=5, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
#test accuracy
Random.fit(x_train,y_train)
pred2=Random.predict(x_test)
Random.score(x_test, y_test)

#train accuracy
Random.fit(x_test,y_test)
pred12=Random.predict(x_train)
random = Random.score(x_train, y_train)

**************************Knn**************************************
#test accuracy
from sklearn.neighbors import KNeighborsRegressor  
reg1= KNeighborsRegressor(n_neighbors=10)  
reg1.fit(x_train, y_train)  
pred123=reg1.predict(x_test)
reg1.score(x_test, y_test)



#train accuracy
reg1.fit(x_test, y_test)  
pred1234=reg1.predict(x_train)
knn = reg1.score(x_train, y_train)


******************************GRADIENT BOOSTER REGRESSOR****************************
from sklearn.ensemble import GradientBoostingRegressor
grade=GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
#test errors
grade.fit(x_train,y_train)
pyu5=grade.predict(x_test)
grade.score(x_test, y_test)
error5=y_test-pyu5
error5.mean()

#train errors
grade.fit(x_test,y_test)
pyu6=grade.predict(x_train)
grade = grade.score(x_train, y_train)
error7=y_train-pyu6
error7.mean()



************************Linear model**************************************************
from sklearn import linear_model, metrics
reg = linear_model.LinearRegression()
  
# train the model using the training sets
reg.fit(x_train, y_train)
  
# regression coefficients
print('Coefficients: ', reg.coef_)
  
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(x_test, y_test)))


Li = [tree,random,knn,grade,linear]

plt.scatter(reg.predict(x_train), reg.predict(x_train) - y_train,
            color = "green", s = 10, label = 'Train data')
  
## plotting residual errors in test data
plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
  
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
  
## plotting legend
plt.legend(loc = 'upper right')
  
## plot title
plt.title("Residual errors")
  
## method call for showing the plot
plt.show()

a=house.columns
a=a[0:5]

from statsmodels.stats.outliers_influence import variance_inflation_factor




# VIF dataframe
vif_data = pd.DataFrame()
house1 = house[a]
vif_data["feature"] = house1.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(house1.values, i)
                          for i in range(len(house1.columns))]
  
print(vif_data)

#Due to vif value house is highly correlated  to Number of bedrooms 

k=reg.fit(x_train[:,:4], y_train)
  
# regression coefficients
print('Coefficients: ', k.coef_)
  
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(k.score(x_test[:,:4], y_test)))
linear = k.score(x_test[:,:4],y_test)

Accuracy =pd.DataFrame([['Decision_tree',tree],['GRADIENT BOOSTER REGRESSOR',grade],['Linear model',linear],['Random forest model',random],['KNN',knn]],columns=['Algorithm','Accuracy'])





#**********************Second Question ************************
#Using ML/DL techniques, match similar products from the Flipkart dataset with the Amazon dataset. Once
#similar products are matched, display the retail price from FK and AMZ side by side. Please explore as
#many techniques as possible before choosing the final technique.
#You may either display the final result in single table format OR You may create a simple form where we
#input the product name and the output of prices of the product from both websites are displayed


amazon = pd.read_csv(r'C:\Users\Admin\Desktop\shake labs\amz_com-ecommerce_sample.csv',encoding='unicode_escape')
amazon.isna().sum()
flipcart = pd.read_csv(r'C:\Users\Admin\Desktop\shake labs\flipkart_com-ecommerce_sample.csv',encoding='unicode_escape')

amazon_flipcart = pd.merge(flipcart,amazon,left_on= "uniq_id", right_on= "uniq_id", how= "inner")
amazon_flipcart.drop_duplicates()
amazon_flipcart.columns

#'product_name_x','retail_price_x', 'discounted_price_x','product_name_y', 'retail_price_y',
#'discounted_price_y'
flipcart_amazon = amazon_flipcart[['product_name_x','retail_price_x','discounted_price_x','product_name_y', 'retail_price_y','discounted_price_y']]
flipcart_amazon=flipcart_amazon.dropna()


flipcart_amazon1 = flipcart_amazon.rename(columns={'product_name_x':'Product name in Flipkart',
                                'retail_price_x':'Retail Price in Flipkart',
                                'discounted_price_x':'Discounted Price in Flipkart',
                                'product_name_y':'Product name in Amazon',
                                'retail_price_y':'Retail Price in Amazon',
                                'discounted_price_y':'Discounted Price in Amazon'})

products = flipcart_amazon1['Product name in Flipkart'].unique()


a =str(input())

pro = pd.DataFrame(flipcart_amazon1[flipcart_amazon1['Product name in Flipkart'] == a])

print(pro)
