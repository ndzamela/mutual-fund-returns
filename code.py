# --------------
# import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# Code starts here

# load data
data =pd.read_csv(path)
print(data.shape)

#Summary Stats
print(data.describe())

#Drop column Serial Number
data.drop(['Serial Number'], axis=1, inplace=True)

# code ends here




# --------------
#Importing header files
from scipy.stats import chi2_contingency
import scipy.stats as stats

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 11)   # Df = number of variable categories(in purpose) - 1

# Code starts here
return_rating = data['morningstar_return_rating'].value_counts()

risk_rating = data['morningstar_risk_rating'].value_counts()

observed = pd.concat([return_rating.transpose(),risk_rating.transpose()], axis=1,keys= ['return','risk'])

#observed=pd.concat([yes.transpose(),no.transpose()], 1,keys=['Yes','No'])

chi2, p, dof, ex = chi2_contingency(observed)

print("Critical value")
print(critical_value)

print("Chi Statistic")
print(chi2)

print("p-value: ", p)

# Code ends here


# --------------
# Code starts here


# code ends here
correlation = abs(data.corr())
#print(correlation)

us_correlation = correlation.unstack()
us_correlation = us_correlation.sort_values(ascending=False)

max_correlated = (us_correlation > 0.75) & (us_correlation < 1)
print(max_correlated)

data.drop(['morningstar_rating'], axis=1, inplace=True)
data.drop(['portfolio_stocks'], axis=1,inplace=True)
data.drop(['category_12'], axis=1,inplace=True)
data.drop(['sharpe_ratio_3y'], axis=1, inplace=True)
#raw =corr[(corr.abs()>0.75) & (corr.abs() < 1.0)]


# --------------
# Code starts here

fig,(ax_1,ax_2) = plt.subplots(1,2,figsize=(15,8))

ax_1 = data.boxplot(column='price_earning')
ax_1.set(title ='price_earning')

ax_2 = data.boxplot(column='net_annual_expenses_ratio')
ax_2.set(title ='net_annual_expenses_ratio')

plt.show()
# code ends here


# --------------
# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

# Code starts here

#Store all the features
X = data.drop('bonds_aaa', axis =1)
print(X.shape)
# Store target variable bonds_aaa (dependent value)
y = data['bonds_aaa']
print(y.shape)
#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 3)

#Initiate model
lr = LinearRegression()

#Fit the mode
lr.fit(X_train, y_train)

#Make predictions
y_pred = lr.predict(X_test)

#Calculate the root mean squared error 
rmse = np.sqrt(mean_squared_error(y_pred,y_test))
print(rmse)
# Code ends here


# --------------
# import libraries
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge,Lasso

# regularization parameters for grid search
ridge_lambdas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60]
lasso_lambdas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1]

# Code starts here
#Initiate a model with Ridge() 
ridge_model = Ridge()

#grid search ridge, fit, predicict
ridge_grid = GridSearchCV(estimator=ridge_model,param_grid=dict(alpha=ridge_lambdas))
ridge_grid.fit(X_train,y_train)
ridge_pred = ridge_grid.predict(X_test)
# Calculate Ridge rmse
ridge_rmse = np.sqrt(mean_squared_error(ridge_pred, y_test))
print(ridge_rmse)

#Initiate a model with Lasso() 
lasso_model = Lasso()

#grid search lasso, fit, predicict
lasso_grid = GridSearchCV(estimator=lasso_model,param_grid=dict(alpha=lasso_lambdas))
lasso_grid.fit(X_train,y_train)
lasso_pred = lasso_grid.predict(X_test) 

# Calculate Lasso rmse
lasso_rmse = np.sqrt(mean_squared_error(lasso_pred, y_test))
print(lasso_rmse)

# Code ends here


