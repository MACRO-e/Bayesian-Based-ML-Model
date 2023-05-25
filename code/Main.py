#Importing the libraries
import os
os.chdir('C:/Study/Bayesian/ Bayesian-Based-ML-Model/FACTOR_MODEL/')
import sys
sys.path.append('C:/Study/Bayesian/ Bayesian-Based-ML-Model/FACTOR_MODEL/')

import os
import sys
import pickle
import bz2
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import chi2
from scipy.stats import f
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import patsy
import math
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn import tree
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import neighbors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from statsmodels.formula.api import ols
from scipy.stats import gaussian_kde
import scipy
import scipy.sparse
import patsy
from statistics import median
import bz2
import math

# Importing the dataset
model_dir = 'C:/Study/Bayesian/ Bayesian-Based-ML-Model/FACTOR_MODEL/'

def sort_cols(test): 
    return(test.reindex(sorted(test.columns), axis=1))

frames = {}
for year in [2003,2004,2005,2006,2007]:
    fil = model_dir + "pandas-frames." + str(year) + ".pickle.bz2"
    frames.update(pickle.load( bz2.open( fil, "rb" ) ))

for x in frames: 
    frames[x] = sort_cols(frames[x])
    
covariance = {}
for year in [2003,2004,2005,2006]:
    fil = model_dir + "covariance." + str(year) + ".pickle.bz2"
    covariance.update(pickle.load( bz2.open(fil, "rb" ) ))

## the following dates array is used later for constructing time series plots

my_dates = sorted(list(map(lambda date: pd.to_datetime(date, format='%Y%m%d'), frames.keys())))

def wins(x,a,b):
    return(np.where(x <= a,a, np.where(x >= b, b, x)))

def clean_nas(df): 
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for numeric_column in numeric_columns: 
        df[numeric_column] = np.nan_to_num(df[numeric_column])
    
    return df

def density_plot(data, title): 
    density = gaussian_kde(data)
    xs = np.linspace(np.min(data),np.max(data),200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(xs,density(xs))
    plt.title(title)
    plt.show()
    
test = frames['20040102']
density_plot(test['Ret'], 'Daily return pre-winsorization')
density_plot(wins(test['Ret'],-0.2,0.2), 'Daily return winsorized')

D = (test['SpecRisk'] / (100 * math.sqrt(252))) ** 2
density_plot(np.sqrt(D), 'SpecRisk')


industry_factors = ['AERODEF', 'AIRLINES', 'ALUMSTEL', 'APPAREL', 'AUTO',
       'BANKS','BEVTOB', 'BIOLIFE', 'BLDGPROD','CHEM', 'CNSTENG', 'CNSTMACH', 'CNSTMATL', 'COMMEQP', 'COMPELEC',
       'COMSVCS', 'CONGLOM', 'CONTAINR', 'DISTRIB',
       'DIVFIN', 'ELECEQP', 'ELECUTIL', 'FOODPROD', 'FOODRET', 'GASUTIL',
       'HLTHEQP', 'HLTHSVCS', 'HOMEBLDG', 'HOUSEDUR','INDMACH', 'INSURNCE', 'INTERNET', 
        'LEISPROD', 'LEISSVCS', 'LIFEINS', 'MEDIA', 'MGDHLTH','MULTUTIL',
       'OILGSCON', 'OILGSDRL', 'OILGSEQP', 'OILGSEXP', 'PAPER', 'PHARMA',
       'PRECMTLS','PSNLPROD','REALEST',
       'RESTAUR', 'ROADRAIL','SEMICOND', 'SEMIEQP','SOFTWARE', 'SPLTYRET', 'SPTYCHEM', 'SPTYSTOR',
       'TELECOM', 'TRADECO', 'TRANSPRT', 'WIRELESS']

style_factors = ['BETA', 'SIZE', 'MOMENTUM', 'VALUE', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 
                 'DIVYILD', 'LTREVRSL', 'EARNQLTY']

## an R-style formula which can be used to construct a cross sectional regression
def get_formula(alphas, Y):
    L = ["0"]
    L.extend(alphas)
    L.extend(style_factors)
    L.extend(industry_factors)
    return Y + " ~ " + " + ".join(L)

## The term 'estu' is short for estimation universe
def get_estu(df): 
    estu = df.loc[df.IssuerMarketCap > 1e9].copy(deep=True)
    return estu

def estimate_factor_returns(df, alphas): 
    ## build universe based on filters 
    estu = get_estu(df)
    
    ## winsorize returns for fitting 
    estu['Ret'] = wins(estu['Ret'], -0.25, 0.25)
    
    model = ols(get_formula(alphas, "Ret"), data=estu)
    return(model.fit())

alpha_factors = ['STREVRSL', 'MGMTQLTY', 'SENTMT', 'EARNYILD', 'SEASON', 'INDMOM']

facret = {}

for date in frames:
    facret[date] = estimate_factor_returns(frames[date], list(alpha_factors)).params
    
facret_df = pd.DataFrame(index = my_dates)

for dt in my_dates: 
    for alp in alpha_factors: 
        facret_df.at[dt, alp] = facret[dt.strftime('%Y%m%d')][alp]

facret_df.cumsum().plot()



### Problem 2. 
'''
Referring to formula (4.3) for the Markowitz portfolio, and the covariance matrix in (4.29), code up a function to compute the Markowitz portfolio for each date in our sample. Refer to the helpful code above to get the different pieces in (4.29). For the risk-aversion constant, use $\kappa$ = 1e-5. Restrict yourself to the estimation universe that was used above in get_estu. Use your composite alpha factor from Problem 1 as the substitute for $ \mathbb{E}[\mathbf{r}]$. It is recommended to use the fast inversion formula (4.32) to speed things up. Compute the dot product of your portfolio with the return, that is compute $\mathbf{h} \cdot \mathbf{r}$ for each date in the sample, and plot the cumulative sum of the results. For $\mathbf{r}$, use the column called "Ret" in the same data frame that was used to compute the portfolio itself. 
'''
### Extra credit
'''
Plot other interesting metrics which help understand the portfolios in problem 2. For example, plot their long/short/net in dollars, number of holdings, factor model's predicted volatility of the portfolio, percent of variance from idio, style, industry. 
'''
### Teamwork
'''
Note that problem 2 only needs the output from Problem 1, but it does not depend on the details of how Problem 1 is being solved. So a team could split up along those lines. Part of the team can start on Problem 2, just assuming the output from problem 1 is, say, a single one of the alpha factors, until the part of the team doing problem 1 has something more interesting. 
'''

def colnames(X) -> list:
    '''
    Returns the column names of a pandas dataframe or a patsy design matrix
    
    '''
    if(type(X) == patsy.design_info.DesignMatrix): 
        return(X.design_info.column_names)
    if(type(X) == pd.core.frame.DataFrame): 
        return(X.columns.tolist())
    return(None)

def diagonal_factor_cov(date: str, X: pd.DataFrame) -> np.array:
    '''
    Returns a diagonal matrix of factor covariance for a given date and a given design matrix

    Parameters
    ----------
    date : str
        date string
    X : pd.DataFrame
        risk exposure

    Returns
    -------
    diagonal factor covariance

    '''
    cv = covariance[date]
    k = np.shape(X)[1]
    Fm = np.zeros([k, k])
    for j in range(0, k): 
        fac = colnames(X)[j]
        Fm[j,j] = (0.01**2) * cv.loc[(cv.Factor1==fac) & (cv.Factor2==fac), "VarCovar"].iloc[0]
    return(Fm)

def risk_exposures(estu: pd.DataFrame): 
    '''

    Parameters
    ----------
    estu : pd.DataFrame
        daily data

    Returns
    -------
    patsy.dmatrix
        risk exposure

    '''
    L = ["0"]
    L.extend(style_factors)
    L.extend(industry_factors)
    my_formula = " + ".join(L)
    return patsy.dmatrix(my_formula, data=estu)


my_date = '20040102'

# estu = estimation universe
estu = get_estu(frames[my_date])
n = estu.shape[0]

estu['Ret'] = wins(estu['Ret'], -0.25, 0.25)

rske = risk_exposures(estu)
F = diagonal_factor_cov(my_date, rske)

X = np.asarray(rske)
D = np.asarray((estu['SpecRisk'] / (100 * math.sqrt(252))) ** 2)

kappa = 1e-5

print(F)

### Problem 1. 
'''
Consider the list of potential alpha factors whose factor returns were plotted above. Consider ways of combining all of the alpha factors into a single, composite alpha factor. The goal is to produce a composite alpha factor which does the best job of predicting the dependent variable "Ret" on out-of-sample data. Restrict your training set on each day to the estimation universe that was used above in get_estu. Take the period 2008-2010 as the ultimate test set which you will hold in a "vault" until you are ready to do a final evaluation of your composite alpha factor. Use the period 2003-2007 for training/validation. Use cross-validation to select a model from the full family of models you are considering. It could be as simple as lasso, but we encourage you to be creative and try non-linear combinations of factors as well. Use methods from the course. 
'''

from sklearn.linear_model import Lasso, Ridge
from sklearn import svm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import random
import scipy.stats as stats
from bayes_opt import BayesianOptimization

train_dates = random.sample(sorted(frames), len(frames.keys()))
train_data = pd.concat([frames[date] for date in train_dates])[alpha_factors + ['Ret']]

# Create interaction features
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_train_poly = poly.fit_transform(train_data.iloc[:, :-1])

# Function to optimize Lasso
def optimize_lasso(alpha):
    model = Lasso(alpha=alpha)
    return cross_val_score(model, X_train_poly, train_data['Ret'], cv=5).mean()

# Function to optimize Ridge
def optimize_ridge(alpha):
    model = Ridge(alpha=alpha)
    return cross_val_score(model, X_train_poly, train_data['Ret'], cv=5).mean()

# Function to optimize SVR
def optimize_svr(C):
    model = svm.SVR(kernel='rbf', C=C, gamma='scale')
    return cross_val_score(model, X_train_poly, train_data['Ret'], cv=5).mean()

# Initialize Bayesian Optimization
optimizer_lasso = BayesianOptimization(optimize_lasso, {'alpha': (0.01, 1)}, random_state=0)
optimizer_ridge = BayesianOptimization(optimize_ridge, {'alpha': (0.01, 1)}, random_state=0)
optimizer_svr = BayesianOptimization(optimize_svr, {'C': (0.1, 10)}, random_state=0)

# Perform optimization
optimizer_lasso.maximize(n_iter=10, init_points=2)
optimizer_ridge.maximize(n_iter=10, init_points=2)
# optimizer_svr.maximize(n_iter=10, init_points=2)

# Extract best parameters
best_params_lasso = optimizer_lasso.max['params']
best_params_ridge = optimizer_ridge.max['params']
# best_params_svr = optimizer_svr.max['params']

print(f'Best parameters for Lasso: {best_params_lasso}')
print(f'Best parameters for Ridge: {best_params_ridge}')
# print(f'Best parameters for SVR: {best_params_svr}')

'''===================== Out Sample testing ===================''' 
test_frames = {}
for year in [2008,2009,2010]:
    fil = model_dir + "pandas-frames." + str(year) + ".pickle.bz2"
    test_frames.update(pickle.load( bz2.open( fil, "rb" ) ))

test_dates = random.sample(sorted(test_frames), len(test_frames.keys()))
test_data = pd.concat([test_frames[date] for date in test_dates])[alpha_factors + ["Ret"]]

# Create interaction features for the test data
X_test_poly = poly.transform(test_data.iloc[:, :-1])
y_test = test_data['Ret']

# Train Lasso with the best parameters
model_lasso = Lasso(alpha=best_params_lasso['alpha'])
model_lasso.fit(X_train_poly, train_data['Ret'])
# Predict on the test set and calculate MSE
predictions_lasso = model_lasso.predict(X_test_poly)
mse_lasso = mean_squared_error(y_test, predictions_lasso)
print(f'MSE for Lasso: {mse_lasso}')

# Train Ridge with the best parameters
model_ridge = Ridge(alpha=best_params_ridge['alpha'])
model_ridge.fit(X_train_poly, train_data['Ret'])
# Predict on the test set and calculate MSE
predictions_ridge = model_ridge.predict(X_test_poly)
mse_ridge = mean_squared_error(y_test, predictions_ridge)
print(f'MSE for Ridge: {mse_ridge}')

### Problem 2. 
'''
Referring to formula (4.3) for the Markowitz portfolio, and the covariance matrix in (4.29), code up a function to compute the Markowitz portfolio for each date in our sample. Refer to the helpful code above to get the different pieces in (4.29). For the risk-aversion constant, use $\kappa$ = 1e-5. Restrict yourself to the estimation universe that was used above in get_estu. Use your composite alpha factor from Problem 1 as the substitute for $ \mathbb{E}[\mathbf{r}]$. It is recommended to use the fast inversion formula (4.32) to speed things up. Compute the dot product of your portfolio with the return, that is compute $\mathbf{h} \cdot \mathbf{r}$ for each date in the sample, and plot the cumulative sum of the results. For $\mathbf{r}$, use the column called "Ret" in the same data frame that was used to compute the portfolio itself. 
'''
from scipy.linalg import inv
import cvxpy as cp
import numpy as np

def markowitz_portfolio(date: str, Er, frames, kappa: float = 1e-5, analytical : bool = False) -> list:
    """
    Using markowitz portfolio optimization technique to find the best weights

    Parameters
    ----------
    date : TYPE
        date string 
    frames : TYPE
        data
    kappa : float, optional
        risk aversion parameter. The default is 1e-5.
    analytical : TYPE, optional
        whether to use analytical solution. The default is False.

    Returns
    -------
    list
        factor weights (or stock weights)

    """
    # Prepare data
    estu = get_estu(frames[date])
    n = estu.shape[0]
    estu['Ret'] = wins(estu['Ret'], -0.25, 0.25)
    rske = risk_exposures(estu)
    
    F = diagonal_factor_cov(date, rske)
    X = np.asarray(rske)
    D = np.asarray( (estu['SpecRisk'] / (100 * math.sqrt(252))) ** 2 )
    
    if analytical:
        cov_inv =inv(D) - inv(D) @ X @ inv(inv(F)+ X.T @ inv(D) @ X)@ x.T @ inv(D)
        h = (1 / kappa) * (cov_inv @ Er)
    return (h @ estu['Ret'])

    # Portfolio optimization
    w = cp.Variable(n)
    portfolio_variance = cp.quad_form(w, X @ F @ X.T + np.diag(D)) # which indicates the covarance matrix
    objective = cp.Minimize(portfolio_variance - kappa * Er)
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    # Return optimal weights
    return w.value

from tqdm import tqdm
portfolios = {}
for date in tqdm(frames.keys()):
    portfolios[date] = markowitz_portfolio(date, frames)


### Extra credit
'''
Plot other interesting metrics which help understand the portfolios in problem 2. For example, plot their long/short/net in dollars, number of holdings, factor model's predicted volatility of the portfolio, percent of variance from idio, style, industry. 
'''

import riskfolio as rp
Y = facret_df

# Creating the Portfolio Object
port = rp.Portfolio(returns=Y)

# To display dataframes values in percentage format
pd.options.display.float_format = '{:.4%}'.format

# Choose the risk measure
rm = 'MV'  # Standard Deviation

# Estimate inputs of the model (historical estimates)
method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate the portfolio that maximizes the risk adjusted return ratio
w = port.optimization(model='Classic', rm=rm, obj='Sharpe', rf=0.0, l=0, hist=True)

rp.excel_report(Y,
                w,
                rf=0,
                alpha=0.05,
                t_factor=252,
                ini_days=1,
                days_per_year=252,
                name="report")

ax = rp.jupyter_report(Y,
                       w,
                       rm='MV',
                       rf=0,
                       alpha=0.05,
                       height=6,
                       width=14,
                       others=0.05,
                       nrow=25)