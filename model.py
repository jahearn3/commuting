import numpy as np 
import pandas as pd 
from statsmodels.formula.api import ols
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
#from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error as MSE
from sklearn.inspection import permutation_importance 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, make_scorer

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

def linear_prediction_line(df, start): # runs but does not behave as expected 
    x = np.linspace(np.amin(df[start + '_departure_time_hr']), np.amax(df[start + '_departure_time_hr']), len(df[start + '_departure_time_hr']))
    m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, df['minutes_to_work'], rcond=None)[0]
    return x, m*x + c

def nonlinear_prediction_line(df, start, end):
    x = np.linspace(np.amin(df[start + '_departure_time_hr']), np.amax(df[start + '_departure_time_hr']), len(df[start + '_departure_time_hr']))
    df[start + '_departure_time_hr2'] = df[start + '_departure_time_hr']**2
    results = ols('minutes_to_' + end + ' ~ ' + start +'_departure_time_hr + ' + start + '_departure_time_hr2', data=df).fit()
    pred = results.predict(df)
    coeffs = np.polyfit(df[start + '_departure_time_hr'], df['minutes_to_' + end], 2)
    # print(coeffs)
    return x, coeffs[0] * x + coeffs[1] * x + coeffs[2]



def decision_tree_regressor(df, start, end):
    seed = 3
    X = df[start + '_departure_time_hr'].dropna()
    y = df['minutes_to_' + end].dropna() 
    #X_train = df[start + '_departure_time_hr'].dropna()
    #y_train = df['minutes_to_' + end].dropna()
    #X_test = np.linspace(X_train.min(), X_train.max(), 100)
    #TODO: k-fold cross validation 
    #kf = KFold(n_splits=4, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    dt = DecisionTreeRegressor(max_depth=8, random_state=seed)
    #X_train.reshape(-1, 1)
    #y_train.reshape(-1, 1)
    dt.fit(X_train, y_train)
    # dt_scores = cross_val_score(dt_fit, X_train, y_train, cv = 5)

    y_pred = dt.predict(X_test)
    mse_dt = MSE(y_test, y_pred)
    rmse_dt = mse_dt**(1/2)
    print('Root mean square: ' + str(rmse_dt))
    return X_test, y_pred

def linear_prediction_from_statsmodels(df, start, end, num=20):
    mdl_duration_vs_departure = ols('minutes_to_' + end + ' ~ ' + start + '_departure_time_hr', data=df).fit() # Create the model object and fit the model
    #print(mdl_duration_vs_departure.params) # Print the parameters of the fitted model
    coeffs = mdl_duration_vs_departure.params # Get the coefficients of mdl_price_vs_conv
    x = np.linspace(df[start + '_departure_time_hr'].min(), df[start + '_departure_time_hr'].max(), num)
    return x, coeffs[0] + (coeffs[1] * x) 

def compare(df, start, end):
    print('correlation: ' + str(df[start + '_departure_time_hr'].corr(df['minutes_to_' + end])))

def fit_gbr(X_train, X_test, y_train, y_test, params):
    reg = GradientBoostingRegressor(**params)
    reg_fit = reg.fit(X_train, y_train)
    mse = MSE(y_test, reg.predict(X_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

    # reg_scores = cross_val_score(reg_fit, X_train, y_train, cv = 5)
    # print("mean cross validation score: {}".format(np.mean(reg_scores)))
    # print("score without cv: {}".format(reg_fit.score(X_train, y_train)))   

    # # on the test or hold-out set
    # print(r2_score(y_test, reg_fit.predict(X_test)))
    # print(reg_fit.score(X_test, y_test)) 

    # scoring = make_scorer(r2_score)
    # g_cv = GridSearchCV(GradientBoostingRegressor(random_state=0),
    #             param_grid={'min_samples_split': range(2, 10)},
    #             scoring=scoring, cv=5, refit=True)

    # g_cv.fit(X_train, y_train)
    # g_cv.best_params_

    # result = g_cv.cv_results_
    # # print(result)
    # result = r2_score(y_test, g_cv.best_estimator_.predict(X_test))
    # print(result)

    # mse = MSE(y_test, reg.predict(X_test))
    # print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
    return reg, mse 

def prediction_from_gbr(reg, df, start):
    num = int(np.sqrt(df[start + '_departure_time_hr'].notnull().sum())) + 1
    #print('Earliest departure: ' + str(df[start + '_departure_time_hr'].min()))
    #print('Latest departure: ' + str(df[start + '_departure_time_hr'].max()))
    t = np.linspace(df[start + '_departure_time_hr'].min(), df[start + '_departure_time_hr'].max(), num) # assign time 
    # future: count day_of_week weights and weight the days according to the count 
    df = pd.get_dummies(data=df, columns=['day_of_week'], drop_first=True)
    # print(df['day_of_week_Mon'].sum())
    # print(df['day_of_week_Tue'].sum())
    # print(df['day_of_week_Wed'].sum())
    # print(df['day_of_week_Thu'].sum())
    # assign day_of_week: one line for Mondays, one line for Tuesdays, one line for Thursdays
    mondays  = np.transpose(np.array([t, np.ones(num) , np.zeros(num), np.zeros(num), np.zeros(num)]))
    tuesdays = np.transpose(np.array([t, np.zeros(num), np.ones(num) , np.zeros(num), np.zeros(num)]))
    thursdays = np.transpose(np.array([t, np.zeros(num), np.zeros(num) , np.zeros(num), np.ones(num)]))
    mon_line = reg.predict(mondays)
    tue_line = reg.predict(tuesdays)
    thu_line = reg.predict(thursdays)
    x = (mondays + tuesdays + thursdays)/3
    x_t = x.T 
    y = (mon_line + tue_line + thu_line)/3
    return x_t[0], y