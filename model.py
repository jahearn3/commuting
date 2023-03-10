import numpy as np 
import pandas as pd 
from statsmodels.formula.api import ols
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
#from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error as MSE
from sklearn.inspection import permutation_importance 
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import r2_score, make_scorer
from keras.models import Sequential
from keras.layers import Dense 
from xgboost import XGBRegressor

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

def linear_prediction_from_statsmodels(df, start, end, num=20):
    mdl_duration_vs_departure = ols('minutes_to_' + end + ' ~ ' + start + '_departure_time_hr', data=df).fit() # Create the model object and fit the model
    #print(mdl_duration_vs_departure.params) # Print the parameters of the fitted model
    coeffs = mdl_duration_vs_departure.params # Get the coefficients of mdl_price_vs_conv
    x = np.linspace(df[start + '_departure_time_hr'].min(), df[start + '_departure_time_hr'].max(), num)
    y = coeffs[0] + (coeffs[1] * x)
    return x, y

def compare(df, start, end):
    print('correlation: ' + str(df[start + '_departure_time_hr'].corr(df['minutes_to_' + end])))

def compute_mse(reg, abbreviation, X_train, X_test, y_train, y_test):
    '''Computes the mean squared error for training, test, and complete datasets'''
    X = [X_train, X_test, pd.concat([X_test, X_train])]
    Y = [y_train, y_test, np.concatenate((y_test, y_train), axis=0)]
    labels = ['training', 'test', 'complete']
    for (x, y, label) in zip(X, Y, labels):
        mse = MSE(y, reg.predict(x))
        print(f'{abbreviation} mean squared error (MSE) on {label} set: {mse}')
    return mse

def fit_dtr(X_train, X_test, y_train, y_test):
    seed = 3
    dt = DecisionTreeRegressor(max_depth=8, random_state=seed)
    dt.fit(X_train, y_train)
    mse = compute_mse(dt, 'DTR', X_train, X_test, y_train, y_test)
    return dt, mse

def fit_gbr(X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "min_samples_split": 4,
        "learning_rate": 0.01,
        "loss": "squared_error"#,
    }
    # hyperparameters -> minimum deviance
    #  500, 4,  5, 0.01 -> 200 @ 500
    # 1000, 4,  5, 0.01 -> 200 @ 1000
    #  500, 8,  5, 0.01 -> 250 @ 500
    #  500, 4, 10, 0.01 -> 280 @ 500
    # 1000, 4, 10, 0.01 -> 200 @ 1000
    #  500, 3,  4, 0.01 -> 200 @ 500 <- selected
    #  500, 2,  3, 0.01 -> 270 @ 500

    reg = GradientBoostingRegressor(**params)
    reg_fit = reg.fit(X_train, y_train)
    mse = compute_mse(reg, 'GBR', X_train, X_test, y_train, y_test)

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

    return reg, mse, params 

def fit_xgbr(X_train, X_test, y_train, y_test):
    # params = {
    #     "n_estimators": 500,
    #     "max_depth": 6,
    #     "min_samples_split": 4,
    #     "learning_rate": 0.01,
    #     "loss": "squared_error"#,
    # }
    
    # reg = XGBRegressor(**params)
    reg = XGBRegressor()
    reg.fit(X_train, y_train)
    mse = compute_mse(reg, 'XGBR', X_train, X_test, y_train, y_test)
    return reg, mse

def fit_rfr(X_train, X_test, y_train, y_test):
    # These params were determined the best through the below GridSearchCV
    params = {
        "n_estimators": 200,
        "max_depth": 3,
        "max_features": 'sqrt',
        "random_state": 18,
    }
    rf = RandomForestRegressor(**params)
    # rf = RandomForestRegressor(n_estimators = 300, max_features = 'sqrt', max_depth = 5, random_state = 18)
    rf.fit(X_train, y_train)
    mse = compute_mse(rf, 'RFR', X_train, X_test, y_train, y_test)

    ## Using GridSearchCV to find the best params: this works but takes a few minutes to run 
    ## Define Grid 
    # grid = { 
    #     'n_estimators': [200,300,400,500],
    #     'max_features': ['sqrt','log2'],
    #     'max_depth' : [3,4,5,6,7],
    #     'random_state' : [18]
    # }
    # ## show start time
    # # print(datetime.now())
    # ## Grid Search function
    # CV_rfr = GridSearchCV(estimator=RandomForestRegressor(), param_grid=grid, cv= 5)
    # CV_rfr.fit(X_train, y_train)
    # mse = MSE(y_test, CV_rfr.predict(X_test))
    # print(f'CVRFR mean squared error (MSE) on test set: {mse}')
    # ## show end time
    # # print(datetime.now())
    # print('Best params: ')
    # print(CV_rfr.best_params_)
    # Best params: 
    # {'max_depth': 3, 'max_features': 'sqrt', 'n_estimators': 200, 'random_state': 18}

    return rf, mse 

def fit_nn(X_train, X_test, y_train, y_test):
    nn = Sequential()
    nn.add(Dense(5, kernel_initializer='normal', activation='relu', input_dim=X_train.shape[1]))
    nn.add(Dense(5, kernel_initializer='normal', activation='tanh'))
    nn.add(Dense(1, kernel_initializer='normal'))
    nn.compile(loss='mean_squared_error', optimizer='adam')
    nn_fit = nn.fit(X_train, y_train, batch_size=int(X_train.shape[0]), epochs=3)
    mse = compute_mse(nn, 'NN', X_train, X_test, y_train, y_test)
    return nn, mse

def fit_ensemble(X_train, X_test, y_train, y_test, estimators):
    # This ensemble approach uses the VotingRegressor method
    reg = VotingRegressor(estimators=estimators)
    reg.fit(X_train, y_train)
    mse = compute_mse(reg, 'Ensemble', X_train, X_test, y_train, y_test)
    return reg, mse

def prediction(reg, df, start):
# def prediction_from_gbr(reg, df, start):
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
