import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 
import pandas as pd 
from statsmodels.formula.api import ols
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.inspection import permutation_importance 
from sklearn import ensemble

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import model 

def plot_residuals(start, end, df):
    fig = plt.figure()
    ax = sns.residplot(data=df, x=start + '_departure_time_hr', y='minutes_to_' + end, lowess=True)
    ax = time_xticks(ax, df[start + '_departure_time_hr'].min(), df[start + '_departure_time_hr'].max())
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.savefig('duration_vs_departure_residuals_from_' + start + '_to_' + end + '.png')
    plt.clf()

def preprocess_data(start, end, df):
    #print(df)
    df = pd.get_dummies(data=df, columns=['day_of_week'], drop_first=True)
    df_notna = df[df[start + '_departure_time_hr'].notna()]
    X = df_notna[[start + '_departure_time_hr','day_of_week_Mon','day_of_week_Tue','day_of_week_Wed','day_of_week_Thu']]
    #df = df.get_dummies(data=df, prefix=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], columns='day_of_week', drop_first=True)
    #X = np.array(df[start + '_departure_time_hr'].dropna()).reshape(-1, 1)
    #X = np.array(df_notna[start + '_departure_time_hr']).reshape(-1, 1)
    #X = np.array([df_notna[start + '_departure_time_hr'], df_notna['day_of_week_Mon'], df_notna['day_of_week_Tue'], df_notna['day_of_week_Wed'], df_notna['day_of_week_Thu']])
    #print(X.shape)
    #X = np.array([df[start + '_departure_time_hr'], df['day_of_week_Mon'], df['day_of_week_Tue'], df['day_of_week_Wed'], df['day_of_week_Thu']])
    #y = np.array(df['minutes_to_' + end].dropna())
    y = np.array(df_notna['minutes_to_' + end])
    #(len1, len2) = X.shape
    #X = np.reshape(X, (len2, len1))
    #print(y)
    #X = df_notna.drop(['minutes_to_' + end], axis=1)
    #print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "min_samples_split": 4,
        "learning_rate": 0.01,
        "loss": "squared_error"#,
    }
    return X_train, X_test, y_train, y_test, params 

    # hyperparameters -> minimum deviance
    #  500, 4,  5, 0.01 -> 200 @ 500
    # 1000, 4,  5, 0.01 -> 200 @ 1000
    #  500, 8,  5, 0.01 -> 250 @ 500
    #  500, 4, 10, 0.01 -> 280 @ 500
    # 1000, 4, 10, 0.01 -> 200 @ 1000
    #  500, 3,  4, 0.01 -> 200 @ 500 <- selected
    #  500, 2,  3, 0.01 -> 270 @ 500

def plot_gbr_training_deviance(start, end, params, reg, X_test, y_test):
    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        test_score[i] = reg.loss_(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(np.arange(params["n_estimators"]) + 1, reg.train_score_, "b-", label="Training Set Deviance")
    plt.plot(np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance")
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    plt.savefig('duration_vs_departure_training_deviance_from_' + start + '_to_' + end + '.png')
    plt.clf()

def plot_feature_importance(start, end, reg, X_test, y_test, df):
    feature_importance = reg.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    #plt.yticks(pos, np.array(df.feature_names)[sorted_idx])
    #plt.yticks(pos, [start + '_departure_time_hr', 'day_of_week_Mon', 'day_of_week_Tue', 'day_of_week_Wed', 'day_of_week_Thu'][sorted_idx])
    plt.title("Feature Importance (MDI)")

    result = permutation_importance(
        reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        #labels=np.array(df.feature_names)[sorted_idx],
    )
    plt.title("Permutation Importance (test set)")
    plt.savefig('duration_vs_departure_feature_importance_from_' + start + '_to_' + end + '.png')
    plt.clf()

def time_xticks(ax, earliest_departure, latest_departure):
    # Change x-axis ticks from decimal form to HH:MM form
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6, steps=[1, 5, 10]))
    ticks_loc = ax.get_xticks().tolist()
    # print(ticks_loc)
    # ticks_interval = ticks_loc[1] - ticks_loc[0]
    # print(ticks_interval)
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
    # final_ticklabels = [("%d:%02d" % (int(x), int((x*60) % 60))).format(x) for x in ticks_loc]
    # print(final_ticklabels)
    ax.set_xticklabels([("%d:%02d" % (int(x), int((x*60) % 60))).format(x) for x in ticks_loc])
    # ax.set_xticks()

    # Ensure axis bounds are good (GradientBoostingRegressor fit lines include some weird vertical lines way off to the left)
    ax.set_xlim(earliest_departure - 0.1, latest_departure + 0.1)
    #ax.set_xlim(np.fmin(earliest_departure, ticks_loc[0]) - 0.1, np.fmax(latest_departure, ticks_loc[-1]) + 0.1)
    return ax

def duration_vs_departure(df, start='home', end='work', order=1, gbr=False):
    fig = plt.figure()
    # Apply the default theme
    sns.set_theme()

    # Create a visualization
    #ax = sns.regplot(data=df, x=start + '_departure_time_hr', y='minutes_to_' + end, scatter=False, ci=None, color='r', order=order)#, scatter_kws={'s':1}) # to get the linear trendline
    ax = sns.scatterplot(data=df, x=start + '_departure_time_hr', y='minutes_to_' + end, hue='day_of_week', hue_order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], size='mileage_to_' + end, legend=False) # to plot the scatter points colored by day of the week
    #ax = sns.lmplot(data=df, x='home_departure_time_hr', y='minutes_to_work', ci=None) # This has not worked due to FacetGrid issues
    
    ax = time_xticks(ax, df[start + '_departure_time_hr'].min(), df[start + '_departure_time_hr'].max())
    
    #plt.axhline(y=65, linestyle='dotted', color='g') # this line represents the amount of time it should take without traffic

    # Plot nonlinear prediction line
    #ax = nonlinear_prediction_line(ax, df)
    
    # Plot linear prediction line
    #ax = linear_prediction_line(ax, df) # This produces a line in the plot but does not look right
    
    # Plot decision tree regressor - not yet working
    # Last error: ValueError: Number of labels=13 does not match number of samples=1
    # ax = decision_tree_regressor(ax, start, end, df)

    # Practice with statsmodels
    x, y = model.linear_prediction_from_statsmodels(df, start, end)
    ax.plot(x, y, c='b')

    # Practice with gradient boosting regression
    if(gbr):
        X_train, X_test, y_train, y_test, params = preprocess_data(start, end, df)
        reg, mse = model.fit_gbr(X_train, X_test, y_train, y_test, params)
        x, y = model.prediction_from_gbr(reg, df, start)
        ax.plot(x, y, c='c')

    # Specfiy axis labels
    ax.set(xlabel=start.capitalize() + ' Departure Time',
       ylabel='Minutes to ' + end.capitalize(),
       title='Commuting Time')

    # Save the plot 
    plt.savefig('duration_vs_departure_from_' + start + '_to_' + end + '.png')
    plt.clf()

    # Plot the residuals 
    plot_residuals(start, end, df)
    
    # Plot the gradient boosting regression training deviance
    if(gbr):
        plot_gbr_training_deviance(start, end, params, reg, X_test, y_test)
    
    if(gbr):
        plot_feature_importance(start, end, reg, X_test, y_test, df)
    
#fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
#sns.set(color_codes=True)
#sns.set_palette('colorblind')
# make histograms of commuting minutes
#sns.displot(df['minutes_to_work'], ax=ax0, kind='kde', rug=True, fill=True)
#sns.despine(top=True, right=True)
#ax0.set(xlabel='Minutes to Work', title='Commuting Durations')

#ax0.axvline(x=df['minutes_to_work'].median(), color='m', label='Median', linestyle='--', linewidth=2)
#ax0.axvline(x=df['minutes_to_work'].mean(), color='b', label='Mean', linestyle='-', linewidth=2)
#ax0.legend()
#sns.displot(df['minutes_to_home'], ax=ax1, kind='kde', rug=True, fill=True) # put this in a different row
#plt.show()
#plt.clf()

#print(df['home_departure_time'])
#print(df['home_departure_time'][5:])

# make plots of commuting minutes vs. departure time
#fig, ax = plt.subplots()
#sns.lmplot(x='home_departure_time'[5:], y='minutes_to_work', data=df, hue='day_of_week')
#ax.set(xlabel='Minutes to Work', title='Commuting Duration vs. Departure Time')
#plt.show()
#plt.clf()