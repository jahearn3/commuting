import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 
import pandas as pd 
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error as MSE
from sklearn.inspection import permutation_importance 
from sklearn import ensemble
import time
import os
import re 
import textwrap 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import model 
import data_processing as dp 

def plot_residuals(plots_folder, start, end, df):
    fig = plt.figure()
    ax = sns.residplot(data=df, x=start + '_departure_time_hr', y='minutes_to_' + end, lowess=True)
    ax = time_xticks(ax, df[start + '_departure_time_hr'].min(), df[start + '_departure_time_hr'].max())
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.savefig(f'{plots_folder}/duration_vs_departure_residuals_from_{start}_to_{end}.png')
    plt.clf()

def plot_gbr_training_deviance(plots_folder, start, end, params, reg, X_test, y_test):
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
    plt.savefig(f'{plots_folder}/duration_vs_departure_training_deviance_from_{start}_to_{end}.png')
    plt.clf()

def plot_feature_importance(plots_folder, start, end, reg, X_test, y_test, df):
    feature_importance = reg.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    #plt.yticks(pos, np.array(df.feature_names)[sorted_idx])
    #plt.yticks(pos, [start + '_departure_time_hr', 'day_of_week_Mon', 'day_of_week_Tue', 'day_of_week_Wed', 'day_of_week_Thu'][sorted_idx])
    plt.title("Feature Importance (MDI)")

    result = permutation_importance(reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        #labels=np.array(df.feature_names)[sorted_idx],
        )
    plt.title("Permutation Importance (test set)")
    plt.savefig(f'{plots_folder}/duration_vs_departure_feature_importance_from_{start}_to_{end}.png')
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

def duration_vs_departure(filename, df, start='home', end='work', gbr=False, dtr=False, rfr=False, nn=False, xgb=False, ensemble_r=False, comments=False):
    fig = plt.figure()
    # Apply the default theme
    sns.set_theme()

    # Initialize the visualization
    #ax = sns.regplot(data=df, x=start + '_departure_time_hr', y='minutes_to_' + end, scatter=False, ci=None, color='r', order=order)#, scatter_kws={'s':1}) # to get the linear trendline
    ax = sns.scatterplot(data=df, x=start + '_departure_time_hr', y='minutes_to_' + end, hue='day_of_week', hue_order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], s=1)
    
    # Highlight the most recent trip by putting a yellow halo around it
    df_subset = df[[start + '_departure_time_hr', 'minutes_to_' + end]]
    df_subset = df_subset.dropna()
    x_latest = df_subset[start + '_departure_time_hr'][df_subset.index[-1]]
    y_latest = df_subset['minutes_to_' + end][df_subset.index[-1]]
    # x_latest = df[start + '_departure_time_hr'][df.index[-1]]
    # y_latest = df['minutes_to_' + end][df.index[-1]]
    ax.scatter(x_latest, y_latest, c='#FFFF14', s=100)
    
    # Add comments near points
    if(comments):
        for i, row in df.iterrows():
            try:
                chars = len(str(row['comments_from_' + start + '_to_' + end]))
                if(chars > 5):
                    wrapped_text = textwrap.fill(str(row['comments_from_' + start + '_to_' + end]), 25)
                    ax.text(row[start + '_departure_time_hr'], row['minutes_to_' + end], wrapped_text, fontsize=6)
            except KeyError:
                print('Key Error. Skipping the labeling of points.')

    # Add size of scatter points by mileage, but don't add mileage to the legend
    ax = sns.scatterplot(data=df, x=start + '_departure_time_hr', y='minutes_to_' + end, hue='day_of_week', hue_order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], size='mileage_to_' + end, legend=False) # to plot the scatter points colored by day of the week
    #ax = sns.lmplot(data=df, x='home_departure_time_hr', y='minutes_to_work', ci=None) # This has not worked due to FacetGrid issues
    
    ax = time_xticks(ax, df[start + '_departure_time_hr'].min(), df[start + '_departure_time_hr'].max())
    
    #plt.axhline(y=65, linestyle='dotted', color='g') # this line represents the amount of time it should take without traffic

    # Plot nonlinear prediction line
    #ax = nonlinear_prediction_line(ax, df)
    
    # Plot linear prediction line
    #ax = linear_prediction_line(ax, df) # This produces a line in the plot but does not look right

    # y_complete = np.array(df_notna['minutes_to_' + end])

    # Practice with statsmodels
    x, y = model.linear_prediction_from_statsmodels(df, start, end)
    ax.plot(x, y, c='b', label='Linear', linestyle='dotted')

    # Split data into training and test sets
    if(gbr or dtr or rfr or nn or xgb):
        X_train, X_test, y_train, y_test = dp.preprocess_data(start, end, df)
        print(f'shape of X_train: {X_train.shape}')
        print(f'shape of y_train: {y_train.shape}')
        print(f'shape of X_test: {X_test.shape}')
        print(f'shape of y_test: {y_test.shape}')

    # Practice with gradient boosting regression
    if(gbr):
        start_time = time.time()
        # gbreg, mse, params = model.fit_gbr(X_train, X_test, y_train, y_test)
        gbreg, mse, params = model.fit_gbr_with_grid_search(X_train, X_test, y_train, y_test)
        print(params)
        x, y = model.prediction(gbreg, df, start)
        print("GBR --- %s seconds ---" % (time.time() - start_time))
        ax.plot(x, y, c='c', label='Gradient Boosting')

    # Practice with decision tree regression
    if(dtr):
        start_time = time.time()
        # dtreg, mse = model.fit_dtr(X_train, X_test, y_train, y_test)
        dtreg, mse, params = model.fit_dtr_with_grid_search(X_train, X_test, y_train, y_test)
        print(params)
        x, y = model.prediction(dtreg, df, start)
        print("DTR --- %s seconds ---" % (time.time() - start_time))
        ax.plot(x, y, c='g', label='Decision Tree')

    if(rfr):
        start_time = time.time()
        # rfreg, mse = model.fit_rfr(X_train, X_test, y_train, y_test)
        rfreg, mse, params = model.fit_rfr_with_grid_search(X_train, X_test, y_train, y_test)
        print(params)
        x, y = model.prediction(rfreg, df, start)
        print("RFR --- %s seconds ---" % (time.time() - start_time))
        ax.plot(x, y, c='m', label='Random Forest')
    
    if(nn):
        start_time = time.time()
        nnreg, mse = model.fit_nn(X_train, X_test, y_train, y_test)
        x, y = model.prediction(nnreg, df, start)
        print("NN --- %s seconds ---" % (time.time() - start_time))
        ax.plot(x, y, c='orange', label='Neural Network')

    if(xgb):
        start_time = time.time()
        # xgbreg, mse = model.fit_xgbr(X_train, X_test, y_train, y_test)
        xgbreg, mse, params = model.fit_xgbr_with_grid_search(X_train, X_test, y_train, y_test)
        print(params)
        x, y = model.prediction(xgbreg, df, start)
        print("XGB --- %s seconds ---" % (time.time() - start_time))
        ax.plot(x, y, c='k', label='XGBoost')

    if(ensemble_r):
        start_time = time.time()
        estimators = []
        if(gbr):
            estimators.append(('gb', gbreg))
        if(dtr):
            estimators.append(('dt', dtreg))
        if(rfr):
            estimators.append(('rf', rfreg))
        if(nn):
            estimators.append(('nn', nnreg))
        if(xgb):
            estimators.append(('xg', xgbreg))
        ensmbl, mse = model.fit_ensemble(X_train, X_test, y_train, y_test, estimators)
        x, y = model.prediction(ensmbl, df, start)
        print("Ensemble --- %s seconds ---" % (time.time() - start_time))
        ax.plot(x, y, c='chartreuse', label='Ensemble')

    if(gbr or dtr or rfr or nn or xgb):
        # ax.legend(loc=1, fontsize=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Specfiy axis labels
    ax.set(xlabel=start.capitalize() + ' Departure Time',
       ylabel='Minutes to ' + end.capitalize(),
       title='Commuting Time')
    
    # Make directory for plots if it doesn't already exist
    pattern = r'_(.*)\.csv'
    match = re.search(pattern, filename)
    if(match):
        dataset = match.group(1)
    else:
        dataset = 'tenino'
        print('Filename or dataset not recognized.')
    plots_folder = f'plots_{dataset}' 
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    # Save the plot 
    plt.savefig(f'{plots_folder}/duration_vs_departure_from_{start}_to_{end}.png', bbox_inches='tight')
    plt.clf()

    # Plot the residuals 
    plot_residuals(plots_folder, start, end, df)
    
    # Plot the gradient boosting regression training deviance
    # if(gbr):
    #     plot_gbr_training_deviance(plots_folder, start, end, params, gbreg, X_test, y_test)
    
    # if(gbr):
    #     plot_feature_importance(plots_folder, start, end, gbreg, X_test, y_test, df)
    
    minutes_violin(plots_folder, start, end, df)
    
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

# violinplot of minutes to work
def minutes_violin(plots_folder, start, end, df):
    """Violin plots for minutes and departure time, both as a whole and by day of week"""
    ax = sns.violinplot(data=df, x='minutes_to_' + end)
    plt.savefig(f'{plots_folder}/minutes_from_{start}_to_{end}_violinplot')
    plt.clf()

    ax = sns.violinplot(data=df, x='minutes_to_' + end, y='day_of_week', order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])
    plt.savefig(f'{plots_folder}/minutes_from_{start}_to_{end}_by_day_violinplot')
    plt.clf()

    ax = sns.violinplot(data=df, x=start + '_departure_time_hr')
    plt.savefig(f'{plots_folder}/departure_time_from_{start}_to_{end}_violinplot')
    plt.clf()

    ax = sns.violinplot(data=df, x=start + '_departure_time_hr', y='day_of_week', order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])
    plt.savefig(f'{plots_folder}/departure_time_from_{start}_to_{end}_by_day_violinplot')
    plt.clf()

def driving_and_waiting_vs_departure(filename, df, start='home', launch_port='southworth', land_port='fauntleroy', end='work', gbr=False, dtr=False, rfr=False, nn=False, xgb=False, ensemble_r=False):
    fig = plt.figure()
    # Apply the default theme
    sns.set_theme()

    if(end == 'home'):
        launch_port = 'fauntleroy'
        land_port = 'southworth'

    # print(df[[start + '_departure_time', 'park_in_line_' + launch_port]].head())

    first_leg = (df['park_in_line_' + launch_port] - df[start + '_departure_time']).dt.total_seconds()/60
    second_leg = (df[end + '_arrival_time'] - df[land_port + '_ferry_departure_time']).dt.total_seconds()/60
    df['driving_time'] = first_leg + second_leg
    df['waiting_time'] = df['minutes_to_' + end] - df['driving_time']
    df['driving_to_waiting_ratio'] = df['driving_time'] / df['waiting_time']

    # print(df[[start + '_departure_time_hr', 'driving_time', 'waiting_time', 'driving_to_waiting_ratio']].head())

    # Initialize the visualization
    #ax = sns.regplot(data=df, x=start + '_departure_time_hr', y='minutes_to_' + end, scatter=False, ci=None, color='r', order=order)#, scatter_kws={'s':1}) # to get the linear trendline
    ax = sns.scatterplot(data=df, x=start + '_departure_time_hr', y='driving_time', hue='day_of_week', hue_order=['Mon', 'Tue', 'Thu'], s=1)
    
    # Highlight the most recent trip by putting a yellow halo around it
    x_latest = df[start + '_departure_time_hr'][df.index[-1]]
    y_latest = df['driving_time'][df.index[-1]]
    ax.scatter(x_latest, y_latest, c='#FFFF14', s=100)
    
    # Add size of scatter points by mileage, but don't add mileage to the legend
    ax = sns.scatterplot(data=df, x=start + '_departure_time_hr', y='driving_time', hue='day_of_week', hue_order=['Mon', 'Tue', 'Thu'], size='mileage_to_' + end, legend=False) # to plot the scatter points colored by day of the week
    
    ax = time_xticks(ax, df[start + '_departure_time_hr'].min(), df[start + '_departure_time_hr'].max())

    # ax.legend(fontsize=8)
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Specfiy axis labels
    ax.set(xlabel=start.capitalize() + ' Departure Time',
       ylabel='Minutes Driving to ' + end.capitalize(),
       title='Ferry Route Driving Time')
    
    # Make directory for plots if it doesn't already exist
    pattern = r'_(.*)\.csv'
    match = re.search(pattern, filename)
    if(match):
        dataset = match.group(1)
    else:
        dataset = 'tenino'
        print('Filename or dataset not recognized.')
    plots_folder = f'plots_{dataset}' 
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    # Save the plot 
    plt.savefig(f'{plots_folder}/driving_time_vs_departure_from_{start}_to_{end}.png', bbox_inches='tight')
    plt.clf()

    # Waiting time plot
    ax = sns.scatterplot(data=df, x=start + '_departure_time_hr', y='waiting_time', hue='day_of_week', hue_order=['Mon', 'Tue', 'Thu'], s=1)
    y_latest = df['waiting_time'][df.index[-1]]
    ax.scatter(x_latest, y_latest, c='#FFFF14', s=100)
    ax = sns.scatterplot(data=df, x=start + '_departure_time_hr', y='waiting_time', hue='day_of_week', hue_order=['Mon', 'Tue', 'Thu'], size='mileage_to_' + end, legend=False) # to plot the scatter points colored by day of the week
    ax = time_xticks(ax, df[start + '_departure_time_hr'].min(), df[start + '_departure_time_hr'].max())
    ax.legend(loc=0, fontsize=10)
    ax.set(xlabel=start.capitalize() + ' Departure Time',
       ylabel='Minutes Waiting on route to ' + end.capitalize(),
       title='Ferry Route Waiting Time')
    plt.savefig(f'{plots_folder}/waiting_time_vs_departure_from_{start}_to_{end}.png')
    plt.clf()

    # Ratio plot
    # Convert dates to a new format - MM-DD
    # df[start + '_departure_time'] = [d.strftime('%m-%d') for d in df[start + '_departure_time']]
    #TODO: x axis points are not spaced correctly
    ax = sns.scatterplot(data=df, x=start + '_departure_time', y='driving_to_waiting_ratio', hue='day_of_week', hue_order=['Mon', 'Tue', 'Thu'], s=1)
    ax = sns.scatterplot(data=df, x=start + '_departure_time', y='driving_to_waiting_ratio', hue='day_of_week', hue_order=['Mon', 'Tue', 'Thu'], size='mileage_to_' + end, legend=False) 
    
    # Rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # Add text to make understanding the ratio more intuitive
    earliest = df[start + '_departure_time'].min()
    latest = df[start + '_departure_time'].max()

    median_date = earliest + ((latest - earliest) / 3.14)
    ax.text(median_date, df['driving_to_waiting_ratio'].max() - 0.1, 'More Driving', c='b')
    ax.text(median_date, df['driving_to_waiting_ratio'].min() + 0.1, 'More Waiting', c='b')

    ax.legend(loc=0, fontsize=10)
    ax.set(xlabel=start.capitalize() + ' Departure Time and Date',
       ylabel='Driving to Waiting Ratio on route to ' + end.capitalize(),
       title='Ferry Route Driving to Waiting Ratio')
    plt.xticks(rotation=35) 
    plt.savefig(f'{plots_folder}/driving_waiting_ratio_vs_departure_from_{start}_to_{end}.png')
    plt.clf()

# TODO: Calculate time at which to leave to arrive by 9am
# TODO: Make a plot to show seasonality