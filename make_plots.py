import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import numpy as np 
import pandas as pd 
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error as MSE
from sklearn.inspection import permutation_importance 
from sklearn import ensemble
import math
import time
import os
import re 
import scipy.interpolate as sci
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
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(steps=[1]))
    # ticks_loc = ax.get_xticks().tolist()
    # ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
    # ax.set_xticklabels([("%d:%02d" % (int(x), int((x*60) % 60))).format(x) for x in ticks_loc])

    start_hour = math.floor(earliest_departure)
    end_hour = math.ceil(latest_departure)
    ax.set_xticks(range(start_hour, end_hour + 1, 1))  # Set ticks every hour
    ax.set_xticklabels([f"{h}:00" for h in range(start_hour, end_hour + 1)])  # Set x-tick labels to HH:MM format
    # Rotate x-tick labels if crowded
    max_ticks = 10  # Set a threshold for the number of ticks
    ticks_loc = ax.get_xticks()

    if len(ticks_loc) > max_ticks:
        plt.xticks(rotation=45)
    else:
        plt.xticks(rotation=0)  # Reset rotation if not crowded


    # Ensure axis bounds are good (GradientBoostingRegressor fit lines include some weird vertical lines way off to the left)
    ax.set_xlim(earliest_departure - 0.1, latest_departure + 0.1)
    return ax


def duration_vs_departure(filename, df, start='home', end='work', gbr=False, dtr=False, rfr=False, nn=False, xgb=False, ensemble_r=False, comments=False, annotate=True, show_extra_prediction_lines=False):
    fig = plt.figure()
    # Apply the default theme
    sns.set_theme()

    # Initialize the visualization
    ax = sns.scatterplot(data=df, x=start + '_departure_time_hr', y='minutes_to_' + end, hue='day_of_week', hue_order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], s=1)

    # Highlight the most recent trip by putting a yellow halo around it
    df_subset = df[[start + '_departure_time_hr', 'minutes_to_' + end]]
    df_subset = df_subset.dropna()
    x_latest = df_subset[start + '_departure_time_hr'][df_subset.index[-1]]
    y_latest = df_subset['minutes_to_' + end][df_subset.index[-1]]
    ax.scatter(x_latest, y_latest, c='#FFFF14', s=100)

    # Print percentile of most recent trip compared to similar departure times
    df_similar_departure = df_subset[(df_subset[start + '_departure_time_hr'] > x_latest - 0.15) & (df_subset[start + '_departure_time_hr'] < x_latest + 0.15)]
    if len(df_similar_departure) > 1:
        percentile = (df_similar_departure['minutes_to_' + end] > y_latest).sum() / (len(df_similar_departure) - 1)
        x_latest_hmm = ("%d:%02d" % (int(x_latest), int((x_latest*60) % 60))).format(x_latest)
        x_similar_min = df_similar_departure[start + '_departure_time_hr'].min()
        x_similar_max = df_similar_departure[start + '_departure_time_hr'].max()
        x_similar_min_hmm = ("%d:%02d" % (int(x_similar_min), int((x_similar_min*60) % 60))).format(x_similar_min)
        x_similar_max_hmm = ("%d:%02d" % (int(x_similar_max), int((x_similar_max*60) % 60))).format(x_similar_max)
        percentile_text = f'The most recent trip departing \n{start} at {x_latest_hmm} took {int(y_latest)} minutes, \nwhich is '
        if percentile == 0:
            percentile_text += 'the slowest '
        elif percentile == 1:
            percentile_text += 'the fastest '
        else:
            percentile_text += f'faster than {percentile:.0%} '
        percentile_text += f'of {len(df_similar_departure)} \ntrips with departure times \nbetween {x_similar_min_hmm} and {x_similar_max_hmm}.'
        print(percentile_text)
        # percentile_text = f'The most recent trip departing \n{start} at {x_latest_hmm} took {int(y_latest)} minutes, \nwhich is faster than {percentile:.0%} of {len(df_similar_departure)} \ntrips with departure times \nbetween {x_similar_min_hmm} and {x_similar_max_hmm}.'
        if annotate:
            ax.text(1.05, 0.25, percentile_text, fontsize=8, transform=ax.transAxes, verticalalignment='top')

    # Add horizontal line at mean 
    mean = df['minutes_to_' + end].mean()
    ax.axhline(mean, color='c', linestyle='dotted', label='Mean')

    # Add horizontal line at 3 * sigma
    sigma = 3 * df['minutes_to_' + end].std()
    ax.axhline(mean + sigma, color='gray', linestyle='dotted', label=r'Mean + $3\sigma$')

    # Add diagonal line to indicate what time to depart to arrive on schedule
    # if end == 'work':
    #     ax.plot([7, 8, 8 + (5/60)], [120, 60, 55], color='r', linestyle='dotted', label='Target Departure Time')
    # elif end == 'home':
    #     ax.plot([16.5 - (10/60), 16.5, 17.5, 17.5 + (5/60)], [130, 120, 60, 55], color='r', linestyle='dotted', label='Target Departure Time')

    # Add comments near points
    if comments:
        for i, row in df.iterrows():
            try:
                chars = len(str(row['comments_from_' + start + '_to_' + end]))
                if chars > 5:
                    wrapped_text = textwrap.fill(str(row['comments_from_' + start + '_to_' + end]), 25)
                    ax.text(row[start + '_departure_time_hr'], row['minutes_to_' + end], wrapped_text, fontsize=6)
            except KeyError:
                print('Key Error. Skipping the labeling of points.')

    # Add size of scatter points by mileage, but don't add mileage to the legend
    ax = sns.scatterplot(data=df, x=start + '_departure_time_hr', y='minutes_to_' + end, hue='day_of_week', hue_order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], size='mileage_to_' + end, legend=False) # to plot the scatter points colored by day of the week
    ax = time_xticks(ax, df[start + '_departure_time_hr'].min(), df[start + '_departure_time_hr'].max())

    # Practice with statsmodels
    x, y = model.linear_prediction_from_statsmodels(df, start, end)
    if show_extra_prediction_lines:
        ax.plot(x, y, c='b', label='Linear', linestyle='dotted')

    # TODO: Consider filtering out outliers using the 3 sigma from linear fit line instead of the 3 sigma from the mean line
    # ax.plot(x, y, c='b', label='Linear', linestyle='dotted')
    # ax.plot(x, y + sigma, c='b', label=r'Linear + $3\sigma$', linestyle='dotted')

    # Split data into training and test sets
    if gbr or dtr or rfr or nn or xgb:
        X_train, X_test, y_train, y_test = dp.preprocess_data(start, end, df)
        print(f'shape of X_train: {X_train.shape}')
        print(f'shape of y_train: {y_train.shape}')
        print(f'shape of X_test: {X_test.shape}')
        print(f'shape of y_test: {y_test.shape}')

    if gbr:
        start_time = time.time()
        # gbreg, mse, params = model.fit_gbr(X_train, X_test, y_train, y_test)
        gbreg, mse, params = model.fit_gbr_with_grid_search(X_train, X_test, y_train, y_test)
        print(params)
        x, y = model.prediction(gbreg, df, start)
        print("GBR --- %s seconds ---" % (time.time() - start_time))
        if show_extra_prediction_lines:
            ax.plot(x, y, c='c', label='Gradient Boosting')
    if dtr:
        start_time = time.time()
        # dtreg, mse = model.fit_dtr(X_train, X_test, y_train, y_test)
        dtreg, mse, params = model.fit_dtr_with_grid_search(X_train, X_test, y_train, y_test)
        print(params)
        x, y = model.prediction(dtreg, df, start)
        print("DTR --- %s seconds ---" % (time.time() - start_time))
        if show_extra_prediction_lines:
            ax.plot(x, y, c='g', label='Decision Tree')
    if rfr:
        start_time = time.time()
        # rfreg, mse = model.fit_rfr(X_train, X_test, y_train, y_test)
        rfreg, mse, params = model.fit_rfr_with_grid_search(X_train, X_test, y_train, y_test)
        print(params)
        x, y = model.prediction(rfreg, df, start)
        print("RFR --- %s seconds ---" % (time.time() - start_time))
        if show_extra_prediction_lines:
            ax.plot(x, y, c='m', label='Random Forest')
    if nn:
        start_time = time.time()
        nnreg, mse = model.fit_nn(X_train, X_test, y_train, y_test)
        x, y = model.prediction(nnreg, df, start)
        print("NN --- %s seconds ---" % (time.time() - start_time))
        if show_extra_prediction_lines:
            ax.plot(x, y, c='orange', label='Neural Network')
    if xgb:
        start_time = time.time()
        # xgbreg, mse = model.fit_xgbr(X_train, X_test, y_train, y_test)
        xgbreg, mse, params = model.fit_xgbr_with_grid_search(X_train, X_test, y_train, y_test)
        print(params)
        x, y = model.prediction(xgbreg, df, start)
        print("XGB --- %s seconds ---" % (time.time() - start_time))
        if show_extra_prediction_lines:
            ax.plot(x, y, c='k', label='XGBoost')

    if ensemble_r:
        start_time = time.time()
        estimators = []
        if gbr:
            estimators.append(('gb', gbreg))
        if dtr:
            estimators.append(('dt', dtreg))
        if rfr:
            estimators.append(('rf', rfreg))
        if nn:
            estimators.append(('nn', nnreg))
        if xgb:
            estimators.append(('xg', xgbreg))
        ensmbl, mse = model.fit_ensemble(X_train, X_test, y_train, y_test, estimators)
        x, y = model.prediction(ensmbl, df, start)
        print("Ensemble --- %s seconds ---" % (time.time() - start_time))
        if show_extra_prediction_lines:
            ax.plot(x, y, c='chartreuse', label='Ensemble')
        else:
            # degree = 5
            # coefficients = np.polyfit(x, y, degree)
            # polynomial = np.poly1d(coefficients)
            # x_new = np.linspace(x.min() - 2, x.max() + 2, 100)
            # y_new = polynomial(x_new)
            # remove any points less than x.min() or greater than x.max()
            # x_new_cut = x_new[(x_new >= x.min()) & (x_new <= x.max())]
            # y_new_cut = y_new[(x_new >= x.min()) & (x_new <= x.max())]
            ax.plot(x, y, c='k', label='ML Prediction')
            # ax.plot(x_new_cut, y_new_cut, label='Smoothed ML Prediction')

            # add a line that averages the ML prediction and smoothed ML prediction
            # x_j = np.linspace(x.min(), x.max(), len(x))
            # y_j = sci.interp1d(x_j, x)(np.linspace(x.min(), x.max(), len(x_new_cut)))
            # avg_y = 1.82 * np.mean([y_j, y_new_cut], axis=0)
            # ax.plot(x_new_cut, avg_y, c='r', label='Half-Smoothed ML Prediction')

    if gbr or dtr or rfr or nn or xgb:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Specfiy axis labels
    ax.set(xlabel=start.capitalize() + ' Departure Time',
       ylabel='Minutes to ' + end.capitalize(),
       title='Commuting Time')

    # Make directory for plots if it doesn't already exist
    pattern = r'_(.*)\.csv'
    match = re.search(pattern, filename)
    if match:
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

    if ensemble_r:
        predictions_by_month(plots_folder, ensmbl, df, start, end)

    departure_times_over_time(plots_folder, start, end, df)
    arrival_times_over_time(plots_folder, start, end, df)


# violinplot of minutes to work
def minutes_violin(plots_folder, start, end, df):
    """Violin plots for minutes and departure time, both as a whole and by day of week"""
    sns.violinplot(data=df, x='minutes_to_' + end)
    plt.savefig(f'{plots_folder}/minutes_from_{start}_to_{end}_violinplot')
    plt.clf()
    sns.violinplot(data=df, x='minutes_to_' + end, y='day_of_week', order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])
    plt.savefig(f'{plots_folder}/minutes_from_{start}_to_{end}_by_day_violinplot')
    plt.clf()
    sns.violinplot(data=df, x=start + '_departure_time_hr')
    plt.savefig(f'{plots_folder}/departure_time_from_{start}_to_{end}_violinplot')
    plt.clf()
    sns.violinplot(data=df, x=start + '_departure_time_hr', y='day_of_week', order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])
    plt.savefig(f'{plots_folder}/departure_time_from_{start}_to_{end}_by_day_violinplot')
    plt.clf()


def driving_and_waiting_vs_departure(filename, df, start='home', launch_port='southworth', land_port='fauntleroy', end='work', gbr=False, dtr=False, rfr=False, nn=False, xgb=False, ensemble_r=False):
    fig = plt.figure()
    # Apply the default theme
    sns.set_theme()

    if end == 'home':
        launch_port = 'fauntleroy'
        land_port = 'southworth'

    first_leg = (df['park_in_line_' + launch_port] - df[start + '_departure_time']).dt.total_seconds()/60
    second_leg = (df[end + '_arrival_time'] - df[land_port + '_ferry_departure_time']).dt.total_seconds()/60
    df['driving_time'] = first_leg + second_leg
    df['waiting_time'] = df['minutes_to_' + end] - df['driving_time']
    df['driving_to_waiting_ratio'] = df['driving_time'] / df['waiting_time']

    # Initialize the visualization
    ax = sns.scatterplot(data=df, x=start + '_departure_time_hr', y='driving_time', hue='day_of_week', hue_order=['Mon', 'Tue', 'Thu'], s=1)

    # Highlight the most recent trip by putting a yellow halo around it
    x_latest = df[start + '_departure_time_hr'][df.index[-1]]
    y_latest = df['driving_time'][df.index[-1]]
    ax.scatter(x_latest, y_latest, c='#FFFF14', s=100)

    # Add size of scatter points by mileage, but don't add mileage to the legend
    ax = sns.scatterplot(data=df, x=start + '_departure_time_hr', y='driving_time', hue='day_of_week', hue_order=['Mon', 'Tue', 'Thu'], size='mileage_to_' + end, legend=False) # to plot the scatter points colored by day of the week
    ax = time_xticks(ax, df[start + '_departure_time_hr'].min(), df[start + '_departure_time_hr'].max())
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Specfiy axis labels
    ax.set(xlabel=start.capitalize() + ' Departure Time',
       ylabel='Minutes Driving to ' + end.capitalize(),
       title='Ferry Route Driving Time')

    # Make directory for plots if it doesn't already exist
    pattern = r'_(.*)\.csv'
    match = re.search(pattern, filename)
    if match:
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


def predictions_by_month(plots_folder, reg, df, start, end):
    fig = plt.figure()
    colormap = plt.get_cmap('viridis')
    values = np.linspace(0, 1, 12)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # TODO Consider using month as a hue
    ax = sns.scatterplot(data=df, x=start + '_departure_time_hr', y='minutes_to_' + end, s=5)
    ax = time_xticks(ax, df[start + '_departure_time_hr'].min(), df[start + '_departure_time_hr'].max())

    # subset df containing 'month', start + '_departure_time_hr'
    df_ss = df[['month', start + '_departure_time_hr']]
    # drop rows where start + '_departure_time_hr' is NaN
    df_notna = df_ss.dropna()

    counts = df_notna['month'].value_counts().sort_index().to_list()
    Y = []
    monthly_avg = []
    for i in range(1, 13):
        color = colormap(values[i-1])
        x, y = model.prediction(reg, df, start, i)
        Y.append(y)
        monthly_avg.append(np.mean(y))
        ax.plot(x, y, color=color, label=months[i-1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set(xlabel=start.capitalize() + ' Departure Time',
       ylabel='Minutes to ' + end.capitalize(),
       title='Commuting Time')
    plt.savefig(f'{plots_folder}/monthly_prediction_curves_from_{start}_to_{end}.png',  bbox_inches='tight')
    plt.clf()

    # Bar chart with months as x-axis and +/- compared to average on y-axis
    Y = np.array(Y)
    y_avg = np.mean(Y)
    y_diff = monthly_avg - y_avg
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots()
    bars = ax.bar(months, y_diff, color=cmap(np.linspace(0, 1, len(months))))
    # Add text annotations to the bars
    for i, bar in enumerate(bars):
        if y_diff[i] >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), counts[i], ha='center', va='bottom')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), counts[i], ha='center', va='top')
    ax.set(xlabel='Month',
       ylabel=f'Time Delta Compared to Mean [Minutes]',
       title=f'Prediction Variations in Commuting Time from {start.capitalize()} to {end.capitalize()}')
    plt.savefig(f'{plots_folder}/monthly_prediction_variations_from_{start}_to_{end}.png',  bbox_inches='tight')
    plt.clf()


def departure_times_over_time(plots_folder, start, end, df):
    fig = plt.figure()
    # Apply the default theme
    sns.set_theme()

    # Initialize the visualization
    ax = sns.scatterplot(data=df, x='date', y=start + '_departure_time_hr', hue='day_of_week', hue_order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], legend=False)

    # Rotate and align the tick labels so they look better
    # fig.autofmt_xdate()

    ax.set(xlabel='Date',
       ylabel=start.capitalize() + ' Departure Time',
       title='Departure Time Evolution')
    plt.xticks(rotation=35)
    if end =='home':
        ax.axhline(17, color='c', linestyle='dotted')
    plt.savefig(f'{plots_folder}/departure_time_evolution_from_{start}_to_{end}.png',  bbox_inches='tight')
    plt.clf()


def arrival_times_over_time(plots_folder, start, end, df):
    fig = plt.figure()
    # Apply the default theme
    sns.set_theme()

    # Initialize the visualization
    ax = sns.scatterplot(data=df, x='date', y=end + '_arrival_time_hr', hue='day_of_week', hue_order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], legend=False)

    # Rotate and align the tick labels so they look better
    # fig.autofmt_xdate()

    ax.set(xlabel='Date',
       ylabel=end.capitalize() + ' Arrival Time',
       title='Arrival Time Evolution')
    plt.xticks(rotation=35)
    if end == 'work':
        ax.axhline(9, color='c', linestyle='dotted')
    plt.savefig(f'{plots_folder}/arrival_time_evolution_from_{start}_to_{end}.png',  bbox_inches='tight')
    plt.clf()

    mon_df = df[df['day_of_week'] == 'Mon']
    tue_df = df[df['day_of_week'] == 'Tue']
    wed_df = df[df['day_of_week'] == 'Wed']
    thu_df = df[df['day_of_week'] == 'Thu']
    fri_df = df[df['day_of_week'] == 'Fri']

    # Calculate the 80th percentile of commuting times for each day of the week
    mon_80 = mon_df['minutes_to_' + end].quantile(0.8)
    tue_80 = tue_df['minutes_to_' + end].quantile(0.8)
    wed_80 = wed_df['minutes_to_' + end].quantile(0.8)
    thu_80 = thu_df['minutes_to_' + end].quantile(0.8)
    fri_80 = fri_df['minutes_to_' + end].quantile(0.8)

    if end == 'work':
        print('80th percentile of commuting times for each day of the week:')
        print(round(mon_80, 1), round(tue_80, 1), round(wed_80, 1), round(thu_80, 1), round(fri_80, 1))
        print('Depart by the following times to arrive by 9 am 80 percent of the time:')
        mon_depart = 9 - (mon_80 / 60)
        tue_depart = 9 - (tue_80 / 60)
        wed_depart = 9 - (wed_80 / 60)
        thu_depart = 9 - (thu_80 / 60)
        fri_depart = 9 - (fri_80 / 60)
        # format the time as HH:MM
        mon_depart = ("%d:%02d" % (int(mon_depart), int((mon_depart*60) % 60))).format(mon_depart)
        tue_depart = ("%d:%02d" % (int(tue_depart), int((tue_depart*60) % 60))).format(tue_depart)
        wed_depart = ("%d:%02d" % (int(wed_depart), int((wed_depart*60) % 60))).format(wed_depart)
        thu_depart = ("%d:%02d" % (int(thu_depart), int((thu_depart*60) % 60))).format(thu_depart)
        fri_depart = ("%d:%02d" % (int(fri_depart), int((fri_depart*60) % 60))).format(fri_depart)
        print(mon_depart, tue_depart, wed_depart, thu_depart, fri_depart)
