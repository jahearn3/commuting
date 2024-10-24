# Introduction
In the second week of my job, I started tracking my commuting times to and from work. I wanted to see how big of an effect rush hour would have on my commutes. I also wanted an easy way to practice Machine Learning (ML) techniques that I’ve recently learned about. 

# Data Collection
For this sort of project, I collect my own data. Before shifting gears into drive, I write down my departure time and the odometer reading (mileage) in a notebook that I have with me. I do this again as soon as I park upon arrival at my destination. Later on I transfer these entries into a csv file using Microsoft Excel. 
I do not always record my commutes. If I stop for groceries, for example, or go somewhere else after work, then I do not record anything. I still do write down the entry when I stop briefly, such as for gas, but for these cases I have comment columns in my csv file where I include those notes. 
I have different datasets that correspond to different routes, which are due to moving, taking the ferry, or needing to commute to my company's corporate headquarters on some days. 

# Data Preparation
The rest of the project uses Python and takes place in Visual Studio Code. I have a couple of functions to preprocess the data. 
The first of these starts with loading the csv file using the pandas read_csv function. Then I change the types of the time columns to be datetime types. I next calculate the number of minutes from home to work and from work to home and store these as new columns. They will become the y axes in my main plots. Next, because the x axes will be the departure time, I need to keep only the departure time without also including the date. To do this, I subtracted the date of the commute and the time of midnight from the departure time with the date. I stored these as new columns. I also calculated the mileage difference between my origin and destination, and stored these as new columns as well. 
The other function in my data processing module does some feature engineering. It starts by using the pandas function to get dummy variables for the days of the week, dropping one of them. I then drop null rows from days when I only recorded one trip that day but happen to be processing the data subset for the other direction. Next I set the predictors (X) and target (y) equal to the appropriate columns from my dataframe. For my predictors, I am using the departure time as well as dummy columns for Monday through Thursday. My target is the number of minutes to get to my destination. Lastly, I use the scikit-learn function to do a train-test split of the X and y columns, including 30% of the rows in the test dataset, and using a random state of 3 (not 42 because I’m a rebel) to ensure reproducibility. 

I added as features to my regression Machine Learning models the following features:
• Quarter
• sin(Month)
• cos(Month)

Using sine and cosine is an attempt to capture the cyclic nature of the months. 

Now when I generate a line for my predictions, the default is to use the current month (see first set of plots).

# Plot Initialization
Next I call from my make_plots module my duration_vs_departure function for each direction of the commute. All the model training and plotting and everything else is done from within this function. 
I start with setting the pyplot figure and seaborn theme. Then I use seaborn’s scatterplot function to plot the departure time along the x-axis and the minutes to my destination along the y-axis. The hue (colors) of the data points is set according to the day of the week. It is necessary to specify the order of these days, lest they default to alphabetical order. I set the size of the points to 1 because I then call the scatterplot function again, letting the size be determined by the mileage difference between arrival and departure, and setting legend equal to false. That way, the data points can be sized according to the mileage of the trip, but the quantities corresponding to different size points won’t show up in the legend, which will already be crowded enough with the days of the week and a few other lines. 
The next step was to get my x-ticks in order. I wrote a function for this, whose main purpose is to change the x-axis ticks from decimal form to HH:MM form. It also ensures a constant number of x-ticks (6 seems appropriate) and sets the limits just outside the earliest and latest departures. 



# Model Training
I practiced one algorithm at a time, starting with the most basic, an ordinary least squares (OLS) linear prediction from the statsmodels library. Then I proceeded through gradient boosting, decision tree, random forest, and XGBoost regressors. I finished with an ensemble model. 
The OLS line is the only one that I did not use the split training and test sets for but instead used all the data because I was using it more as a sanity check. Another part of my sanity check is to print out the shapes of the training and test datasets. 
I will go through four of the ML algorithms with one broad stroke: regression using gradient boosting, XGBoost, a decision tree, and a random forest. I enabled each of these to run with specified hyperparameters or do a grid search with cross-validation. The training dataset is used to fit the model, and then the X_test data is used to predict the target y-value. I wrote a function to compute the mean squared error for the training, test, and complete datasets. This helped to see whether some algorithms were overfitting the data. Also to avoid overfitting in the graphical representation, I've written an adaptive spacing function to have more clustered predictions wherever the data is more clustered, but with minimum and maximum spacing between prediction points. I run the scikit-learn predict function thrice, specifying a different travel day each time, assuming the trip would take place on either Monday, Tuesday, or Thursday (the three days I typically commute). Then I take the average of the three. 
I also generated the predictions according to each month. 
After those four ML models have run, I combine them in an ensemble model. A voting regressor is instantiated, in which each model gets one vote. I then fit and predict for the ensemble model in the same way as before. The ensemble model typically results in the predictions I would trust the most, and so this has become the only one I plot.  

# Finishing Up
👉 Added text below the legend that provides stats related to my most recent trip (which is highlighted on the plot in yellow) compared to similar trips. 

👉 Added dotted lines for the mean and mean + 3 times the standard deviation. Any trips above this line get filtered out as outliers. (On that one outlier trip to work, I had driven to the ferry, but it was full, so then I had to drive around.)

👉 Reduced the number of solid lines that display the ML model predictions to one (less clutter), which corresponds to the ensemble model, but I've renamed it "ML Prediction" to make it easier to understand to the layperson. It's still derived from combining gradient boosting, XGBoost, decision tree, and random forest regressor models. 

👉 Modified the adaptive spacing function I wrote to have more clustered predictions wherever the data is more clustered, but with minimum and maximum spacing between prediction points. 

⏰ One more enhancement, not displayed in the plot, is calculation of what time I should depart home on each day of the week in order to badge in at work by 9 am 80% of the time (when traffic is really bad I share this plot with my manager so that I don't get in trouble). Just yesterday I finally set alarms on my phone to go off 10 minutes before that time. 

After then making the legend, tidying up the axis labels and title, and making a directory for the plots if it doesn’t already exist, it is time to save the plot. I then make a few additional plots for post-analysis, which I will only briefly mention. I plot the residuals using seaborn’s residplot function, the training deviance and feature importances for the gradient boosting regression model, and violin plots for commuting minutes and departure time, both as a whole and by weekday. 
Due to the addition of the ferry option, I’ve been experimenting with plots to visualize the exploration of more nuances, such as driving time (subtracting out the waiting time) and a driving-to-waiting ratio.
But to visualize the differences better, I made bar charts to show the difference in the predictions by month compared to the mean (see final set of plots), with the number on top or below each bar indicating how many data points went into the input for that month. 

# Flourishing Touches
Once I had a lot of data points, it became more difficult to pinpoint which one corresponded to my most recent trip, so I decided to highlight the most recent trip with a yellow halo around the data point. To do this, between my first and second seaborn scatterplot calls, I get the last x and y coordinates from the data subset, and then do a pyplot scatter of that single datum, setting the color to #ffff14 (because I wanted to call a color by a hex code at least once in my life) and the size to 100 (after some experimentation with sizes). 
Another feature I added was to time how long it took for each ML algorithm to run. Of course, the output of this was quite different depending on whether I used grid-search cross-validation or specified the parameters outright. It could also be thrown off if my laptop was performing other tasks at the time. Using grid-search, here are the times from the most recent time I ran the code:
Gradient Boosting Regressor 2.67 s
Decision Tree Regressor 0.25 s
Random Forest Regressor 29.01 s
XGBoost Regressor 0.88 s
Ensemble Voting Regressor 0.27 s
The Random Forest Regressor sticks out as taking longer than the others. By specifying the parameters outright, the times of the first four models is reduced. 

# Future Steps
A quick enhancement I may do soon is to add one-sided error bars to the scatterplot points to correspond to trips on which I stopped for gas. 
At one point I attempted to use a neural network, but got all zeros. I have not yet dedicated time to debug this but hope to do so soon, even though a neural network is overkill for this type of project. 
I intend to add post-analysis for the other ML models instead of only for Gradient Boosting Regression. 


