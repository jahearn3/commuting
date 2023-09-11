import data_processing as dp
import make_plots as mp 

# filename = 'commuting_tenino.csv'
# df = dp.process_data(filename)
# mp.duration_vs_departure(filename, df, gbr=True, dtr=True, rfr=True, nn=False, xgb=True, ensemble_r=True)
# mp.duration_vs_departure(filename, df, start='work', end='home', gbr=True, dtr=True, rfr=True, nn=False, xgb=True, ensemble_r=True)

filename = 'commuting_port_orchard_driving.csv'
df = dp.process_data(filename)
# mp.duration_vs_departure(filename, df, gbr=False, dtr=False, rfr=False, nn=False, xgb=False, ensemble_r=False)
# mp.duration_vs_departure(filename, df, start='work', end='home', gbr=False, dtr=False, rfr=False, nn=False, xgb=False, ensemble_r=False)
# mp.duration_vs_departure(filename, df, gbr=True, dtr=True, rfr=True, nn=False, xgb=True, ensemble_r=True)
# mp.duration_vs_departure(filename, df, start='work', end='home', gbr=True, dtr=True, rfr=True, nn=False, xgb=True, ensemble_r=True)

filename = 'commuting_port_orchard_ferry.csv'
df = dp.process_data(filename)
# mp.duration_vs_departure(filename, df, gbr=False, dtr=False, rfr=False, nn=False, xgb=False, ensemble_r=False)
# mp.duration_vs_departure(filename, df, gbr=True, dtr=True, rfr=True, nn=False, xgb=True, ensemble_r=True)
mp.driving_and_waiting_vs_departure(filename, df)

filename = 'commuting_bellevue.csv'
df = dp.process_data(filename)
# mp.duration_vs_departure(filename, df, gbr=False, dtr=False, rfr=False, nn=False, xgb=False, ensemble_r=False)
# mp.duration_vs_departure(filename, df, start='work', end='home', gbr=False, dtr=False, rfr=False, nn=False, xgb=False, ensemble_r=False)


# from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error, mean_absolute_error, mean_squared_error
# y_pred = [11]
# y_test = [10]
# mape = mean_absolute_percentage_error(y_test, y_pred)
# print(mape)

# mde = median_absolute_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# print(mde, mae, mse)