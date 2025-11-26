import data_processing as dp
import make_plots as mp

# filename = 'commuting_tenino.csv'
# df = dp.process_data(filename)
# mp.duration_vs_departure(filename, df,
#                          gbr=True, dtr=True, rfr=True, nn=False,
#                          xgb=True, ensemble_r=True)
# mp.duration_vs_departure(filename, df, start='work', end='home',
#                          gbr=True, dtr=True, rfr=True, nn=False,
#                          xgb=True, ensemble_r=True)

filename = 'commuting_port_orchard_driving.csv'
df = dp.process_data(filename)
# mp.duration_vs_departure(filename, df,
#                          gbr=False, dtr=False, rfr=False, nn=False,
#                          xgb=False, ensemble_r=False)
# mp.duration_vs_departure(filename, df, start='work', end='home',
#                          gbr=False, dtr=False, rfr=False, nn=False,
#                          xgb=False, ensemble_r=False)
mp.duration_vs_departure(filename, df,
                         gbr=True, dtr=True, rfr=True, nn=False,
                         xgb=True, ensemble_r=True)
mp.duration_vs_departure(filename, df, start='work', end='home',
                         gbr=True, dtr=True, rfr=True, nn=False,
                         xgb=True, ensemble_r=True)

# filename = 'commuting_port_orchard_ferry.csv'
# df = dp.process_data(filename)
# mp.duration_vs_departure(filename, df,
#                          gbr=False, dtr=False, rfr=False, nn=False,
#                          xgb=False, ensemble_r=False)
# mp.duration_vs_departure(filename, df,
#                          gbr=True, dtr=True, rfr=True, nn=False,
#                          xgb=True, ensemble_r=True)
# mp.driving_and_waiting_vs_departure(filename, df)

filename = 'commuting_bellevue.csv'
df = dp.process_data(filename)
# mp.duration_vs_departure(filename, df,
#                          gbr=False, dtr=False, rfr=False, nn=False,
#                          xgb=False, ensemble_r=False)
mp.duration_vs_departure(filename, df, start='work', end='home',
                         gbr=False, dtr=False, rfr=False, nn=False,
                         xgb=False, ensemble_r=False)
