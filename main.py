import data_processing as dp
import make_plots as mp 

df = dp.process_data()
mp.duration_vs_departure(df, order=3, gbr=True, dtr=True, rfr=True, nn=False, xgb=True)
mp.duration_vs_departure(df, start='work', end='home', gbr=True, dtr=True, rfr=True, nn=False, xgb=True)