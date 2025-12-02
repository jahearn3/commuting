import data_processing as dp
import make_plots as mp

trips = ['tenino', 'port_orchard_driving', 'port_orchard_ferry', 'bellevue']
for trip in trips:
    if trip in ['tenino', 'port_orchard_ferry', 'bellevue']:
        continue

    gbr = True if trip != 'bellevue' else False
    dtr = True if trip != 'bellevue' else False
    rfr = True if trip != 'bellevue' else False
    nn = False
    xgb = True if trip != 'bellevue' else False
    ensemble_r = True if trip != 'bellevue' else False

    filename = f'commuting_{trip}.csv'
    df = dp.process_data(filename)
    mp.duration_vs_departure(filename, df,
                             gbr=gbr, dtr=dtr, rfr=rfr, nn=nn,
                             xgb=xgb, ensemble_r=ensemble_r)
    mp.duration_vs_departure(filename, df, start='work', end='home',
                             gbr=gbr, dtr=dtr, rfr=rfr, nn=nn,
                             xgb=xgb, ensemble_r=ensemble_r)
    if trip == 'port_orchard_ferry':
        mp.driving_and_waiting_vs_departure(filename, df)
