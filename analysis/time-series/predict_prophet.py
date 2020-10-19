import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from fbprophet import Prophet


#####
# Create time series estimate of Australia covid cases to
# compare to my model
#####



def read_in_cases(case_file_date='29Jun'):
    """
    Read in NNDSS data
    """
    #from data, find rho
    from datetime import timedelta
    import glob
    
    path = "data/COVID-19 UoM "+case_file_date+"*.xlsx"
    for file in glob.glob(path):
        df_NNDSS = pd.read_excel(file,
                       parse_dates=['SPECIMEN_DATE','NOTIFICATION_DATE','NOTIFICATION_RECEIVE_DATE','TRUE_ONSET_DATE'],
                       dtype= {'PLACE_OF_ACQUISITION':str})
    if glob.glob(path) is None:
        print("No file found for ")
        print(path)
    df_NNDSS.PLACE_OF_ACQUISITION.fillna('00038888',inplace=True) #Fill blanks with simply unknown

    df_NNDSS['date_inferred'] = df_NNDSS.TRUE_ONSET_DATE
    df_NNDSS.loc[df_NNDSS.TRUE_ONSET_DATE.isna(),'date_inferred'] = df_NNDSS.loc[df_NNDSS.TRUE_ONSET_DATE.isna()].NOTIFICATION_DATE - timedelta(days=5)
    df_NNDSS.loc[df_NNDSS.date_inferred.isna(),'date_inferred'] = df_NNDSS.loc[df_NNDSS.date_inferred.isna()].NOTIFICATION_RECEIVE_DATE - timedelta(days=6)

    df_NNDSS['imported'] = df_NNDSS.PLACE_OF_ACQUISITION.apply(lambda x: 1 if x[-4:]=='8888' and x != '00038888' else 0)
    df_NNDSS['local'] = 1 - df_NNDSS.imported


    df_state = df_NNDSS[['date_inferred','STATE','imported','local']].groupby(['STATE','date_inferred']).sum()

    df_state['rho'] = [ 0 if (i+l == 0) else i/(i+l) for l,i in zip(df_state.local,df_state.imported)  ]
    return df_state

#inputs
data_date = pd.to_datetime('2020-10-12')
forecast_length = 28
truncation = 10

## Read in NNDSS data
cases = read_in_cases(case_file_date=data_date.strftime("%d%b"))



#include missing dates in index
cases = cases.local.unstack(level='STATE')
all_dates = pd.date_range("2020-03-01",data_date)

cases = cases.reindex(all_dates).fillna(0).astype(int)
cases = cases[['NSW','QLD','SA','TAS','VIC','WA','ACT','NT']]
# get jurisdictions
states = cases.columns.values

#initilise dataframes
results = pd.DataFrame()
df_prophet = pd.DataFrame()
df_prophet['ds'] = cases.index.values[:-truncation]

fig,ax = plt.subplots(ncols=2, nrows=3, sharex=True)
fig2, ax2 = plt.subplots(ncols=2, nrows=3, sharex=True)

os.makedirs("./results/prophet/"+data_date.strftime("%m-%d"), exist_ok=True)

for i, state in enumerate(states):
    #loop over each state. Parallelise?
    #initialise model 
    m = Prophet(mcmc_samples=1000)
    
    df_prophet['y'] = cases[state].values[:-truncation]

    m.fit(df_prophet)
    future = m.make_future_dataframe(
        periods = truncation+forecast_length,
        include_history=True)
    df_forecast = m.predict(future)
    forecast = m.predictive_samples(future)
    print(forecast)
    fig1 = m.plot(df_forecast,ax=ax[i//2,i%2])

    fig2 = m.plot_components(df_forecast)
    
    fig2.savefig("./results/prophet/"+data_date.strftime("%m-%d")+"/"+state+"components.png",dpi=144)
    
    samples = m.predictive_samples(future)
    temp = pd.DataFrame(samples['yhat'])
    temp['state'] = state
    temp['date'] = future.ds

    results = results.append(temp)
fig.savefig("./results/prophet/"+data_date.strftime("%m-%d")+"/forecast.png",dpi=144)
results.to_csv("./results/prophet"+data_date.srftime("%m-%d")+".csv")