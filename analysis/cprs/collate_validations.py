import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sys import argv

dates=("2020-04-01", "2020-04-08", "2020-04-15", "2020-04-22",
"2020-04-29" ,"2020-05-06", "2020-05-13","2020-05-20", "2020-05-27", "2020-06-03",
 "2020-06-10", "2020-06-17", "2020-06-24", "2020-07-01", "2020-07-08", 
 "2020-07-15", "2020-07-22", "2020-07-29", "2020-08-05", "2020-08-12", 
 "2020-08-19", "2020-08-26", "2020-09-02", "2020-09-16", "2020-09-23", 
 "2020-09-30", "2020-10-07", "2020-10-14", "2020-10-21") 
days_list=(
   60, 67, 74, 81, 88, 95, 102, 109, 116, 123, 130, 
    137, 144, 151, 158, 165, 172,179,186,193,200,207,
    214, #skip 221, data missing 2020-09-09
    228,235, 242, 249,256,263)

df = pd.DataFrame()

for i,date in enumerate(dates):
    states = ['NSW','QLD','SA','TAS','VIC','WA','ACT','NT']
    n_sims = int(argv[1])
    start_date = '2020-03-01'
    days = days_list[i]
    forecast_type = "R_L" #default None


    forecast_date = date #format should be '%Y-%m-%d'

    end_date = pd.to_datetime(start_date,format='%Y-%m-%d') + timedelta(days=days-1)
    sims_dict={
        'state': [],
        'onset date':[],
    }
    for n in range(n_sims):
        if n <2000:
            sims_dict['sim'+str(n)] = []

    print("forecast up to: {}".format(end_date))
    date_col = [day.strftime('%Y-%m-%d') for day in pd.date_range(start_date,end_date)]

    for i,state in enumerate(states):
        
        df_results = pd.read_parquet("./results/"+state+start_date+"sim_"+forecast_type+str(n_sims)+"days_"+str(days)+".parquet",columns=date_col)
        
        df_local = df_results.loc['total_inci_obs']

        sims_dict['onset date'].extend(date_col)
        sims_dict['state'].extend([state]*len(date_col))
        n=0
        print(state)
        for index, row in df_local.iterrows():
            if n==2000:
                break
            #if index>=2000:
            #    continue
            #else:
            if np.all(row.isna()):
                continue
            else:
                sims_dict['sim'+str(n)].extend(row.values)
                n +=1
        print(n)
        while n < 2000:
            print("Resampling")
            for index, row in df_local.iterrows():
                if n==2000:
                    break
                if np.all(row.isna()):
                    continue
                else:
                    sims_dict['sim'+str(n)].extend(row.values)
                    n +=1

    df_single = pd.DataFrame.from_dict(sims_dict)
    df_single["data date"] = forecast_date

    key ='local_obs'
    df_single[df_single.select_dtypes(float).columns] = df_single.select_dtypes(float).astype(int)
    #df.to_csv('./analysis/UoA_'+forecast_date+str(key)+'.csv')

    df = df.append(df_single)

df['data date'] = pd.to_datetime(df['data date'])
df['onset date'] = pd.to_datetime(df['onset date'])
#include some of nowcast but exclude backcast
mask = df['onset date'].values >= (df['data date'].values - 
pd.to_timedelta(7, unit='days'))
df = df.loc[mask]


df['data date'] = [val.strftime("%Y-%m-%d") for val in df['data date']]
df['onset date'] = [val.strftime("%Y-%m-%d") for val in df['onset date']]

print(df)
df.to_csv("./analysis/cprs/UoA_"+pd.to_datetime("today").strftime(
    "%Y-%m-%d"
    )+".csv")