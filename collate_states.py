import pandas as pd

states = ['NSW','QLD','SA','TAS','VIC','WA','ACT','NT']
start_date = '2020-03-01'

forecast_type= 'R_L'
n_sims = 20
days = 80
dic_states={
    'state':[],
    'date':[],
    'type':[],
    'bottom':[],
    'lower':[],
    'median':[],
    'upper':[],
    'top':[],
}
dates =pd.date_range(start = start_date,
                        periods=days #num of days
                    )
vars_l = ['symp_inci_obs','imports_inci_obs','asymp_inci_obs','symp_inci','asymp_inci','imports_inci','total_inci','total_inci_obs']
for var in vars_l:
    for state in states:
        df = pd.read_parquet(
            "./results/"+state+start_date+"sim_"+forecast_type+str(n_sims)+"days_"+str(days)+".parquet")
        df = df[[col.strftime('%Y-%m-%d') for 
                    col in dates]]


        quantiles = df.loc[var].quantile([0.05,0.25,0.5,0.75,0.95],axis=0)
        dic_states['state'].extend([state]*len(dates))
        dic_states['date'].extend(df.columns)
        dic_states['type'].extend([var]*len(dates))
        dic_states['bottom'].extend(quantiles.loc[0.05])
        dic_states['lower'].extend(quantiles.loc[0.25])
        dic_states['median'].extend(quantiles.loc[0.50])
        dic_states['upper'].extend(quantiles.loc[0.75])
        dic_states['top'].extend(quantiles.loc[0.95])

plots =pd.DataFrame.from_dict(dic_states)
plots.to_parquet('./results/quantiles'+forecast_type+start_date+"sim_"+str(n_sims)+"days_"+str(days)+".parquet")
        
