import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from datetime import date, timedelta, datetime

from Reff_functions import read_in_cases, read_in_google
from Reff_constants import *

def read_in_posterior(date='2020-07-18'):
    """
    read in samples from posterior from inference
    """
    df = pd.read_hdf("../data/soc_mob_posterior"+date+".h5", key='samples')
    
    return df



df_google_all = read_in_google(Aus_only=True,moving=True,local=True)
states = ['NSW','QLD','SA','VIC','TAS','WA','ACT','NT']#,'AUS']
plot_states = states.copy()
#plot_states.remove('AUS')
df_samples = read_in_posterior()

## grab survey data

surveys = pd.read_csv("../data/md/Barometer wave 1 to 10.csv",parse_dates = ['date'])
surveys = surveys.append(pd.read_csv("../data/md/Barometer wave 11 complience.csv",parse_dates=['date'])) #they spelt compliance wrong??

for i in range(12,16):
    surveys = surveys.append(pd.read_csv("../data/md/Barometer wave "+str(i)+" compliance.csv",parse_dates=['date']))

surveys.loc[surveys.state!='ACT','state'] = surveys.loc[surveys.state!='ACT','state'].map(states_initials).fillna(surveys.loc[surveys.state!='ACT','state'])
surveys['proportion'] = surveys['count']/surveys.respondents
surveys.date = pd.to_datetime(surveys.date)

always =surveys.loc[surveys.response=='Always'].set_index(["state",'date'])
always = always.unstack(['state'])

#fill in date range
idx = pd.date_range('2020-03-01','2020-07-08')

always = always.reindex(idx, fill_value=np.nan)

always.index.name = 'date'

always =always.fillna(method='bfill')
always = always.stack(['state'])

#Zero out before first survey 20th March
always = always.reset_index().set_index('date')
always.loc[:'2020-03-20','count'] =0
always.loc[:'2020-03-20','respondents'] =0
always.loc[:'2020-03-20','proportion'] =0

always = always.reset_index().set_index(['state','date'])

survey_X = pd.pivot_table(data=always,
                          index='date',columns='state',values='proportion')
prop_all = survey_X

###Create dates
cprs_start_date = pd.to_datetime('2020-04-01')
cprs_end_date = pd.to_datetime('2020-07-22')

cprs_dates = pd.date_range(cprs_start_date, cprs_end_date, freq='7D')

for data_date in cprs_dates:

    cases = read_in_cases(data_date.strftime('%d%b%Y'))
    
    one_month = data_date + timedelta(
    days= 42)

    days_from_March = one_month.dayofyear -pd.to_datetime('2020-03-01').dayofyear

    ##filter out future info
    prop = prop_all.loc[:data_date]
    df_google = df_google_all.loc[df_google.date<=data_date]
    
    #forecast time parameters
    n_training =28
    today = data_date.strftime('%Y-%m-%d')
    n_forecast = 28

    #cap = 0 #10?
    training_start_date = datetime(2020,3,1,0,0)
    print("Forecast ends at {} days after 1st March".format(
        pd.to_datetime(today).dayofyear + 28 - pd.to_datetime(training_start_date).dayofyear)
        )
    print("Final date is {}".format(pd.to_datetime(today) + timedelta(days=28)))
    df_google = df_google.loc[df_google.date>= training_start_date]
    outdata = {'date': [],
            'type': [],
            'state': [],
                'mean': [],
                'std' : [],
            }
    predictors = mov_values.copy()
    predictors.remove('residential_7days')

    mob_samples = 1000
        
        
    state_Rmed = {}
    state_sims = {}
    for i,state in enumerate(states):
        

        rows = df_google.loc[df_google.state==state].shape[0]
        #Rmed currently a list, needs to be a matrix
        Rmed_array = np.zeros(shape=(rows,len(predictors), mob_samples))
        for j, var in enumerate(predictors):
            for n in range(mob_samples):
                Rmed_array[:,j,n] = df_google[df_google['state'] == state][var].values.T + np.random.normal(loc=0,
                    scale = df_google[df_google['state'] == state][var+'_std'],
               )
        dates = df_google[df_google['state'] == state]['date']
            
            #cap min and max at historical or (-50,0)
        minRmed_array = np.minimum(-50,np.amin(Rmed_array, axis = 0)) #1 by predictors by mob_samples size
        maxRmed_array = np.maximum(0,np.amax(Rmed_array, axis=0))
        sims  =  np.zeros(shape=(n_forecast,len(predictors),mob_samples)) # days by predictors by samples
        for n in range(mob_samples):
            Rmed = Rmed_array[:,:,n]
            minRmed = minRmed_array[:,n]
            maxRmed = maxRmed_array[:,n]
                
            R_diffs = np.diff(Rmed[-n_training:,:], axis=0)


            mu = np.mean(R_diffs, axis=0)
            std = np.cov(R_diffs, rowvar=False) #columns are vars, rows are obs
            sims[:,:,n] = np.minimum(maxRmed,np.mean(Rmed[-7:,:],axis=0) + np.cumsum(np.random.multivariate_normal(mu,
                                                                    std,
                                                                    size=(n_forecast)),
                                                    axis=0))#rows are sim, dates are columns
            sims[:,:,n] = np.maximum(minRmed, sims[:,:,n])
                #dates of forecast to enter
                
        dd = [dates.tolist()[-1] + timedelta(days=x) for x in range(1,n_forecast+1)]

            #     print(state)

        sims_med = np.median(sims,axis=2) #N by predictors
        sims_q25 = np.percentile(sims,25,axis=2)
        sims_q75 = np.percentile(sims,75,axis=2)
        
        
        ##forecast mircodistancing
        if state!='AUS':
            md_diffs = np.diff(prop[state].values[-n_training:])
            mu = np.mean(md_diffs)
            std = np.std(md_diffs)
            extra_days_md = pd.to_datetime(df_google.date.values[-1]).dayofyear - pd.to_datetime(
                prop[state].index.values[-1]).dayofyear
            md_sims = np.minimum(1,prop[state].values[-1] + np.cumsum(
                np.random.normal(mu,std, size=(n_forecast + extra_days_md, 1000)),
                                axis=0
            )
                                )
            md_sims = np.maximum(0, md_sims)
            #get dates
            dd_md = [prop[state].index[-1] + timedelta(days=x) for x in range(1,n_forecast+extra_days_md+1)]

        
        for j, var in enumerate(predictors+['md_prop']):
            #Record data
            axs = axes[j]
            if (state=='AUS') and (var=='md_prop'):
                continue

            if var != 'md_prop':
                outdata['type'].extend([var]*len(dd))
                outdata['state'].extend([state]*len(dd))
                outdata['date'].extend([d.strftime('%Y-%m-%d') for d in dd])
                outdata['mean'].extend(np.mean(sims[:,j,:],axis=1))
                outdata['std'].extend(np.std(sims[:,j,:],axis=1))
            else:
                outdata['type'].extend([var]*len(dd_md))
                outdata['state'].extend([state]*len(dd_md))
                outdata['date'].extend([d.strftime('%Y-%m-%d') for d in dd_md])
                outdata['mean'].extend(np.mean(md_sims,axis=1))
                
                outdata['std'].extend(np.std(md_sims,axis=1))


        state_Rmed[state] = Rmed_array
        state_sims[state] = sims

