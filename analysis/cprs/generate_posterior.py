####
# Read in LSHTM results and perform inference on 

####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.special import expit
from datetime import date, timedelta, datetime
import pystan

from Reff_functions import *
from Reff_constants import *

# Reff estimates from Price et al 2020
#df_Reff = read_in_Reff() #estimates up to 14th April

df_Reff = read_in_LSHTM()#read_in_Reff()


#########
### here is where I can loop over to perform inference##
#######
df_Reff = df_Reff.loc[df_Reff.date_of_analysis=='2020-07-16']

df_state = read_in_cases(case_file_date=['20Jul','0930'])

df_Reff = df_Reff.merge(df_state,how='left',left_on=['state','date'], right_on=['STATE','date_inferred']) #how = left to use Reff days, NNDSS missing dates
df_Reff['rho_moving'] = df_Reff.groupby(['state'])['rho'].transform(lambda x: x.rolling(7,1).mean()) #minimum number of 1

#some days have no cases, so need to fillna
df_Reff['rho_moving'] = df_Reff.rho_moving.fillna(method='bfill')
df_Reff['local'] = df_Reff.local.fillna(0)
df_Reff['imported'] = df_Reff.imported.fillna(0)
#Add Insight traffic
#df_ai = read_AddInsight()

df_google = read_in_google(local=True,moving=True)

df= df_google.merge(df_Reff[['date','state','mean','lower','upper',
                             'top','bottom','std','rho','rho_moving','local','imported']], on=['date','state'],how='inner')

### Read in md surveys
surveys = pd.read_csv("../data/md/Barometer wave 1 to 10.csv",parse_dates = ['date'])
surveys = surveys.append(pd.read_csv("../data/md/Barometer wave 11 complience.csv",parse_dates=['date'])) #they spelt compliance wrong??

for i in range(12,17):
    surveys = surveys.append(pd.read_csv("../data/md/Barometer wave "+str(i)+" compliance.csv",parse_dates=['date']))

surveys.loc[surveys.state!='ACT','state'] = surveys.loc[surveys.state!='ACT','state'].map(states_initials).fillna(surveys.loc[surveys.state!='ACT','state'])
surveys['proportion'] = surveys['count']/surveys.respondents
surveys.date = pd.to_datetime(surveys.date)
display(surveys)
always =surveys.loc[surveys.response=='Always'].set_index(["state",'date'])
always = always.unstack(['state'])


idx = pd.date_range('2020-03-01','2020-07-15')

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
survey_counts =pd.pivot_table(data=always,
                          index='date',columns='state',values='count').drop(['Australia','Other'],axis=1).astype(int)

survey_respond = pd.pivot_table(data=always,
                          index='date',columns='state',values='respondents').drop(['Australia','Other'],axis=1).astype(int)

## Read in pystan model that is saved on disk

