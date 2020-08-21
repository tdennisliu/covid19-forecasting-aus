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
import pickle

from Reff_functions import *
from Reff_constants import *

# Reff estimates from Price et al 2020
#df_Reff = read_in_Reff() #estimates up to 14th April

df_Reff = read_in_LSHTM()#read_in_Reff()


### Read in md surveys
surveys = pd.read_csv("../data/md/Barometer wave 1 to 10.csv",parse_dates = ['date'])
surveys = surveys.append(pd.read_csv("../data/md/Barometer wave 11 complience.csv",parse_dates=['date'])) #they spelt compliance wrong??

for i in range(12,21):
    surveys = surveys.append(pd.read_csv("../data/md/Barometer wave "+str(i)+" compliance.csv",parse_dates=['date']))

surveys.loc[surveys.state!='ACT','state'] = surveys.loc[surveys.state!='ACT','state'].map(states_initials).fillna(surveys.loc[surveys.state!='ACT','state'])
surveys['proportion'] = surveys['count']/surveys.respondents
surveys.date = pd.to_datetime(surveys.date)

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

sm = pickle.load(open('model/sm_pol_gamma.pkl','rb'))


data_date =  pd.to_datetime('2020-07-20')
#########
### here is where I can loop over to perform inference##
#######
df_Reff = df_Reff.loc[df_Reff.date_of_analysis==data_date.strftime("%Y-%m-%d")]

df_state = read_in_cases(case_file_date=data_date.strftime("%d%b%Y"))

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

##### Create useable dataset
## ACT and NT not in original estimates, need to extrapolated
states_to_fit = sorted(['NSW','VIC','QLD','SA','WA','TAS']) #sorting keeps consistent with sort in data_by_state
fit_post_March = True
ban = '2020-03-20'
start_date = '2020-03-01'
end_date = '2020-03-31'

##Second wave inputs
sec_states=sorted(['VIC'])
sec_start_date = '2020-06-01'
sec_end_date = '2020-07-07' #all we have for now

fit_mask = df.state.isin(states_to_fit)
if fit_post_March:
    fit_mask = (fit_mask) & (df.date >= start_date)
    
fit_mask = (fit_mask) & (df.date <= end_date )

second_wave_mask = df.state.isin(sec_states)
second_wave_mask = (second_wave_mask) & (df.date >= sec_start_date)
second_wave_mask = (second_wave_mask) & (df.date <= sec_end_date)


predictors = mov_values.copy()
#predictors.extend(['driving_7days','transit_7days','walking_7days','pc'])

#remove residential to see if it improves fit
predictors.remove('residential_7days')

df['post_policy'] = (df.date >= ban).astype(int)

dfX = df.loc[fit_mask].sort_values('date')
df2X = df.loc[second_wave_mask].sort_values('date')

data_by_state= {}
sec_data_by_state={}
for value in ['mean','std','local','imported']:
    data_by_state[value] = pd.pivot(dfX[['state',value,'date']], 
                           index='date',columns='state',values=value).sort_index(
        axis='columns')
    sec_data_by_state[value] = pd.pivot(df2X[['state',value,'date']], 
                           index='date',columns='state',values=value).sort_index(
        axis='columns')
    
mobility_by_state =[]
mobility_std_by_state=[]
count_by_state =[]
respond_by_state=[]

#FIRST PHASE
for state in states_to_fit:

    mobility_by_state.append(dfX.loc[dfX.state==state, predictors].values/100)
    mobility_std_by_state.append(
        dfX.loc[dfX.state==state,[val+'_std' for val in predictors]].values/100
    )
    count_by_state.append(survey_counts.loc[start_date:end_date,state].values)
    respond_by_state.append(survey_respond.loc[start_date:end_date,state].values)

sec_mobility_by_state =[]
sec_mobility_std_by_state=[]
sec_count_by_state=[]
sec_respond_by_state=[]

#SECOND PHASE
for state in sec_states:

    sec_mobility_by_state.append(df2X.loc[df2X.state==state, predictors].values/100)
    sec_mobility_std_by_state.append(
        df2X.loc[df2X.state==state,[val+'_std' for val in predictors]].values/100
    )
    sec_count_by_state.append(survey_counts.loc[sec_start_date:sec_end_date,state].values)
    sec_respond_by_state.append(survey_respond.loc[sec_start_date:sec_end_date,state].values)

    
policy_v = [1]*df2X.loc[df2X.state=='VIC'].shape[0]
policy = dfX.loc[dfX.state=='NSW','post_policy']



##Make state by state arrays
input_data ={
    'N': dfX.loc[dfX.state=='NSW'].shape[0],
    'K': len(predictors),
    'j':len(states_to_fit),
    'Reff': data_by_state['mean'].values,
    'Mob': mobility_by_state,
    'Mob_std':mobility_std_by_state,
    'sigma2': data_by_state['std'].values**2,
    'policy': policy.values,
    'local':data_by_state['local'].values,
    'imported':data_by_state['imported'].values,
    
    'N_v': df2X.loc[df2X.state=='VIC'].shape[0],
    'j_v': len(sec_states),
    'Reff_v': sec_data_by_state['mean'].values,
    'Mob_v': sec_mobility_by_state,
    'Mob_v_std':sec_mobility_std_by_state,
    'sigma2_v': sec_data_by_state['std'].values**2,
    'policy_v': policy_v,
    'local_v':sec_data_by_state['local'].values,
    'imported_v':sec_data_by_state['imported'].values,
    
    'count_md':count_by_state,
    'respond_md':respond_by_state,
    'count_md_v':sec_count_by_state,
    'respond_md_v':sec_respond_by_state,

}
iterations=10000
chains=2
fit = sm_pol_gamma.sampling(
    data=input_data,
    iter=iterations,
    chains=chains,
    #control={'max_treedepth':15}
)

filename = "stan_posterior_fit" + data_date.strftime("%Y-%m-%d") + ".txt"
with open("../data/"+filename, 'w') as f:
    print(fit.stansummary(pars=['bet','R_I','R_L','theta_md']), file=f)
samples_mov_gamma = fit.to_dataframe(pars=['bet','R_I','R_L','brho','theta_md'])

var_to_csv = predictors
samples_mov_gamma[predictors] = samples_mov_gamma[['bet['+str(i)+']' for i in range(1,1+len(predictors))]]
var_to_csv = ['R_I']+['R_L']+['theta_md']+predictors
today = datetime.strftime(datetime.today(),format='%Y-%m-%d')

samples_mov_gamma[var_to_csv].to_hdf('../data/soc_mob_posterior'+today+'.h5',key='samples')