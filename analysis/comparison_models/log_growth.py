####
# Read in LSHTM results and perform inference on

####
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.special import expit
from sys import argv
from datetime import timedelta, datetime
import pystan
import pickle
import os, glob
from Reff_functions import *
from Reff_constants import *


iterations=5000
chains=2

### Read in md surveys

surveys = pd.DataFrame()
##Improve this to read by glob.glob and get all of them

path = "data/md/Barometer wave*.csv"
for file in glob.glob(path):
    surveys = surveys.append(pd.read_csv(file,parse_dates=['date']))

surveys = surveys.sort_values(by='date')
print("Latest Microdistancing survey is {}".format(surveys.date.values[-1]))

surveys.loc[surveys.state!='ACT','state'] = surveys.loc[surveys.state!='ACT','state'].map(states_initials).fillna(surveys.loc[surveys.state!='ACT','state'])
surveys['proportion'] = surveys['count']/surveys.respondents
surveys.date = pd.to_datetime(surveys.date)

always =surveys.loc[surveys.response=='Always'].set_index(["state",'date'])
always = always.unstack(['state'])


idx = pd.date_range('2020-03-01',pd.to_datetime("today"))

always = always.reindex(idx, fill_value=np.nan)

always.index.name = 'date'

#fill back to earlier and between weeks.
# Assume survey on day x applies for all days up to x - 6
always =always.fillna(method='bfill')

#assume values continue forward if survey hasn't completed
always = always.fillna(method='ffill')
always = always.stack(['state'])

#Zero out before first survey 20th March
always = always.reset_index().set_index('date')
always.loc[:'2020-03-20','count'] =0
always.loc[:'2020-03-20','respondents'] =0
always.loc[:'2020-03-20','proportion'] =0

always = always.reset_index().set_index(['state','date'])

survey_X = pd.pivot_table(data=always,
                          index='date',columns='state',values='proportion')
survey_counts_base =pd.pivot_table(data=always,
                          index='date',columns='state',values='count').drop(['Australia','Other'],axis=1).astype(int)

survey_respond_base = pd.pivot_table(data=always,
                          index='date',columns='state',values='respondents').drop(['Australia','Other'],axis=1).astype(int)

## Read in pystan model that is saved on disk
rho_model_gamma = """
data {
    int N; //data length num days
    int K; //Number of mobility indices
    int j; //Number of states
    matrix[N,K] Mob[j]; //Mobility indices
    matrix[N,K] Mob_std[j]; ///std of mobility
    vector[N] policy; //Indicators for post policy or not
    matrix[N,j] local; //local number of cases
    matrix[N,j] imported; //imported number of cases


    int N_v; //length of VIC days
    int j_v; //second wave states
    matrix[N_v,K] Mob_v[j_v]; //Mob for VIC June
    matrix[N_v,K] Mob_v_std[j_v];// std of mobility
    vector[N_v] policy_v;// micro distancing compliance
    matrix[N_v,j_v] local_v; //local cases in VIC
    matrix[N_v,j_v] imported_v; //imported cases in VIC

    vector[N] count_md[j]; //count of always
    vector[N] respond_md[j]; // num respondants

    vector[N_v] count_md_v[j_v]; //count of always
    vector[N_v] respond_md_v[j_v]; // num respondants

    int map_to_state_index[j_v];// indices of second wave to map to first
    int a[j]; //carrying capacity of states
}
parameters {
    vector[K] bet; //coefficients
    real<lower=0> sig[j]; //variance
    real theta_md; // md weighting
    real bet_imported; //imported cases influence
    matrix<lower=0,upper=1>[N,j] prop_md; // proportion who are md'ing
    matrix<lower=0,upper=1>[N_v,j_v] prop_md_v;
}
transformed parameters {
    matrix<lower=0>[N,j] mu_hat;
    matrix<lower=0>[N_v,j_v] mu_hat_v;

    for (i in 1:j) {
        for (n in 1:N){
            mu_hat[n,i] = a[i]*inv_logit(
                Mob[i][n,:]*(bet) + theta_md* prop_md[n,i]
                + bet_imported *imported[n,i]
                ); //mean estimate
        }
    }
    for (i in 1:j_v){
        for (n in 1:N_v){

            mu_hat_v[n,i] = a[map_to_state_index[i]]*inv_logit(
                Mob_v[i][n,:]*(bet) + theta_md* prop_md_v[n,i]
                + bet_imported *imported_v[n,i]
                );
            }

        }

}
model {
    bet ~ normal(0,1);
    theta_md ~ normal(0,1);
    bet_imported ~ normal(0,1);


    sig ~ exponential(1); //mean is 1/5

    for (i in 1:j) {
        for (n in 1:N){
            prop_md[n,i] ~ beta(1 + count_md[i][n], 1+ respond_md[i][n] - count_md[i][n]);

            local[n,i] ~ normal(mu_hat[n,i], sig[i]) ;
        }
    }
    for (i in 1:j_v){
        for (n in 1:N_v){
            prop_md_v[n,i] ~ beta(1 + count_md_v[i][n], 1+ respond_md_v[i][n] - count_md_v[i][n]);
            
            local_v[n,i] ~ normal(mu_hat_v[n,i], sig[map_to_state_index[i]]);
        }
    }
}
"""


sm_pol_gamma = pystan.StanModel(
    model_code = rho_model_gamma,
    model_name ='gamma_pol_state'
)
#sm_pol_gamma = pickle.load(open('model/sm_pol_gamma.pkl','rb'))

###Create dates
try:
    cprs_start_date = pd.to_datetime(argv[1])#2020-04-01')
    cprs_end_date = pd.to_datetime(argv[1])#'2020-07-22')
except:
    print("Running full validation dates")
    cprs_start_date = pd.to_datetime('2020-04-01')
    cprs_end_date = pd.to_datetime('2020-10-07')
cprs_all_dates = pd.date_range(cprs_start_date, cprs_end_date, freq='7D')
cprs_dates = cprs_all_dates[cprs_all_dates!='2020-09-09']

#if argv[1]=='2020-09-09':
#    print("This won't run due to cprs date definitions, please comment out line 215.")

for data_date in cprs_dates:
    print(data_date.strftime('%d%b%Y'))

    #########
    ### here is where I can loop over to perform inference##
    #######

    if data_date < pd.to_datetime('2020-06-02'):
        #no leading zero on early dates
        if data_date.day <10:
            df_state = read_in_cases(case_file_date=data_date.strftime('%d%b%Y')[1:])
        else:
            df_state = read_in_cases(case_file_date=data_date.strftime('%d%b%Y'))
    else:
        df_state = read_in_cases(case_file_date=data_date.strftime('%d%b%Y'))

    df_Reff = df_state.reset_index().set_index(['STATE','date_inferred'])
    df_Reff.index = df_Reff.index.rename(['state','date'])
    print(df_Reff)
    print('cases loaded')
    #df_Reff = df_Reff.merge(df_state,how='left',left_on=['state','date'], right_on=['STATE','date_inferred']) #how = left to use Reff days, NNDSS missing dates
    df_Reff['rho_moving'] = df_Reff.groupby(['state'])['rho'].transform(lambda x: x.rolling(7,1).mean()) #minimum number of 1

    #some days have no cases, so need to fillna
    df_Reff['rho_moving'] = df_Reff.rho_moving.fillna(method='bfill')


    #shift counts to align with infection date not symptom date
    # dates should be complete at this point, no days skipped
    # will be some end days with NaN, but that should be fine since
    # we don't use the most recent 10 days
    df_Reff['local'] = df_Reff.local.shift(periods=-5)
    df_Reff['imported'] = df_Reff.imported.shift(periods=-5)
    df_Reff['rho_moving'] = df_Reff.rho_moving.shift(periods=-5)
    df_Reff['rho'] = df_Reff.rho.shift(periods=-5)
    df_Reff['local'] = df_Reff.local.fillna(0)
    df_Reff['imported'] = df_Reff.imported.fillna(0)
    #Add Insight traffic
    #df_ai = read_AddInsight()

    df_google = read_in_google(local=True,moving=True)
    df_Reff = df_Reff.reset_index()
    df= df_google.merge(df_Reff[['date','state','rho','rho_moving','local','imported']], on=['date','state'],how='inner')

    ##### Create useable dataset
    ## ACT and NT not in original estimates, need to extrapolated
    states_to_fit = sorted(['NSW','VIC','QLD','SA','WA','TAS']) #sorting keeps consistent with sort in data_by_state
    fit_post_March = True
    ban = '2020-03-20'
    start_date = '2020-03-01'
    end_date = '2020-03-31'

    ##Second wave inputs
    sec_states=sorted(['NSW','VIC'])
    sec_start_date = '2020-06-01'
    if data_date > pd.to_datetime("2020-06-12"):
        if data_date < pd.to_datetime("2020-10-01"):
            possible_end_date = data_date - timedelta(10)#subtract 10 days to aovid right truncation
        else:
            possible_end_date = pd.to_datetime("2020-09-21")
    else:
        possible_end_date = pd.to_datetime("2020-06-01")
    sec_end_date = possible_end_date.strftime('%Y-%m-%d')
    #min('2020-08-14',possible_end_date.strftime('%Y-%m-%d')) #all we have for now

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

    #filter out the surveys we don't have
    if df2X.shape[0]>0:
        survey_respond = survey_respond_base.loc[:df2X.date.values[-1]]
        survey_counts = survey_counts_base.loc[:df2X.date.values[-1]]
    else:
        survey_respond = survey_respond_base.loc[:dfX.date.values[-1]]
        survey_counts = survey_counts_base.loc[:dfX.date.values[-1]]



    data_by_state= {}
    sec_data_by_state={}
    for value in ['local','imported']:
        data_by_state[value] = pd.pivot(dfX[['state',value,'date']],
                            index='date',columns='state',values=value).sort_index(
            axis='columns')
        sec_data_by_state[value] = pd.pivot(df2X[['state',value,'date']],
                            index='date',columns='state',values=value).sort_index(
            axis='columns')
            #account for dates pre second wave
        if df2X.loc[df2X.state==sec_states[0]].shape[0]==0:
            print("making empty")
            sec_data_by_state[value] = pd.DataFrame(columns=sec_states).astype(float)

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

    policy_v = [1]*df2X.loc[df2X.state==sec_states[0]].shape[0]
    policy = dfX.loc[dfX.state==states_to_fit[0],'post_policy']


    state_index = { state : i+1  for i, state in enumerate(states_to_fit)}
    pop = {
    'NSW': 5730000,
    'VIC': 5191000,
    'SA': 1408000,
    'WA': 2385000,
    'TAS': 240342,
    'NT': 154280,
    'ACT': 410199,
    'QLD': 2560000,
        }
    a = [pop[state] for state in states_to_fit]
    ##Make state by state arrays
    print(data_by_state['imported'])
    input_data ={
        'N': dfX.loc[dfX.state==states_to_fit[0]].shape[0],
        'K': len(predictors),
        'j':len(states_to_fit),
        'Mob': mobility_by_state,
        'Mob_std':mobility_std_by_state,
        'policy': policy.values,
        'local':data_by_state['local'].values,
        'imported':data_by_state['imported'].values,

        'N_v': df2X.loc[df2X.state==sec_states[0]].shape[0],
        'j_v': len(sec_states),
        'Mob_v': sec_mobility_by_state,
        'Mob_v_std':sec_mobility_std_by_state,
        'policy_v': policy_v,
        'local_v':sec_data_by_state['local'].values,
        'imported_v':sec_data_by_state['imported'].values,

        'count_md':count_by_state,
        'respond_md':respond_by_state,
        'count_md_v':sec_count_by_state,
        'respond_md_v':sec_respond_by_state,
        'map_to_state_index': [state_index[state] for state in sec_states],
        'a': a,

    }

    fit = sm_pol_gamma.sampling(
        data=input_data,
        iter=iterations,
        chains=chains,
        #control={'max_treedepth':15}
    )

    #make results dir
    results_dir ="analysis/comparision_models/results/"
    os.makedirs(results_dir,exist_ok=True)

    filename = "stan_posterior_fit" + data_date.strftime("%Y-%m-%d") + ".txt"
    with open(results_dir+filename, 'w') as f:
        print(fit.stansummary(pars=['bet','theta_md','bet_imported']), file=f)
    samples_mov_gamma = fit.to_dataframe(pars=['bet','theta_md','bet_imported'])

    # Plot ratio of imported to total cases
    # First phase
    #rho calculated at data entry
    if isinstance(df_state.index, pd.MultiIndex):
        df_state = df_state.reset_index()


    states=sorted(['NSW','QLD','VIC','TAS','SA','WA','ACT','NT'])
    fig,ax = plt.subplots(figsize=(24,9), ncols=len(states),sharey=True)
    states_to_fitd = {state: i+1 for i,state in enumerate(states_to_fit)      }

    for i, state in enumerate(states):
        if state in states_to_fit:
            dates = df_Reff.loc[(df_Reff.date>=start_date) &
                                (df_Reff.state==state)&(df_Reff.date<=end_date)].date
            rho_samples = samples_mov_gamma[['brho['+str(j+1)+','+str(states_to_fitd[state])+']' for j in range(dfX.loc[dfX.state==states_to_fit[0]].shape[0])]]
            ax[i].plot(dates, rho_samples.median(),label='fit',color='C0')
            ax[i].fill_between(dates, rho_samples.quantile(0.25),rho_samples.quantile(0.75),color='C0',alpha=0.4)

            ax[i].fill_between(dates, rho_samples.quantile(0.05),rho_samples.quantile(0.95),color='C0',alpha=0.4)
        else:
            sns.lineplot(x='date_inferred',y='rho',
                data=df_state.loc[(df_state.date_inferred>=start_date) & (df_state.STATE==state)&(df_state.date_inferred<=end_date)], ax=ax[i],color='C1',label='data')
        sns.lineplot(x='date',y='rho',
                data=df_Reff.loc[(df_Reff.date>=start_date) & (df_Reff.state==state)&(df_Reff.date<=end_date)], ax=ax[i],color='C1',label='data')
        sns.lineplot(x='date',y='rho_moving',
                data=df_Reff.loc[(df_Reff.date>=start_date) & (df_Reff.state==state)&(df_Reff.date<=end_date)], ax=ax[i],color='C2',label='moving')

        dates = dfX.loc[dfX.state==states_to_fit[0]].date

        ax[i].tick_params('x',rotation=20)
        ax[i].xaxis.set_major_locator(plt.MaxNLocator(4))
        ax[i].set_title(state)
    ax[0].set_ylabel('Proportion of imported cases')
    plt.legend()
    plt.savefig(results_dir+data_date.strftime("%Y-%m-%d")+"rho_first_phase.png",dpi = 144)

    # Second phase
    if df2X.shape[0]>0:
        fig,ax = plt.subplots(figsize=(24,9), ncols=len(sec_states),sharey=True, squeeze=False)
        states_to_fitd = {state: i+1 for i,state in enumerate(sec_states)      }

        for i, state in enumerate(sec_states):
            #Google mobility only up to a certain date, so take only up to that value
            dates = pd.date_range(start=sec_start_date,
            end=df2X.loc[df2X.state==sec_states[0]].date.values[-1])
            #df_Reff.loc[(df_Reff.date>=sec_start_date) &
            #                    (df_Reff.state==state)&(df_Reff.date<=sec_end_date)].date
            rho_samples = samples_mov_gamma[
                ['brho_v['+str(j+1)+','+str(states_to_fitd[state])+']'
                for j in range(df2X.loc[df2X.state==sec_states[0]].shape[0])]
                ]
            ax[0,i].plot(dates, rho_samples.median(),label='fit',color='C0')
            ax[0,i].fill_between(dates, rho_samples.quantile(0.25),rho_samples.quantile(0.75),color='C0',alpha=0.4)

            ax[0,i].fill_between(dates, rho_samples.quantile(0.05),rho_samples.quantile(0.95),color='C0',alpha=0.4)

            sns.lineplot(x='date_inferred',y='rho',
                data=df_state.loc[(df_state.date_inferred>=sec_start_date) & (df_state.STATE==state)&(df_state.date_inferred<=sec_end_date)], ax=ax[0,i],color='C1',label='data')
            sns.lineplot(x='date',y='rho',
                    data=df_Reff.loc[(df_Reff.date>=sec_start_date) & (df_Reff.state==state)&(df_Reff.date<=sec_end_date)], ax=ax[0,i],color='C1',label='data')
            sns.lineplot(x='date',y='rho_moving',
                    data=df_Reff.loc[(df_Reff.date>=sec_start_date) & (df_Reff.state==state)&(df_Reff.date<=sec_end_date)], ax=ax[0,i],color='C2',label='moving')

            dates = dfX.loc[dfX.state==sec_states[0]].date

            ax[0,i].tick_params('x',rotation=20)
            ax[0,i].xaxis.set_major_locator(plt.MaxNLocator(4))
            ax[0,i].set_title(state)
        ax[0,0].set_ylabel('Proportion of imported cases')
        plt.legend()
        plt.savefig(results_dir+data_date.strftime("%Y-%m-%d")+"rho_sec_phase.png",dpi = 144)

    #plot marginal distributions
    fig,ax = plt.subplots(figsize=(12,9))

    posterior = samples_mov_gamma[['bet['+str(i)+']' for i in range(1,1+len(predictors))]
                                ]

    split=True
    md = 'power'#samples_mov_gamma.md.values

    posterior.columns = [val for val in predictors]
    long = pd.melt(posterior)

    #plot coefficients
    fig,ax2 =plt.subplots(figsize=(12,9))

    ax2 = sns.violinplot(x='variable',y='value',#hue='policy',
                        data=long,
                        ax=ax2,
                        color='C0'
                    )


    ax2.plot([0]*len(predictors), linestyle='dashed',alpha=0.6, color = 'grey')
    ax2.tick_params(axis='x',rotation=90)

    #ax =plot_posterior_violin(posterior)

    #ax2.set_title('Coefficients of mobility indices')
    ax2.set_xlabel('Social mobility index')
    ax2.set_xticklabels([var[:-6] for var in predictors])
    ax2.tick_params('x',rotation=15)
    plt.savefig(
        results_dir+data_date.strftime("%Y-%m-%d")+'mobility_posteriors.png', dpi =144)


    if df2X.shape[0]>0:
        #plot only if there is second phase data
        ax4 =predict_plot(samples_mov_gamma,df.loc[(df.date>=sec_start_date)&(df.date<=sec_end_date)],gamma=True, moving=True,split=split,grocery=True,ban = ban,
                        R=RL_by_state, var= True, md_arg=md,
                        rho=sec_states, second_phase=True,
                        R_I =samples_mov_gamma.R_I.values,prop=survey_X.loc[sec_start_date:sec_end_date])#by states....
        for ax in ax4:
            for a in ax:
                a.set_ylim((0,3))
                #a.set_xlim((start_date,end_date))
        plt.savefig(
            results_dir+data_date.strftime("%Y-%m-%d")+"Reff_sec_phase.png", dpi=144)

        #remove plots from memory
        fig.clear()
        plt.close(fig)


    var_to_csv = predictors
    samples_mov_gamma[predictors] = samples_mov_gamma[['bet['+str(i)+']' for i in range(1,1+len(predictors))]]
    var_to_csv = ['R_I']+['R_L','sig']+['theta_md']+predictors + [
        'R_Li['+str(i+1)+']' for i in range(len(states_to_fit))
        ]


    samples_mov_gamma[var_to_csv].to_hdf(results_dir+'/soc_mob_posterior'+data_date.strftime("%Y-%m-%d")+'.h5',key='samples')
