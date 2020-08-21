###
# Run EpyReff on NNDSS data
###
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from epyreff import *

from sys import argv

## parameters
tau = 7
prior_a=1
prior_b=5
trunc_days = 21

date = argv[1]

# Read in the data

##read in case file data
df_interim = read_cases_lambda(date)

##generate dataframe with id_vars date and state, variable SOURCE and number of cases
df_linel = tidy_cases_lambda(df_interim)

##generate possible infection dates from the notification data
df_inf = draw_inf_dates(df_linel, nreplicates=1000,
                    shape_rd=2.77, scale_rd=3.17, offset_rd=0,
                    shape_inc=2.0/1.5, scale_inc=1.5, offset_inc=1,
    )

##reindex dataframe to include all dates, 
## return df with index (STATE, INFECTION_DATE, SOURCE), columns are samples
df_inc_zeros = index_by_infection_date(df_inf)


#get all lambdas
lambda_dict = lambda_all_states(df_inc_zeros,
                                shape_gen=2,scale_gen=1,offset=1, 
                                trunc_days=trunc_days)


#get all lambdas
lambda_DL = lambda_all_states(df_inc_zeros)

states = [*df_inc_zeros.index.get_level_values('STATE').unique()]
R_summary_states={}
dates = {}
df= pd.DataFrame()
for state in states:
    lambda_state = lambda_DL[state]
    df_state_I = df_inc_zeros.xs((state,'local'),level=('STATE','SOURCE'))
    #get Reproduciton numbers
    a,b,R = Reff_from_case(df_state_I.values,lambda_state,prior_a=prior_a, prior_b=prior_b, tau=tau)

    #summarise for plots and file printing
    R_summary_states[state] = generate_summary(R)
    dates[state] = df_state_I.index.values[trunc_days-1+tau:]
    
    temp =pd.DataFrame.from_dict(R_summary_states[state])
    temp['INFECTION_DATES'] = dates[state]
    temp['STATE'] = state
    #temp.index =pd.MultiIndex.from_product(([state], dates[state]))
    df = df.append(temp, ignore_index=True)

#make folder to record files
dir_path = os.path.dirname(os.path.realpath(__file__))
results_path =dir_path+"../../../results/EpyEstim/"
os.makedirs( results_path, exist_ok=True)

file_date = pd.to_datetime("2020"+date, format="%Y%d%b").strftime("%Y-%m-%d")
df.to_csv(results_path+'/Reff'+file_date+"tau_"+str(tau)+".csv",index=False)

#plot all the estimates
fig,ax = plot_all_states(R_summary_states,df_interim, dates, 
        start='2020-03-01',end=pd.to_datetime("today").strftime("%Y-%m-%d"),save=True,
        tau=tau, date=date
    )
