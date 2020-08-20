import pandas as pd
from numpy.random import beta, gamma

from ct_sim_class import *

##########
#PARAMETERS TO PLAY WITH
#########
time_end = 30
forecast_type = 'R_L0'
state = 'NSW'
case_file_date = None #'24Jul'
#Reff_file_date = '2020-07-20'
Reff_file_date = '2020-08-06'
#Number of initial symptomatic and asymptomatic cases respectively
initial_cases = [10,0]

# Laura
# sets the seed for only the `action_time’ for the initial cases, 
# and so all simulations will start with the same initial cases and the same time to action
np.random.seed(1)

#############
### These parameters do not need to be changed, ask DL
XBstate = None
start_date = '2020-03-01'
test_campaign_date = '2020-06-25'
test_campaign_factor = 1.25
R_I='R_I'
abc =False
forecast_date = '2020-03-02'
##############################



print("Simulating state " +state)


##Initialise the number of cases as 1st of March data incidence
current = {
    'ACT':[0,0,0],
    'NSW':[10,0,2], #1
    'NT':[0,0,0],
    'QLD':[2,0,0],
    'SA':[2,0,0],
    'TAS':[0,0,0],
    'VIC':[2,0,0], #1
    'WA':[0,0,0],
 } 
current ={state: [0,initial_cases[0],initial_cases[1]]}
forecast_dict = {}

initial_people = ['I']*current[state][0] + \
        ['A']*current[state][1] + \
        ['S']*current[state][2]
people = {}

if abc:
    qs_prior = beta(2,2,size=10000)
    qi_prior = beta(2, 2, size=10000)
    qa_prior = beta(2,2, size=10000)
    #qi_prior = [qi_d[state]]
    #qs_prior = [local_detection[state]]
    #qa_prior = [a_local_detection[state]]
    gam =0.1 + beta(2,2,size=10000) *0.9 #np.minimum(3,gamma(4,0.25, size=1000))
    ps_prior = 0.1+beta(2,2,size=10000)*0.9

else:
    qi_prior = [qi_d[state]]
    qs_prior = [local_detection[state]]
    qa_prior = [a_local_detection[state]]
    gam =[1/2]
    ps_prior = 0.7
    ps_prior= [ps_prior]

##create dictionary to input intial People
# Laura
# give action_times to each initial case
t_a_shape = 3/2
t_a_scale = 2
for i,cat in enumerate(initial_people):
    people[i] = Person(0,0,0,1,cat, action_time = gamma(t_a_shape,t_a_scale))
    
    


#create forecast object    
if state in ['VIC']:
    #XBstate = 'SA'
    Model = Forecast(current[state],
    state,start_date,people,
    alpha_i= 1, k =0.1,gam_list=gam,
    qs_list=qs_prior,qi_list=qi_prior,qa_list=qa_prior,
    qua_ai=1,qua_qi_factor=1,qua_qs_factor=1,
    forecast_R =forecast_type, R_I = R_I,forecast_date=forecast_date,
    cross_border_state=XBstate,cases_file_date=case_file_date,
    ps_list = ps_prior, test_campaign_date=test_campaign_date, 
    test_campaign_factor=test_campaign_factor,Reff_file_date=Reff_file_date
    )
elif state in ['NSW']:
    Model = Forecast(current[state],
    state,start_date,people,
    alpha_i= 1, k =0.1,gam_list=gam,
    qs_list=qs_prior,qi_list=qi_prior,qa_list=qa_prior,
    qua_ai=1,qua_qi_factor=1,qua_qs_factor=1,
    forecast_R =forecast_type, R_I = R_I,forecast_date=forecast_date,
    cross_border_state=None,cases_file_date=case_file_date,
    ps_list = ps_prior,Reff_file_date=Reff_file_date
    )
elif state in ['ACT','NT','SA','WA','QLD']:
    Model = Forecast(current[state],
    state,start_date,people,
    alpha_i= 0.1, k =0.1,gam_list=gam,
    qs_list=qs_prior,qi_list=qi_prior,qa_list=qa_prior,
    qua_ai=1,qua_qi_factor=1,qua_qs_factor=1,
    forecast_R =forecast_type, R_I = R_I,forecast_date=forecast_date,
    cross_border_state=None,cases_file_date=case_file_date,
    ps_list = ps_prior,Reff_file_date=Reff_file_date
    )
else:
    Model = Forecast(current[state],state,
    start_date,people,
    alpha_i= 0.5, k =0.1,gam_list=gam,
    qs_list=qs_prior,qi_list=qi_prior,qa_list=qa_prior,
    qua_ai=1,qua_qi_factor=1,qua_qs_factor=1, 
    forecast_R = forecast_type , R_I = R_I,forecast_date=forecast_date,
    cases_file_date=case_file_date,
    ps_list = ps_prior,Reff_file_date=Reff_file_date
    )



#Set up some required attributes for simulation
Model.end_time = time_end
Model.cross_border_seeds = np.zeros(shape=(time_end,1000),dtype=int)
Model.cross_border_state_cases = np.zeros_like(Model.cross_border_seeds)

Model.num_bad_sims = 0
Model.num_too_many = 0
        
#Read in files
Model.read_in_Reff()
Model.read_in_cases()

#simulate takes arguments days, sim number, and seed
## It will return:
## cases: a n_days by 3 array where each column represents 
##         Imported, Asymptomatic and Symptomatic cases, in that order.
##        Cases are indexed in time by rows by their date of infection.
## observed_cases: a n_days by 3 array, same as cases, but only observed cases, 
##        and are indexed in time by their date of symptom onset.

N=1
p_c=1.0
DAYS = 2
Model.simulate(time_end,1,N)


# Simulation study for delay time

t_a_shape = 3/2
t_a_scale = 2

n=1000

DAYS = 3
p_c = 1
pc_100_day_N3 = []
for N in range(0, n):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N)
    
    #v = list(x)[2]
    #v2 = v.values()
    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_100_day_N3.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)

print('Completed Days = -3 , p = 1.0')