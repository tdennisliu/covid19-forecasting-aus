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
initial_cases = [2,0]

local_detection = {
            'NSW':0.5, #0.8 #0.2 #0.556,#0.65,
            'QLD':0.9,#0.353,#0.493,#0.74,
            'SA':0.7,#0.597,#0.75,
            'TAS':0.4,#0.598,#0.48,
            'VIC':0.55,#0.558,#0.77,
            'WA':0.7,#0.409,#0.509,#0.66,
            'ACT':0.95,#0.557,#0.65,
            'NT':0.95,#0.555,#0.71
        }

a_local_detection = {
            'NSW':0.1,#0.556,#0.65,
            'QLD':0.05,#0.353,#0.493,#0.74,
            'SA':0.05,#0.597,#0.75,
            'TAS':0.05,#0.598,#0.48,
            'VIC':0.05,#0.558,#0.77,
            'WA':0.05,#0.409,#0.509,#0.66,
            'ACT':0.7,#0.557,#0.65,
            'NT':0.7,#0.555,#0.71
        }

qi_d = {
            'NSW':0.95,#0.758,
            'QLD':0.95,#0.801,
            'SA':0.95,#0.792,
            'TAS':0.95,#0.800,
            'VIC':0.95,#0.735,
            'WA':0.95,#0.792,
            'ACT':0.95,#0.771,
            'NT':0.95,#0.761
    }      
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
#p_c=1.0
#DAYS = 2
#Model.simulate(time_end,1,N, DAYS=DAYS, p_c =p_c)


# Simulation study for delay time

t_a_shape = 3/2
t_a_scale = 2

n=1000

DAYS = 3
p_c = 1
pc_100_day_N3 = []

for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_100_day_N3.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)

    #reset the simulation to the start
    Model.reset_to_start(people)

print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))

# Add

# DAYS = -3
# p = 0.9

#n=1000

DAYS = 3
p_c = 0.9
pc_90_day_N3 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)    

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_90_day_N3.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)
        
# print('Completed Days = -3 , p = 0.9')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))



# DAYS = -3
# p = 0.75

#n=1000

DAYS = 3
p_c = 0.75
pc_75_day_N3 = []
#for N in range(0, n):
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_75_day_N3.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)
        
# print('Completed Days =-3, p = 0.75')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))


# DAYS = -3
# p = 0.5

#n=1000

DAYS = 3
p_c = 0.5
pc_50_day_N3 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_50_day_N3.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)
        
# print('Completed Days = -3,  p = 0.5')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))


# DAYS = -2
# p = 1.0

#n=1000

DAYS = 2
p_c = 1
pc_100_day_N2 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_100_day_N2.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
   
    Model.reset_to_start(people)

# print('Completed Days = -2 , p = 1.0')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))



# DAYS = -2
# p = 0.9

#n=1000

DAYS = 2
p_c = 0.9
pc_90_day_N2 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_90_day_N2.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)

        
# print('Completed Days = -2 , p = 0.9')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))



# DAYS = -2
# p = 0.75

#n=1000

DAYS = 2
p_c = 0.75
pc_75_day_N2 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_75_day_N2.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)

        
# print('Completed Days =-2, p = 0.75')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))


# DAYS = -2
# p = 0.5

#n=1000

DAYS = 2
p_c = 0.5
pc_50_day_N2 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_50_day_N2.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
   
    Model.reset_to_start(people)

        
# print('Completed Days = -2,  p = 0.5')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))


# DAYS = -1
# p = 1.0

#n=1000

DAYS = 1
p_c = 1
pc_100_day_N1 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_100_day_N1.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)


# print('Completed Days = -1 , p = 1.0')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))


# DAYS = -1
# p = 0.9

#n=1000

DAYS = 1
p_c = 0.9
pc_90_day_N1 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_90_day_N1.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)

        
# print('Completed Days = -1 , p = 0.9')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))



# DAYS = -1
# p = 0.75

#n=1000

DAYS = 1
p_c = 0.75
pc_75_day_N1 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_75_day_N1.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)

        
# print('Completed Days =-1, p = 0.75')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))

# DAYS = -1
# p = 0.5

#n=1000

DAYS = 1
p_c = 0.5
pc_50_day_N1 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_50_day_N1.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)

        
# print('Completed Days = -1,  p = 0.5')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))

# DAYS = 0
# p = 1.0

#n=1000

DAYS = 0
p_c = 1
pc_100_day_0 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_100_day_0.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)


# print('Completed Days = 0 , p = 1.0')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))


# DAYS = 0
# p = 0.9

#n=1000

DAYS = 0
p_c = 0.9
pc_90_day_0 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_90_day_0.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
   
    Model.reset_to_start(people)

        
# print('Completed Days = 0 , p = 0.9')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))



# DAYS = 0
# p = 0.75

#n=1000

DAYS = 0
p_c = 0.75
pc_75_day_0 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_75_day_0.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)

        
# print('Completed Days =0, p = 0.75')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))

# DAYS = 0
# p = 0.5

#n=1000

DAYS = 0
p_c = 0.5
pc_50_day_0 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_50_day_0.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)

        
# print('Completed Days = 0,  p = 0.5')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))


# DAYS = +1
# p = 1.0

#n=1000

DAYS =-1
p_c = 1
pc_100_day_P1 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)
     
    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_100_day_P1.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)


# print('Completed Days = +1 , p = 1.0')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))


# DAYS = +1
# p = 0.9

#n=1000

DAYS = -1
p_c = 0.9
pc_90_day_P1 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_90_day_P1.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)

        
# print('Completed Days = +1 , p = 0.9')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))



# DAYS = +1
# p = 0.75

#n=1000

DAYS = -1
p_c = 0.75
pc_75_day_P1 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_75_day_P1.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)

        
# print('Completed Days =+1, p = 0.75')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))

# DAYS = +1
# p = 0.5

#n=1000

DAYS = -1
p_c = 0.5
pc_50_day_P1 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_50_day_P1.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)

        
# print('Completed Days = +1,  p = 0.5')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))

# DAYS = +2
# p = 1.0

#n=1000

DAYS =-2
p_c = 1
pc_100_day_P2 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_100_day_P2.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)


# print('Completed Days = +2 , p = 1.0')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))


# DAYS = +2
# p = 0.9

#n=1000

DAYS = -2
p_c = 0.9
pc_90_day_P2 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_90_day_P2.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)

        
# print('Completed Days = +2 , p = 0.9')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))



# DAYS = +1
# p = 0.75

#n=1000

DAYS = -2
p_c = 0.75
pc_75_day_P2 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_75_day_P2.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)

        
# print('Completed Days =+2, p = 0.75')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))

# DAYS = +2
# p = 0.5

#n=1000

DAYS = -2
p_c = 0.5
pc_50_day_P2 = []
for N in range(n, n+1000):
    cases_array, observed_cases_array, params = Model.simulate(time_end,1,N,
     DAYS=DAYS, p_c=p_c, t_a_shape=t_a_shape, t_a_scale= t_a_scale)

    Cases = params['Model_people']
    CasesAfter = params['cases_after']
    CasesTotal = Cases + CasesAfter

    pc_50_day_P2.append((CasesTotal))
    
    if N%100==0:
        print("sim number %i " % N)
        print("Timeline of Cases:\n", cases_array)
        print("Length of People (CasesTotal): %i " % CasesTotal)
    
    Model.reset_to_start(people)

        
# print('Completed Days = +2,  p = 0.5')
print('Completed Days = %i , p = %.2f' % (DAYS, p_c ))


# Export 

import pandas

#df = pandas.DataFrame(data={"CaseTot1Day": CaseTot1Day})
#df.to_csv("./CaseTot1Day.csv", sep=',',index=False)

# DAYS = -3

df = pandas.DataFrame(data={"pc_100_day_N3": pc_100_day_N3})
df.to_csv("./pc_100_day_N3_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_90_day_N3": pc_90_day_N3})
df.to_csv("./pc_90_day_N3_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_75_day_N3": pc_75_day_N3})
df.to_csv("./pc_75_day_N3_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_50_day_N3": pc_50_day_N3})
df.to_csv("./pc_50_day_N3_sc3_05_322.csv", sep=',',index=False)

# DAYS = -2

df = pandas.DataFrame(data={"pc_100_day_N2": pc_100_day_N2})
df.to_csv("./pc_100_day_N2_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_90_day_N2": pc_90_day_N2})
df.to_csv("./pc_90_day_N2_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_75_day_N2": pc_75_day_N2})
df.to_csv("./pc_75_day_N2_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_50_day_N2": pc_50_day_N2})
df.to_csv("./pc_50_day_N2_sc3_05_322.csv", sep=',',index=False)

# DAYS = -1

df = pandas.DataFrame(data={"pc_100_day_N1": pc_100_day_N1})
df.to_csv("./pc_100_day_N1_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_90_day_N1": pc_90_day_N1})
df.to_csv("./pc_90_day_N1_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_75_day_N1": pc_75_day_N1})
df.to_csv("./pc_75_day_N1_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_50_day_N1": pc_50_day_N1})
df.to_csv("./pc_50_day_N1_sc3_05_322.csv", sep=',',index=False)


DAYS = 0

df = pandas.DataFrame(data={"pc_100_day_0": pc_100_day_0})
df.to_csv("./pc_100_day_0_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_90_day_0": pc_90_day_0})
df.to_csv("./pc_90_day_0_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_75_day_0": pc_75_day_0})
df.to_csv("./pc_75_day_0_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_50_day_0": pc_50_day_0})
df.to_csv("./pc_50_day_0_sc3_05_322.csv", sep=',',index=False)



DAYS = 1

df = pandas.DataFrame(data={"pc_100_day_P1": pc_100_day_P1})
df.to_csv("./pc_100_day_P1_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_90_day_P1": pc_90_day_P1})
df.to_csv("./pc_90_day_P1_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_75_day_P1": pc_75_day_P1})
df.to_csv("./pc_75_day_P1_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_50_day_P1": pc_50_day_P1})
df.to_csv("./pc_50_day_P1_sc3_05_322.csv", sep=',',index=False)


DAYS = 2

df = pandas.DataFrame(data={"pc_100_day_P2": pc_100_day_P2})
df.to_csv("./pc_100_day_P2_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_90_day_P2": pc_90_day_P2})
df.to_csv("./pc_90_day_P2_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_75_day_P2": pc_75_day_P2})
df.to_csv("./pc_75_day_P2_sc3_05_322.csv", sep=',',index=False)

df = pandas.DataFrame(data={"pc_50_day_P2": pc_50_day_P2})
df.to_csv("./pc_50_day_P2_sc3_05_322.csv", sep=',',index=False)


