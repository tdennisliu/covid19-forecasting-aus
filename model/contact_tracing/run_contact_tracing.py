import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import beta, gamma

from ct_sim_class import *
import multiprocessing as mp

from sys import argv
import os

def worker(arg):
    """
    worker function that allows multiprocessing to use the class method
    """
    obj, methname = arg[:2]
    return getattr(obj,methname)(*arg[2:-1],**arg[-1])


if __name__ == '__main__':

    #set up parallel processing
    n_cpus = 10
    pool = mp.Pool(n_cpus)


    DAYS_list = (-3,-2,-1,0,1,2)
    DAYS = DAYS_list[int(argv[1])] #select right day from list

    p_c_list = (0.5,0.75,0.9,1)
    p_c_list = p_c_list[::-1]

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
    initial_cases = [10,2]

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
    forecast_date = '2020-03-01'
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
    t_a_shape = 2/1
    t_a_scale = 1
    for i,cat in enumerate(initial_people):
        people[i] = Person(0,0,0,1,cat, action_time = 1 + gamma(t_a_shape,t_a_scale))
        
    #create forecast object    
    if state in ['VIC']:
        #XBstate = 'SA'
        Model = Forecast(current[state],
        state,start_date,people,
        alpha_i= 1, k =0.1,gam_list=gam,
        qs_list=qs_prior,qi_list=qi_prior,qa_list=qa_prior,
        qua_ai=1,qua_qi_factor=1,qua_qs_factor=1,
        forecast_R =forecast_type, R_I = R_I,forecast_date=forecast_date,
        cross_border_state=None,cases_file_date=case_file_date,
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
    Model.cross_border_seeds = np.zeros(shape=(time_end,10000),dtype=int)
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



    # Simulation study for delay time

    t_a_shape = 2/1
    t_a_scale = 1

    n=1000
    pc_100_dict = {}

    pc_dict = {}

    prop_cases_prevented_by_sim = {}
    prop_cases_prevented_by_pc = {}

    #generation_times_by_sim ={}
    generation_times_by_pc = {}

    #set up dataframe

    df = pd.DataFrame(
        columns=[
            'sim','pc','cases',
            'prop_cases_prevented_mean',
            'prop_cases_prevented_25',
            'prop_cases_prevented_75',
            'actual_gen_times_mean',
            'actual_gen_times_25',
            'actual_gen_times_75',
            ])
    for p_c in p_c_list:
        pc_dict[p_c] = []
        prop_cases_prevented_by_pc[p_c] =[]
        generation_times_by_pc[p_c] =[]

    for p_c in p_c_list:
        kwargs = {
            "DAYS":DAYS,
            "p_c":p_c,
            "t_a_shape":t_a_shape,
            "t_a_scale":t_a_scale,
        }
        prop_cases_prevented = []
        actual_gen_times = []
        #lose the ordering with parallel processing unless we record to dict?
        for cases_array, observed_cases_array, params in pool.imap_unordered(worker,
            [(Model,'simulate_then_reset',time_end, N, N, kwargs) 
            for N in range(n)]
            ):
            num_sim = params['num_of_sim']
            Cases = params['Model_people']
            #CasesAfter = params['cases_after']

            prop_cases_prevented.extend(params['secondary_cases'])
            actual_gen_times.extend(params['generation_times'])

            CasesTotal = Cases #+ CasesAfter
            
            pc_100_dict[num_sim] = CasesTotal
            
            prop_cases_prevented_by_sim[num_sim] = np.mean(
                params['secondary_cases'])


        
        #record results back in order into orginal list
        for N in range(n):
            pc_dict[p_c].append(pc_100_dict[N])
            prop_cases_prevented_by_pc[p_c].append(prop_cases_prevented)
            generation_times_by_pc[p_c].append(actual_gen_times)
        temp = pd.DataFrame()
        temp.index.name = 'sim'
        temp['cases'] = pc_dict[p_c]

        temp['prop_cases_prevented_mean'] = np.mean(prop_cases_prevented)
        temp['prop_cases_prevented_25'] = np.quantile(prop_cases_prevented, 0.25)
        temp['prop_cases_prevented_75'] = np.quantile(prop_cases_prevented, 0.75)
        temp['actual_gen_times_mean'] = np.mean(actual_gen_times)
        temp['actual_gen_times_25'] = np.quantile(actual_gen_times, 0.25)
        temp['actual_gen_times_75'] = np.quantile(actual_gen_times, 0.75)

        temp['sim'] = temp.index
        temp['pc'] = p_c
        temp['DAYS'] = DAYS
        df = df.append(temp, ignore_index=True)
        
        print("Finished p_c = %.2f, DAYS = %i" % (p_c, DAYS))
        print(
            "0.05,0.25,0.5,0.75,0.95 quantiles of cases \n {}".format(
                np.quantile(pc_dict[p_c], (
                0.05,0.25,0.5,0.75,0.95)
                )
            )
        )
        os.makedirs("./model/contact_tracing/figs/",exist_ok=True)

        plot_name="pc_"+str(p_c) +"_DAYS"+str(DAYS)
        #Plot actual generation time against original generation time
        fig,ax = plt.subplots(figsize=(12,9))

        Model.t_a_shape = t_a_shape
        Model.t_a_scale = t_a_scale
        Model.generate_times()
        ax.hist(actual_gen_times, label='Actual',density=True,bins=20)
        ax.hist(Model.inf_times, label='Orginal', density=True,alpha=0.4,bins=20)
        plt.legend()
        plt.savefig("./model/contact_tracing/figs/"+plot_name+"actual_gen_dist.png",dpi=300)

        #Plot actual generation time against original generation time
        fig,ax = plt.subplots(figsize=(12,9))

        ax.hist(prop_cases_prevented, label='Actual',density=True,bins=20)

        plt.savefig("./model/contact_tracing/figs/"+plot_name+"actual_prop_cases_dist.png",dpi=300)

        #record and print to csv
        file_name = "allpc_days_"+str(DAYS)
        df.to_csv("./model/contact_tracing/"+file_name+"_sc3DL.csv", sep=',',index=False)
    pool.close()
    pool.join()
    print("Finished DAYS %i" % DAYS)

