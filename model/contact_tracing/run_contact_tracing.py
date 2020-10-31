import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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


    #DAYS_list = (-3,-2,-1,0,1,2)
    DAYS = int(argv[1]) #select right day from list

 
    p_c_list = (0.5,0.75,0.9,1)
    p_c_list = p_c_list[::-1]

    ##########
    #PARAMETERS TO CHANGE
    #########
    
    #time to isolation gamma parameters
    t_a_offset = 0 #number of days minimum to isolation
    t_a_shape = 1
    t_a_scale = 1

    t_p_shape = 1.03108
    t_p_scale = 1/0.415
    t_p_offset = 0  
        
    t_t_shape = 0
    t_t_scale = 7.905380
    t_t_offset =0

    t_n_shape = 0.4533819
    t_n_scale = 1/1.8200
    t_n_offset = 0

    generations_traced = int(argv[2])
    test_capacity = 2000000
    trace_capacity = 200000

    sim_undetected = False
    #number of iterations
    if len(argv)>4:
        n = int(argv[4])
    else:
        n = 10000
    print("Number of simulations: %i" % n )
    time_end = 30
    state = 'NSW'
    case_file_date = None #'24Jul'
    #Reff_file_date = '2020-07-20'
    Reff_file_date = '2020-08-06'
    #Number of initial, detected asymptomatic and symptomatic cases respectively
    initial_cases = [1,5]

    local_detection = {
                'NSW':float(argv[3]), #0.8 #0.2 #0.556,#0.65,
                'QLD':0.9,#0.353,#0.493,#0.74,
                'SA':0.7,#0.597,#0.75,
                'TAS':0.4,#0.598,#0.48,
                'VIC':0.55,#0.558,#0.77,
                'WA':0.7,#0.409,#0.509,#0.66,
                'ACT':0.95,#0.557,#0.65,
                'NT':0.95,#0.555,#0.71
            }

    a_local_detection = {
                'NSW':local_detection['NSW']/5,#0.556,#0.65,
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
    forecast_type = 'R_L0'
    start_date = '2020-03-01'
    test_campaign_date = '2020-06-25'
    test_campaign_factor = 1.25
    R_I='R_I'
    abc =False
    forecast_date = '2020-03-01'
    ##############################



    #print("Simulating state " +state)
    print("gens traced: %i " % generations_traced)
    print("detect: %.2f" % local_detection['NSW'])


    current ={state: [0,initial_cases[0],initial_cases[1]]}
    forecast_dict = {}

    initial_people = ['I']*current[state][0] + \
            ['A']*current[state][1] + \
            ['S']*current[state][2]
    people = {}

    if abc:
        #assign a distribution to parameters
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

    for i,cat in enumerate(initial_people):
        #people were detected today 
        action_time = t_a_offset + gamma(t_a_shape,t_a_scale)
        notify_PHU_time = 0
        test_time  = -1*( t_n_offset + gamma(t_n_shape, t_n_scale))
        present_time = test_time - (t_t_offset + gamma(t_t_shape, t_t_scale))
        symp_time = present_time - (t_p_offset + gamma(t_p_shape, t_p_scale))

        people[i] = Person(-1,0,symp_time,1,cat, 
        action_time = action_time,
        present_time=  present_time,
        test_time = test_time,
        notify_PHU_time= notify_PHU_time,
        )
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

    else:
        Model = Forecast(current[state],state,
        start_date,people,
        alpha_i= 1, k =0.1,gam_list=gam,
        qs_list=qs_prior,qi_list=qi_prior,qa_list=qa_prior,
        qua_ai=1,qua_qi_factor=1,qua_qs_factor=1, 
        forecast_R = forecast_type , R_I = R_I,forecast_date=forecast_date,
        cases_file_date=case_file_date,
        ps_list = ps_prior,Reff_file_date=Reff_file_date
        )



    #Set up some required attributes for simulation
    Model.end_time = time_end
    Model.cross_border_seeds = np.zeros(shape=(time_end,n),dtype=int)
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



    pc_100_dict = {}

    pc_dict = {}

    secondary_cases_by_sim = {}
    secondary_cases_by_pc = {}

    #generation_times_by_sim ={}
    generation_times_by_pc = {}
    die_out_by_pc = {}
    #set up dataframe

    df = pd.DataFrame(
        columns=[
            'sim','pc','cases',
            'secondary_cases_mean',
            'secondary_cases_25',
            'secondary_cases_75',
            'actual_gen_times_mean',
            #'actual_gen_times_25',
            #'actual_gen_times_75',
            ])
    for p_c in p_c_list:
        pc_dict[p_c] = []
        secondary_cases_by_pc[p_c] =[]
        generation_times_by_pc[p_c] =[]
        die_out_by_pc[p_c] = []

    for p_c in p_c_list:
        kwargs = {
            "DAYS":DAYS,
            "p_c":p_c,
            "t_a_shape":t_a_shape,
            "t_a_scale":t_a_scale,
            "t_a_offset":t_a_offset,            
            "t_p_shape":t_p_shape,
            "t_p_scale":t_p_scale,
            "t_p_offset":t_p_offset,            
            "t_t_shape":t_t_shape,
            "t_t_scale":t_t_scale,
            "t_t_offset":t_t_offset,
            "t_n_shape":t_n_shape,
            "t_n_scale":t_n_scale,
            "t_n_offset":t_n_offset,
            "sim_undetected":sim_undetected,
            "generations_traced": generations_traced,
            "test_capacity": test_capacity,
            "trace_capacity": trace_capacity,
        }
        secondary_cases = []
        actual_gen_times = []
        die_out = {}
        #lose the ordering with parallel processing unless we record to dict?
        for cases_array, observed_cases_array, params in pool.imap_unordered(worker,
            [(Model,'simulate_then_reset',time_end, N, N, kwargs) 
            for N in range(n)]
            ):
            num_sim = params['num_of_sim']
            Cases = params['Model_people']
            #CasesAfter = params['cases_after']

            secondary_cases.extend(params['secondary_cases'])
            actual_gen_times.extend(params['generation_times'])

            CasesTotal = Cases #+ CasesAfter
            

            pc_100_dict[num_sim] = CasesTotal
            
            secondary_cases_by_sim[num_sim] = np.mean(
                params['secondary_cases'])

            die_out[num_sim]=params['cases_after']==0
        
        #record results back in order into orginal list
        for N in range(n):
            pc_dict[p_c].append(pc_100_dict[N])
            secondary_cases_by_pc[p_c].append(secondary_cases)
            generation_times_by_pc[p_c].append(actual_gen_times)
            die_out_by_pc[p_c].append(die_out[N])

        temp = pd.DataFrame()
        temp.index.name = 'sim'
        temp['cases'] = pc_dict[p_c]

        temp['secondary_cases_mean'] = np.mean(secondary_cases)
        temp['secondary_cases_25'] = np.quantile(secondary_cases, 0.25)
        temp['secondary_cases_75'] = np.quantile(secondary_cases, 0.75)
        temp['actual_gen_times_mean'] = np.mean(actual_gen_times)
        #temp['actual_gen_times_25'] = np.quantile(actual_gen_times, 0.25)
        #temp['actual_gen_times_75'] = np.quantile(actual_gen_times, 0.75)
        temp['avg_daily_growth_rate'] = (temp.cases/(sum(initial_cases)))**(1/time_end)
        temp['die_out'] = die_out_by_pc[p_c]

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
        os.makedirs("./model/contact_tracing/figs/gen_interval/",exist_ok=True)
        os.makedirs("./model/contact_tracing/figs/secondary_cases/",exist_ok=True)
        os.makedirs("./model/contact_tracing/results/",exist_ok=True)

        plot_name="pc_"+str(p_c) +"_DAYS"+str(DAYS)
        #Plot actual generation time against original generation time
        fig,ax = plt.subplots(figsize=(12,9))

        Model.t_p_shape = t_p_shape
        Model.t_p_scale = t_p_scale
        Model.t_p_offset = t_p_offset        
        
        Model.t_t_shape = t_t_shape
        Model.t_t_scale = t_t_scale
        Model.t_t_offset = t_t_offset

        Model.t_n_shape = t_n_shape
        Model.t_n_scale = t_n_scale
        Model.t_n_offset = t_n_offset

        Model.t_a_shape = t_a_shape
        Model.t_a_scale = t_a_scale
        Model.t_a_offset = t_a_offset


        Model.generate_times()
        ax.hist(actual_gen_times,range=(0,20), label='Actual',density=True,bins=20)
        ax.hist(Model.inf_times,range=(0,20), label='Orginal', density=True,alpha=0.4,bins=20)
        plt.legend()
        plt.savefig("./model/contact_tracing/figs/gen_interval/"+str(n)+plot_name+"actual_gen_dist.png",dpi=300)

        #Plot actual generation time against original generation time
        fig,ax = plt.subplots(figsize=(12,9))
        bins = np.arange(0, 10 + 1.5) - 0.5
        ax.hist(secondary_cases,range=(0,10),bins=bins,label='Actual')
        ax.set_xticks(bins + 0.5)
        plt.savefig("./model/contact_tracing/figs/secondary_cases/"+str(n)+plot_name+"actual_prop_cases_dist.png",dpi=300)

        #record and print to csv
        file_name = "allpc_days_"+str(DAYS)+"init_"+str(initial_cases[0])
        df.to_csv("./model/contact_tracing/results/"+str(n)+file_name+"_detect"+argv[3]+"_gens"+str(generations_traced)+".csv", sep=',',index=False)
    pool.close()
    pool.join()
    print("Finished DAYS %i" % DAYS)

