from sim_class import *
import pandas as pd
from sys import argv
from numpy.random import beta, gamma

#from joblib import Parallel, delayed
import multiprocessing as mp

def worker(arg):
    obj, methname = arg[:2]
    return getattr(obj,methname)(*arg[2:])

n_sims=int(argv[1]) #number of sims
time_end = int(argv[2]) 

states = ['NSW','QLD','SA','TAS','VIC','WA','ACT','NT']
#states = ['QLD','VIC']
start_date = '2020-03-01'
case_file_date = ['29Jun','0900']
R_I='R_I'
abc =True
if R_I is not None:
    print("Using model output for R_L and R_I")




local_detection = {
            'NSW':0.556,#0.556,#0.65,
            'QLD':0.493,#0.353,#0.493,#0.74,
            'SA':0.75,#0.597,#0.75,
            'TAS':0.48,#0.598,#0.48,
            'VIC':0.558,#0.558,#0.77,
            'WA':0.8,#0.409,#0.509,#0.66,
            'ACT':0.65,#0.557,#0.65,
            'NT':0.81,#0.555,#0.71
        }

qi_d = {
            'NSW':0.9,#0.758,
            'QLD':0.95,#0.801,
            'SA':0.95,#0.792,
            'TAS':0.95,#0.800,
            'VIC':0.9,#0.735,
            'WA':0.95,#0.792,
            'ACT':0.95,#0.771,
            'NT':0.95,#0.761
    }

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
if len(argv)>=3:
    forecast_type = argv[3]
else:
    forecast_type = None
forecast_dict = {}
for state in states:
    initial_people = ['I']*current[state][0] + \
            ['A']*current[state][1] + \
            ['S']*current[state][2]
    people = {}
    if abc:
        #qs_prior = beta(10,10,size=10000)
        #qi_prior = beta(17, 3, size=10000)
        qi_prior = [qi_d[state]]
        qs_prior = [local_detection[state]]
        gam =np.maximum(0.1,np.minimum(2,gamma(4,0.25, size=1000)))
        ps_prior = beta(10,10,size=1000)

    else:
        qi_prior = [qi_d[state]]
        qs_prior = [local_detection[state]]
        gam =[1/3]
        ps_prior = 0.7
        ps_prior= [ps_prior]

    for i,cat in enumerate(initial_people):
        people[i] = Person(0,0,0,0,cat)
    
    if state in ['VIC']:
        forecast_dict[state] = Forecast(current[state],
        state,start_date,people,
        alpha_i= 1, k =0.1,gam_list=gam,
        qs_list=qs_prior,qi_list=qi_prior,
        qua_ai=1,qua_qi_factor=1,qua_qs_factor=1,
        forecast_R =forecast_type, R_I = R_I,forecast_date='2020-06-29',
        cross_border_state=None,cases_file_date=case_file_date,
        ps_list = ps_prior,
        )
    elif state in ['NSW']:
        forecast_dict[state] = Forecast(current[state],
        state,start_date,people,
        alpha_i= 1, k =0.1,gam_list=gam,
        qs_list=qs_prior,qi_list=qi_prior,
        qua_ai=1,qua_qi_factor=1,qua_qs_factor=1,
        forecast_R =forecast_type, R_I = R_I,forecast_date='2020-06-29',
        cross_border_state=None,cases_file_date=case_file_date,
        ps_list = ps_prior,
        )
    elif state in ['ACT','NT']:
        forecast_dict[state] = Forecast(current[state],
        state,start_date,people,
        alpha_i= 0.5, k =0.1,gam_list=gam,
        qs_list=qs_prior,qi_list=qi_prior,
        qua_ai=1,qua_qi_factor=1,qua_qs_factor=1,
        forecast_R =forecast_type, R_I = R_I,forecast_date='2020-06-29',
        cross_border_state=None,cases_file_date=case_file_date,
        ps_list = ps_prior,
        )
    else:
        forecast_dict[state] = Forecast(current[state],state,
        start_date,people,
        alpha_i= 0.5, k =0.1,gam_list=gam,
        qs_list=qs_prior,qi_list=qi_prior,
        qua_ai=1,qua_qi_factor=1,qua_qs_factor=1, 
        forecast_R = forecast_type , R_I = R_I,forecast_date='2020-06-29',
        cases_file_date=case_file_date,
        ps_list = ps_prior,
        )



if __name__ =="__main__":

    for key,item in forecast_dict.items():
        item.read_in_Reff()
    pool = mp.Pool(8)
    l_df_results =pool.map(worker,
        [(obj,'simulate_many',time_end, n_sims) 
        for key,obj in forecast_dict.items()]
    )
    pool.close()
    pool.join()
    #record quantiles in separate file
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
                            periods=time_end #num of days
                        )
    vars_l = ['symp_inci_obs','imports_inci_obs','asymp_inci_obs','symp_inci','asymp_inci','imports_inci','total_inci','total_inci_obs']
    for var in vars_l:
        for state,df in l_df_results:
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
    plots.to_parquet('./results/quantiles'+forecast_type+start_date+"sim_"+str(n_sims)+"days_"+str(time_end)+".parquet")
        




       