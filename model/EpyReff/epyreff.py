import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
from datetime import datetime as dt
from datetime import timedelta
import glob
    
#from Reff_functions import *
#from Reff_constants import *
from scipy.stats import gamma

#Code taken from read_in_cases from Reff_functions. Preprocessing was not helpful for this situation.
def read_cases_lambda(case_file_date):
    """
    Read in NNDSS data
    """
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path,"../../data/COVID-19 UoM "+case_file_date+"*.xlsx")
    print("Reading file from:")

    for file in glob.glob(path):
        print(file)
        df_NNDSS = pd.read_excel(file,
                           parse_dates=['SPECIMEN_DATE','NOTIFICATION_DATE','NOTIFICATION_RECEIVE_DATE','TRUE_ONSET_DATE'],
                           dtype= {'PLACE_OF_ACQUISITION':str})
        df_NNDSS.PLACE_OF_ACQUISITION.fillna('00038888',inplace=True) #Fill blanks with simply unknown

       # df_NNDSS['date_inferred'] = df_NNDSS.TRUE_ONSET_DATE
      #  df_NNDSS.loc[df_NNDSS.TRUE_ONSET_DATE.isna(),'date_inferred'] = df_NNDSS.loc[df_NNDSS.TRUE_ONSET_DATE.isna()].NOTIFICATION_DATE - timedelta(days=5)
      #  df_NNDSS.loc[df_NNDSS.date_inferred.isna(),'date_inferred'] = df_NNDSS.loc[df_NNDSS.date_inferred.isna()].NOTIFICATION_RECEIVE_DATE - timedelta(days=6)    
    df_NNDSS['imported'] = df_NNDSS.PLACE_OF_ACQUISITION.apply(lambda x: 1 if x[:4]!='1101' else 0)
    df_NNDSS['local'] = 1 - df_NNDSS.imported

    df_interim = df_NNDSS[['NOTIFICATION_RECEIVE_DATE','STATE','imported','local']] 
    #df_interim = df_interim[~np.isnat(df_interim.NOTIFICATION_DATE)] #Get rid of non-existent dates.
    # Importantly, imported and local are indicator variables in df_interim.

    #df_state = df_NNDSS[['NOTIFICATION_DATE','STATE','imported','local']].groupby(['STATE','NOTIFICATION_DATE']).sum()
    return(df_interim)

def tidy_cases_lambda(interim_data, remove_territories=True):
    #Remove non-existent notification dates
    interim_data = interim_data[~np.isnat(interim_data.NOTIFICATION_RECEIVE_DATE)]
    
    #Filter out territories
    if(remove_territories):
        df_linel = interim_data[(interim_data['STATE']!='NT') & (interim_data['STATE']!='ACT')]

    #Melt down so that imported and local are no longer columns. Allows multiple draws for infection date.
    #i.e. create linelist data
    df_linel = df_linel.melt(id_vars = ['NOTIFICATION_RECEIVE_DATE','STATE'], var_name = 'SOURCE',value_name='n_cases')

    #Reset index or the joining doesn't work
    df_linel = df_linel[df_linel.n_cases!=0]
    df_linel = df_linel.reset_index(drop=True)
    return(df_linel)

##gamma draws take arguments (shape, scale)
def draw_inf_dates(df_linelist, shape_rd=2.77, scale_rd=3.17, offset_rd=0,
                    shape_inc=5.807, scale_inc=0.948, offset_inc=1,nreplicates=1):

    notification_dates = df_linelist['NOTIFICATION_RECEIVE_DATE']
    nsamples = notification_dates.shape[0]

    #    DEFINE DELAY DISTRIBUTION
    #     mean_rd = 5.47
    #     sd_rd = 4.04
    #scale_rd = shape_rd/(scale_rd)**2
    #shape_rd = shape_rd/scale_rd

    # DEFINE INCUBATION PERIOD DISTRIBUTION
    # Taken from Lauer et al 2020
    #     mean_inc = 5.5 days
    #     sd_inc = 1.52
    #scale_inc = (scale_inc)**2/shape_inc #scale**2 = var / shape
    #shape_inc =(scale_inc)**2/scale_inc**2 

    #Draw from distributions - these are long vectors
    inc_period = offset_inc+np.random.gamma(shape_inc, scale_inc, size = (nsamples*nreplicates))
    rep_delay = offset_rd+np.random.gamma(shape_rd, scale_rd, size = (nsamples*nreplicates))

    #infection date is id_nd_diff days before notification date. This is also a long vector.
    id_nd_diff = inc_period + rep_delay

    #Minutes aren't included in df. Take the ceiling because the day runs from 0000 to 2359. This can still be a long vector.
    whole_day_diff = np.ceil(id_nd_diff) 
    time_day_diffmat = whole_day_diff.astype('timedelta64[D]').reshape((nsamples, nreplicates))

    #Vector must be coerced into a nsamples by nreplicates array. Then each column must be subtracted from notification_dates. 
    #Subtract days off of notification dates.

    notification_mat = np.tile(notification_dates, (nreplicates,1)).T #notification_dates is repeated as a column nreplicates times.

    infection_dates = notification_mat - time_day_diffmat

    #Make infection dates into a dataframe
    datecolnames = [*map(str,range(nreplicates))]
    infdates_df = pd.DataFrame(infection_dates,columns = datecolnames)
    
    #Uncomment this if theres errors
    #print([df_linelist.shape, infdates_df.shape])
    
    
    #Combine infection dates and original dataframe
    df_inf = pd.concat([df_linelist, infdates_df], axis=1, verify_integrity=True)

    return(df_inf)

def index_by_infection_date(infections_wide):
    datecolnames = [*infections_wide.columns[4:]]
    df_combined = infections_wide[['STATE','SOURCE',datecolnames[0],'n_cases']].groupby(['STATE', datecolnames[0],'SOURCE']).sum()

    #For each column (cn=column number): concatenate each sample as a column.
    for cn in range(1,len(datecolnames)):
        df_addin = infections_wide[['STATE','SOURCE',datecolnames[cn],'n_cases']].groupby(['STATE', datecolnames[cn],'SOURCE']).sum()
        df_combined = pd.concat([df_combined,df_addin], axis=1, ignore_index = True)

    #NaNs are inserted for missing values when concatenating. If it's missing, there were zero infections
    df_combined[np.isnan(df_combined)]=0
    #Rename the index.
    df_combined.index.set_names(["STATE","INFECTION_DATE","SOURCE"], inplace=True)

    #return(df_combined)

    ##INCLUDE ALL DAYS WITH ZERO INFECTIONS IN THE INDEX AS WELL.

    # Reindex to include days with zero total infections.
    local_infs = df_combined.xs('local',level='SOURCE')
    imported_infs = df_combined.xs('imported',level='SOURCE')
    statelist = [*df_combined.index.get_level_values('STATE').unique()]

    #Should all states have the same start date? Current code starts from the first case in each state.
    #For the same start date:
    local_statedict = dict(zip(statelist, np.repeat(None, len(statelist))))
    imported_statedict = dict(zip(statelist, np.repeat(None, len(statelist))))

    #Determine start date as the first infection date for all. 
    #start_date = np.datetime64("2020-02-01")
    start_date = df_combined.index.get_level_values('INFECTION_DATE').min()

    #Determine end dates as the last infected date by state.
    index_only = df_combined.index.to_frame()
    index_only = index_only.reset_index(drop=True)
    maxdates = index_only['INFECTION_DATE'].max()

    for aus_state in statelist:
        state_data = local_infs.xs(aus_state, level='STATE')
        #start_date = state_data.index.min()

        #dftest.index=dftest.reindex(alldates, fill_value=0)

        alldates = pd.date_range(start_date, maxdates) #All days from start_date to the last infection day.
        local_statedict[aus_state] = state_data.reindex(alldates, fill_value=0)

    for aus_state in statelist:
        state_data = imported_infs.xs(aus_state, level='STATE')
        alldates = pd.date_range(start_date, maxdates)
        imported_statedict[aus_state] = state_data.reindex(alldates, fill_value=0)

    #Convert dictionaries to data frames
    df_local_inc_zeros = pd.concat(local_statedict)
    df_local_inc_zeros['SOURCE']='local'
    df_imp_inc_zeros = pd.concat(imported_statedict)
    df_imp_inc_zeros['SOURCE']='imported'

    #Merge dataframes and reindex. 
    df_inc_zeros = pd.concat([df_local_inc_zeros, df_imp_inc_zeros])

    df_inc_zeros = df_inc_zeros.reset_index()
    df_inc_zeros= df_inc_zeros.groupby(['level_0',"level_1","SOURCE"]).sum()
    df_inc_zeros.index = df_inc_zeros.index.rename(['STATE','INFECTION_DATE',"SOURCE"])

    return(df_inc_zeros)


def generate_lambda(infection_dates, shape_gen=3.64/3.07, scale_gen=3.07, 
                    trunc_days=21,shift=0, offset=1):
    """
    Given array of infection_dates (N_dates by N_samples), where values are possible
    number of cases infected on this day, generate the force of infection Lambda_t,
    a N_dates-tau by N_samples array.
    Default generation interval parameters taken from Ganyani et al 2020.
    """
    from scipy.stats import gamma
 
    #scale_gen = mean_gen/(sd_gen)**2
    #shape_gen = mean_gen/scale_gen
    

    xmids = [x+shift for x in range(trunc_days+1)] #Find midpoints for discretisation
    gamma_vals = gamma.pdf(xmids, a=shape_gen, scale=scale_gen) #double check parameterisation of scipy
    #renormalise the pdf
    disc_gamma = gamma_vals/sum(gamma_vals)
    
    
    ws = disc_gamma[:trunc_days]
    #offset
    ws[offset:] = disc_gamma[:trunc_days-offset]
    ws[:offset] = 0
    lambda_t = np.zeros(shape=(infection_dates.shape[0]-trunc_days+1, infection_dates.shape[1]))
    for n in range(infection_dates.shape[1]):
        lambda_t[:,n] = np.convolve(infection_dates[:,n], ws, mode='valid')
    return lambda_t
    
    
def lambda_all_states(df_infection, **kwargs):
    """
    Use geenrate lambda on every state
    """
    statelist = [*df_infection.index.get_level_values('STATE').unique()]
    
    lambda_dict ={}
    for state in statelist:
        df_total_infections = df_infection.groupby(['STATE','INFECTION_DATE']).agg(sum)
        lambda_dict[state] = generate_lambda(
             df_total_infections.loc[state].values,
            **kwargs
            )
    
    return lambda_dict

def Reff_from_case(cases_by_infection, lamb, prior_a=1, prior_b=5, tau=7, samples=1000):
    """
    Using Cori at al. 2013, given case incidence by date of infection, and the force
    of infection \Lambda_t on day t, estimate the effective reproduction number at time
    t with smoothing parameter \tau.
    
    cases_by_infection: A T by N array, for T days and N samples 
    lamb : A T by N array, for T days and N samples
    """
    csum_incidence = np.cumsum(cases_by_infection, axis = 0)
    #remove first few incidences to align with size of lambda
    # Generation interval length 20
    csum_incidence = csum_incidence[20:,:]
    csum_lambda = np.cumsum(lamb, axis =0)
    
    roll_sum_incidence = csum_incidence[tau:, :] - csum_incidence[:-tau, :]
    roll_sum_lambda = csum_lambda[tau:,:] - csum_lambda[:-tau,:]
    a = prior_a + roll_sum_incidence
    b = 1/(1/prior_b + roll_sum_lambda)
    
    R = np.random.gamma(a,b) #shape, scale
    
    #Need to empty R when there is too few cases...
    
    #Use array inputs to output to same size
    #inputs are T-tau by N, output will be T-tau by N
    # 
    return a,b, R

def generate_summary(samples, dates_by='rows'):
    """
    Given an array of samples (T by N) where rows index the dates, 
    generate summary statistics and quantiles
    """
    
    if dates_by=='rows':
        #quantiles of the columns
        ax = 1
    else:
        #quantiles of the rows
        ax = 0
    mean = np.mean(samples, axis = ax)
    bottom, lower, median, upper, top = np.quantile(samples,
                                                     (0.05, 0.25, 0.5, 0.75, 0.95),
                                                     axis =ax)
    std = np.std(samples, axis = ax)
    output = {
        'mean':mean,
        'std':std,
        'bottom':bottom,
        'lower':lower,
        'median':median,
        'upper':upper,
        'top': top,
        
    }
    return output

def plot_Reff(Reff:dict, dates=None, ax_arg=None, truncate=None, **kwargs):
    """
    Given summary statistics of Reff as a dictionary, plot the distribution over time
    """
    import matplotlib.pyplot as plt
    
    plt.style.use('seaborn-poster')
    from datetime import datetime as dt
    
    if ax_arg is None:
        fig, ax = plt.subplots(figsize=(12,9))
    else:
        fig, ax = ax_arg
        
    color_cycle = ax._get_lines.prop_cycler
    curr_color = next(color_cycle)['color']
    if dates is None:
        dates = range(len(Reff['mean']))

    if truncate is None:    
        ax.plot(dates, Reff['mean'], color= curr_color, **kwargs)
        
        ax.fill_between(dates, Reff['lower'],Reff['upper'], alpha=0.4, color = curr_color)
        ax.fill_between(dates, Reff['bottom'],Reff['top'], alpha=0.4, color= curr_color)
    else:
        ax.plot(dates[truncate[0]:truncate[1]], Reff['mean'][truncate[0]:truncate[1]], color= curr_color, **kwargs)
        
        ax.fill_between(dates[truncate[0]:truncate[1]], Reff['lower'][truncate[0]:truncate[1]],
            Reff['upper'][truncate[0]:truncate[1]],
            alpha=0.4, color = curr_color)
        ax.fill_between(dates[truncate[0]:truncate[1]], Reff['bottom'][truncate[0]:truncate[1]],
            Reff['top'][truncate[0]:truncate[1]],
            alpha=0.4, color= curr_color)
        #plt.legend()

        
       #grid line at R_eff =1
    ax.set_yticks([1],minor=True,)
    ax.set_yticks([0,2,3],minor=False)
    ax.set_yticklabels([0,2,3],minor=False)
    ax.yaxis.grid(which='minor',linestyle='--',color='black',linewidth=2)
    ax.tick_params(axis='x', rotation = 45)
    
    return fig, ax

def plot_all_states(R_summ_states,df_interim, dates,
    start='2020-03-01',end='2020-08-01',save=True, date =None, tau = 7, 
    nowcast_truncation=-10):
    """
    Plot results over time for all jurisdictions.

    dates: dictionary of (region, date) pairs where date holds the relevant
            dates for plotting cases by inferred symptom-onset
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    states = df_interim.STATE.unique().tolist()
    states.remove('NT')
    states.remove('ACT')

    date_filter = pd.date_range(start=start,end=end)

    #prepare NNDSS cases
    df_cases = df_interim.groupby(['NOTIFICATION_RECEIVE_DATE','STATE']).agg(sum)
    df_cases = df_cases.reset_index()


    fig, ax = plt.subplots(nrows=2, ncols=3, 
                        sharex=True, sharey=True,
                        figsize=(15,12)
                        )

    for i,state in enumerate(states):
        row = i//3
        col = i%3
        
        R_summary = R_summ_states[state]

        #a,b,R = Reff_from_case(df_state_I.values,lambda_state,prior_a=1, prior_b=2, tau=tau)

        #R_summary = generate_summary(R)
        
        fig, ax[row,col] = plot_Reff(R_summary, 
                                    dates=dates[state], 
                                    ax_arg=(fig, ax[row,col]),
                                    truncate=(0,nowcast_truncation),
                                    label='Our Model')

        fig, ax[row,col] = plot_Reff(R_summary, 
                                    dates=dates[state], 
                                    ax_arg=(fig, ax[row,col]),
                                    truncate=(nowcast_truncation,None),
                                    label='Nowcast')

        
        #plot formatting
        ax[row,col].set_title(state)
        ax[row,col].set_ylim((0,4))
        ax[row,col].set_xlim((pd.to_datetime(start),pd.to_datetime(end)))
        
        #plot cases behind
        ax2 = ax[row,col].twinx()

        ax2.bar(df_cases.loc[df_cases.STATE==state,'NOTIFICATION_RECEIVE_DATE'], 
                df_cases.loc[df_cases.STATE==state,'local']+df_cases.loc[df_cases.STATE==state,'imported'],
            color='grey',
                alpha=0.3
            )
        ax2.bar(df_cases.loc[df_cases.STATE==state,'NOTIFICATION_RECEIVE_DATE'], 
                df_cases.loc[df_cases.STATE==state,'local'],
            color='grey',
                alpha=0.8
            )

        # Set common labels
        fig.text(0.5, 0.01, 'Date', ha='center', va='center',
         fontsize=20)
        fig.text(0.08, 0.5, 
        'Effective \nReproduction Number', 
        ha='center', va='center', rotation='vertical',
        fontsize=20)
        fig.text(0.95, 0.5, 'Local Cases', ha='center', va='center', 
        rotation=270,
        fontsize=20)
        #plot old LSHTM estimates
        #df_june = df_L_R.loc[(df_L_R.date_of_analysis=='2020-07-27')&(df_L_R.state==state)]
        #df = df_june.loc[(df_june.date.isin(date_filter))]
        
        #ax[row,col].plot(df.date, df['median'], label='Old LSHTM',color='C1')
        #ax[row,col].fill_between(df.date, df['bottom'], df['top'],color='C1', alpha=0.3)
        #ax[row,col].fill_between(df.date, df['lower'], df['upper'],color='C1', alpha=0.3)
        
    
    if save:
        import os
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path,"../../figs/")
        os.makedirs(path+"EpyReff/", exist_ok=True )
        plt.savefig(path+"/EpyReff/Reff_tau_"+str(tau)+"_"+date+".png",dpi=300)
    return fig, ax
