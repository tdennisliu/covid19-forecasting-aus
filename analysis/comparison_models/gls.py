from statsmodels.regression.linear_model import GLS
from analysis.cprs import Reff_constants
from analysis.cprs import Reff_functions
#read in Google data


# read in microdistnacing survey

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


df_google = read_in_google(local=True,moving=True)
###Create dates
try:
    cprs_start_date = pd.to_datetime(argv[1])#2020-04-01')
    cprs_end_date = pd.to_datetime(argv[1])#'2020-07-22')
except:
    print("Running full validation dates")
    #Use July 2nd, Early Aug and Early Sept.
    cprs_start_date = pd.to_datetime('2020-04-01')
    cprs_end_date = pd.to_datetime('2020-10-07')
cprs_all_dates = pd.date_range(cprs_start_date, cprs_end_date, freq='7D')
cprs_dates = cprs_all_dates[cprs_all_dates!='2020-09-09']

#if argv[1]=='2020-09-09':
#    print("This won't run due to cprs date definitions, please comment out line 215.")

for data_date in cprs_dates:
    print(data_date)
    print(data_date.strftime('%d%b%Y'))
#data_date =  pd.to_datetime('2020-08-17')

    ## also filter Reff by 10 days!
    ## need to truncate most recent days of Reff
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

    print('cases loaded')
    df_Reff = df_state.groupby(['STATE','date_inferred']) #how = left to use Reff days, NNDSS missing dates
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



    df= df_google.merge(df_Reff[['date','state','mean','lower','upper',
                                'top','bottom','std','rho','rho_moving','local','imported']], on=['date','state'],how='inner')
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

    for value in ['mean','std','local','imported']:
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
    # Setup Design matrix

    # fit model

    model = GLS(y, X, sigma=None)

    # generate predictions

    backcast = model.predict(exog=X)

    # predict using forecasted X data

    forecast = model.predict(exog =X_forecast)
    # plot results

    # calculate crps