import numpy as np
import pandas as pd
from scipy.stats import nbinom, erlang, beta, binom, poisson, beta
import matplotlib.pyplot as plt
import os
class Person:
    """
    Individuals in the forecast
    """
    # Laura
    # default action_time to 0. This allows for code that doesn’t involve contact tracing (undetected cases) 
    # to continue without modification.
    def __init__(self,parent, infection_time,detection_time, detected,category:str, action_time = 0):
        """
        Category is one of 'I','A','S' for Imported, Asymptomatic and Symptomatic
        """
        self.parent = parent
        self.infection_time = infection_time
        self.detection_time = detection_time
        self.detected = detected
        self.category = category
        # Laura
        # Add action time to Person object
        self.action_time = action_time
    
class Forecast:
    """
    Forecast object that contains methods to simulate a forcast forward, given Reff and current state.
    """
    
    def __init__(self,current, state,start_date, people, 
        Reff=2.2,k=0.1,alpha_i=1,gam_list=[0.8],qi_list=[1], qa_list=[1/8], qs_list=[0.8],
        qua_ai= 1, qua_qi_factor=1, qua_qs_factor=1,forecast_R=None,R_I=None,
        forecast_date='2020-07-01', cross_border_state=None,cases_file_date=('25Jun','0835'),
        ps_list=[0.7], test_campaign_date=None, test_campaign_factor=1,
        Reff_file_date=None,
        ):
        import numpy as np
        import copy
        self.initial_state = current.copy() #Observed cases on start day
        #self.current=current
        self.state = state
        #start date sets day 0 in script to start_date
        self.start_date = pd.to_datetime(start_date,format='%Y-%m-%d')
        self.quarantine_change_date = pd.to_datetime(
            '2020-04-01',format='%Y-%m-%d').dayofyear - self.start_date.dayofyear
        self.initial_people = copy.deepcopy(people) #detected people only
        self.Reff = Reff
        self.alpha_i = alpha_i
        self.gam_list = gam_list
        self.ps_list = ps_list#beta.rvs(7,3,size=1000)
        self.qi_list = qi_list
        self.qa_list = qa_list
        self.qs_list = qs_list
        self.k = k
        self.qua_ai = qua_ai
        self.qua_qi_factor = qua_qi_factor
        self.qua_qs_factor=qua_qs_factor

        self.forecast_R = forecast_R
        self.R_I = R_I
        np.random.seed(1)
        #self.max_cases = 100000

        self.forecast_date = pd.to_datetime(
            forecast_date,format='%Y-%m-%d').dayofyear - self.start_date.dayofyear

        self.Reff_file_date = Reff_file_date
        self.cross_border_state = cross_border_state
        self.cases_file_date = cases_file_date

        if self.cross_border_state is not None:
            self.travel_prob = self.p_travel()
            self.travel_cases_after = 0
            #placeholders for run_state script 
            self.cross_border_seeds = np.zeros(shape=(1,2),dtype=int)
            self.cross_border_state_cases = np.zeros_like(self.cross_border_seeds)

        if test_campaign_date is not None:
            self.test_campaign_date = pd.to_datetime(
                test_campaign_date,format='%Y-%m-%d').dayofyear - self.start_date.dayofyear
            self.test_campaign_factor = test_campaign_factor
        else:
            self.test_campaign_date = None
        #import model parameters
        self.a_dict = {
            'ACT': {
                1:2,
                2:22,
                3:31*1.3,
                4:17,
                5:15,
                6:3,
            },
            'NSW': {
                1: 90,
                2: 408,
                3: 694*1.3,
                4: 380,
                5: 312,
                6: 276,
            },
            'NT': {
                1: 3,
                2: 4,
                3: 7*1.3,
                4: 9,
                5: 6,
                6: 4,
            },
            'QLD': {
                1:61,
                2:190,
                3:305*1.3,
                4:162,
                5:87,
                6:25,
            },
            'SA': {
                1:13,
                2:68,
                3:115*1.3,
                4:67,
                5:27,
                6:6
            },
            'TAS':{
                1:6,
                2:14,
                3:32*1.3,
                4:19,
                5:11,
                6:2,
            },
            'VIC': {
                1:62,
                2:208,
                3:255*1.3,
                4:157,
                5:87,
                6:188,
            },
            'WA': {
                1:15,
                2:73,
                3:154*1.3,
                4:115,
                5:110,
                6:78
            },
        }
        #changes below also need to be changed in simulate
        self.b_dict = {
            1: 6.2,
            2: 7.2,
            3: 5.2,
            4: 5.2,
            5: 22.2,
            6: 145.2 ## this needs to change for
                    # each change in forecast date
        }

        dir_path = os.getcwd()
        self.datapath = os.path.join(dir_path,'data/')


        assert len(people) == sum(current), "Number of people entered does not equal sum of counts in current status"
        
    #def generate_times(self,  i=2.5, j=1.25, m=1.2, n=1, size=10000):
    def generate_times(self,  i=3.64, j=3.07, m=5.51, n=0.948, size=10000):

        """
        Generate large amount of gamma draws to save on simulation time later
        """

        self.inf_times = np.random.gamma(i/j, j, size =size) #shape and scale
        self.detect_times = np.random.gamma(m/n,n, size = size)
        self.action_times = 1 + np.random.gamma(self.t_a_shape, self.t_a_scale, size = size)
        return None
    

    def iter_inf_time(self):
        """
        access Next inf_time
        """
        from itertools import cycle
        for time in cycle(self.inf_times):
            yield time
    
    def iter_detect_time(self):
        """
        access Next detect_time
        """
        from itertools import cycle
        for time in cycle(self.detect_times):
            yield time

    def iter_action_time(self):
        """
        access Next detect_time
        """
        from itertools import cycle
        for time in cycle(self.action_times):
            yield time    
    def initialise_sim(self,curr_time=0):
        """
        Given some number of cases in self.initial_state (copied),
        simulate undetected cases in each category and their 
        infectious times. Updates self.current for each person.
        """
        from math import ceil
        import copy
        if curr_time ==0:
            
            #grab a sample from parameter lists
            self.qs = self.choose_random_item(self.qs_list)
            self.qa = self.choose_random_item(self.qa_list)
            #resample qa until it is less than self.qs
            while self.qa>=self.qs:
                self.qa = self.choose_random_item(self.qa_list)
            self.qi = self.choose_random_item(self.qi_list)
            self.gam = self.choose_random_item(self.gam_list)
            
            self.ps = self.choose_random_item(self.ps_list)
            self.alpha_s = 1/(self.ps + self.gam*(1-self.ps))
            self.alpha_a = self.gam * self.alpha_s
            self.current = self.initial_state.copy()
            self.people = copy.deepcopy(self.initial_people)

            ## Laura
            self.secondary_cases=[]
            self.generation_times=[]

            #N samples for each of infection and detection times
            #Grab now and iterate through samples to save simulation
            self.generate_times(size=10000)
            self.get_inf_time = self.iter_inf_time()
            self.get_detect_time = self.iter_detect_time()
            self.get_action_time = self.iter_action_time()

            #counters for terminating early
            self.inf_backcast_counter = 0
            self.inf_forecast_counter = 0

            #assign infection time to those discovered
            # obs time is day =0
            for person in self.people.keys():
                self.people[person].infection_time = -1*next(self.get_inf_time)
        else:
            #reinitialising, so actual people need times
            #assume all symptomatic
            prob_symp_given_detect = self.qs*self.ps/(
                self.qs*self.ps + self.qa*(1-self.ps)
            )
            num_symp = binom.rvs(n=int(self.current[2]), p=prob_symp_given_detect)
            for person in range(int(self.current[2])):
                self.infected_queue.append(len(self.people))
                
                inf_time = next(self.get_inf_time)
                detection_time = next(self.get_detect_time)
                if person <- num_symp:
                    new_person = Person(-1, 
                    curr_time-1*detection_time ,
                    curr_time, 1, 'S')
                else:
                    new_person = Person(-1, 
                    curr_time-1*detection_time ,
                    curr_time, 1, 'A')
                
                self.people[len(self.people)] = new_person
                
                #self.cases[max(0,ceil(new_person.infection_time)), 2] +=1
                

        #num undetected is nbinom (num failures given num detected)
        if self.current[2]==0:
            num_undetected_s = nbinom.rvs(1,self.qs*self.qua_qs_factor)
        else:
            num_undetected_s = nbinom.rvs(self.current[2],self.qs*self.qua_qs_factor)
        
        ## Laura ,skip this for contact tracing
        #if self.current[0]==0:
        #    num_undetected_i = nbinom.rvs(1,self.qs*self.qua_qs_factor)
        #else:
        #    num_undetected_i = nbinom.rvs(self.current[0], self.qi*self.qua_qi_factor)
        num_undetected_i = 0

        ######
        total_s = num_undetected_s + self.current[2]

        #infer some non detected asymp at initialisation
        if total_s==0:
            num_undetected_a = nbinom.rvs(1, self.ps)
        else:
            num_undetected_a = nbinom.rvs(total_s, self.ps)

        #simulate cases that will be detected within the next week
        #for n in range(1,8):
            #just symptomatic?
            #self.people[len(self.people)] = Person(0, -1*next(self.get_inf_time) , n, 0, 'S')
        if curr_time==0:
            #Add each undetected case into people
            for n in range(num_undetected_i):
                self.people[len(self.people)] = Person(0, curr_time-1*next(self.get_inf_time) , 0, 0, 'I')
                self.current[0] +=1
            for n in range(num_undetected_a):
                self.people[len(self.people)] = Person(0, curr_time-1*next(self.get_inf_time) , 0, 0, 'A')
                self.current[1] +=1
            for n in range(num_undetected_s):
                self.people[len(self.people)] = Person(0, curr_time-1*next(self.get_inf_time) , 0, 0, 'S')
                self.current[2] +=1
        else:
            #reinitialised, so add these cases back onto cases
             #Add each undetected case into people
            for n in range(num_undetected_i):
                new_person = Person(-1, curr_time-1*next(self.get_inf_time) , 0, 0, 'I')
                self.infected_queue.append(len(self.people))
                self.people[len(self.people)] = new_person
                self.cases[max(0,ceil(new_person.infection_time)),0] +=1
            for n in range(num_undetected_a):
                new_person = Person(-1, curr_time-1*next(self.get_inf_time) , 0, 0, 'A')
                self.infected_queue.append(len(self.people))
                self.people[len(self.people)] = new_person
                self.cases[max(0,ceil(new_person.infection_time)),1] +=1
            for n in range(num_undetected_s):
                new_person = Person(-1, curr_time-1*next(self.get_inf_time) , 0, 0, 'S')
                self.infected_queue.append(len(self.people))
                self.people[len(self.people)] = new_person
                self.cases[max(0,ceil(new_person.infection_time)),2] +=1
            
        return None

    def read_in_Reff(self):
        """
        Read in Reff csv from Price et al 2020. Originals are in RDS, are converted to csv in R script
        """
        import pandas as pd
        #df= pd.read_csv(self.datapath+'R_eff_2020_04_23.csv', parse_dates=['date'])
        if self.cross_border_state is not None:
            states = [self.state,self.cross_border_state]
        else:
            states=[self.state]
        

        if self.forecast_R is not None:
            if self.Reff_file_date is None:
                import glob, os

                list_of_files = glob.glob(self.datapath+'soc_mob_R*.h5') 
                latest_file = max(list_of_files, key=os.path.getctime)
                print("Using file "+latest_file)
                df_forecast = pd.read_hdf(latest_file,
            key='Reff')
            else:
                df_forecast = pd.read_hdf(self.datapath+'soc_mob_R'+self.Reff_file_date+'.h5',
            key='Reff')
            num_days = df_forecast.loc[
                (df_forecast.type=='R_L')&(df_forecast.state==self.state)].shape[0]
            if self.R_I is not None:
                self.R_I = df_forecast.loc[
                    (df_forecast.type=='R_I')&
                    (df_forecast.state==self.state),
                    [i for i in range(1000)]].values[0,:]

            #R_L here 
            df_forecast = df_forecast.loc[df_forecast.type==self.forecast_R]

            #df = pd.concat([
            #            df.drop(['type','date_onset','confidence',
            #                 'bottom','top','mean_window','prob_control',
            #                'sd_window'],axis=1),
            #            df_forecast.drop(['type'],axis=1)
            #                ])
            
           #df = df.drop_duplicates(['state','date'],keep='last')
            df = df_forecast
            df = df.set_index(['state','date'])
        
        Reff_lookupdist ={}

        for state in states:
            Reff_lookupstate = {}
            if self.forecast_R =='R_L':
                dfReff_dict = df.loc[state,[0,1]].to_dict(orient='index')

                for key, stats in dfReff_dict.items():
                    #instead of mean and std, take all columns as samples of Reff
                    #convert key to days since start date for easier indexing
                    newkey = key.dayofyear - self.start_date.dayofyear

                    Reff_lookupstate[newkey] = df.loc[(state,key),
                    [i for i in range(1000)]].values

            else:
                #R_L0
                for day in range(num_days):
                    Reff_lookupstate[day] = 0.75*df.loc[state, [i for i in range(1000)]].values[0]
                print("Reff with mean %.2f" %np.mean(Reff_lookupstate[day]))
                print("90\% CrI {}".format(np.quantile(Reff_lookupstate[day],(0.05,0.95))))

            #Nested dict with key to state, then key to date
            Reff_lookupdist[state] = Reff_lookupstate

        if self.cross_border_state is not None:
            self.Reff_travel = Reff_lookupdist[self.cross_border_state]
        
        self.Reff = Reff_lookupdist[self.state]
        return None
    
    def choose_random_item(self, items,weights=None):
        from numpy.random import random
        if weights is None:
            #Create uniform weights
            weights = [1/len(items)] * len(items)
        r = random()
        for i,item in enumerate(items):
            r-= weights[i]
            if r <0:
                return item
        
            
    def new_symp_cases(self,num_new_cases:int):
        """
        Given number of new cases generated, assign them to symptomatic (S) with probability ps
        """
        #repeated Bernoulli trials is a Binomial (assuming independence of development of symptoms)
        
        symp_cases = binom.rvs(n=num_new_cases, p=self.ps)

        return symp_cases
    
    def generate_new_cases(self,parent_key, Reff,k,travel=True):
        """
        Generate offspring for each parent, check if they travel
        """
        
        from math import ceil
        from numpy.random import random, gamma

        #check parent category   
        if self.people[parent_key].category=='S':
            num_offspring = nbinom.rvs(n=k,p= 1- self.alpha_s*Reff/(self.alpha_s*Reff + k))
        elif self.people[parent_key].category=='A':
            num_offspring = nbinom.rvs(n=k, p = 1- self.alpha_a*Reff/(self.alpha_a*Reff + k))
        else:
            #Is imported
            if self.R_I is not None:
                #if splitting imported from local, change Reff to R_I 
                Reff = self.choose_random_item(self.R_I)
            if self.people[parent_key].infection_time < self.quarantine_change_date:
                #factor of 3 times infectiousness prequarantine changes

                num_offspring = nbinom.rvs(n=k, p = 1- self.qua_ai*Reff/(self.qua_ai*Reff + k))
            else:
                num_offspring = nbinom.rvs(n=k, p = 1- self.alpha_i*Reff/(self.alpha_i*Reff + k))
        
        if num_offspring >0:  
            
            num_sympcases = self.new_symp_cases(num_offspring)
            
            #Laura
            actual_offspring_counter =0

            if self.people[parent_key].category=='A':
                child_times = []
            for new_case in range(num_offspring):
                #define each offspring
                
                inf_time = self.people[parent_key].infection_time + next(self.get_inf_time)
               # LAURA
                if self.people[parent_key].detected==1:
                    #if parent detected
                    if inf_time> self.people[parent_key].action_time:
                        #infection occurs after isolation
                        #infection never occurs, skip
                        continue
                    else:
                        #infection does occur
                        actual_offspring_counter +=1
                        #generation interval even if undetected?
                        self.generation_times.append(
                            inf_time - self.people[parent_key].infection_time)
                # Laura
                # add an action_time = 0 when an offspring is first examined:
                action_time = 0

                if inf_time > self.forecast_date:
                    self.inf_forecast_counter +=1
                    if travel:
                        if self.cross_border_state is not None:
                        #check if SA person
                            if random() < self.travel_prob:
                                if ceil(inf_time) <= self.cases.shape[0]:
                                    self.cross_border_state_cases[max(0,ceil(inf_time)-1),self.num_of_sim] += 1
                                    detection_rv = random()
                                    detect_time = 0 #give 0 by default, fill in if passes
                            
                                    recovery_time = 0 #for now not tracking recoveries
                        
                                    if new_case <= num_sympcases-1: #minus 1 as new_case ranges from 0 to num_offspring-1 
                                        #first num_sympcases are symnptomatic, rest are asymptomatic
                                        category = 'S'                                 
                                        if detection_rv < self.qs:
                                            #case detected
                                            detect_time = inf_time + next(self.get_detect_time)

                                    else:
                                        category = 'A'
                                        detect_time = 0
                                        if detection_rv < self.qa:
                                            #case detected
                                            detect_time = inf_time + next(self.get_detect_time)
                                    self.people[len(self.people)] = Person(parent_key, inf_time, detect_time,recovery_time, category)
                                    self.cross_border_sim(len(self.people)-1,ceil(inf_time))
                                    #skip simulating this offspring in VIC
                                    #continue
                                else:
                                    #cross border seed happened after forecast
                                    self.travel_cases_after +=1
                else:
                    self.inf_backcast_counter +=1
                #normal case within state
                if self.people[parent_key].category=='A':
                    child_times.append(ceil(inf_time))
                if ceil(inf_time) > self.cases.shape[0]:
                    #new infection exceeds the simulation time, not recorded
                    self.cases_after = self.cases_after + 1                
                else:
                    #within forecast time
                    detection_rv = random()
                    detect_time = inf_time + next(self.get_detect_time)
                        
                    isdetected = 0 
                    
                    if new_case <= num_sympcases-1: #minus 1 as new_case ranges from 0 to num_offspring-1 
                        #first num_sympcases are symnptomatic, rest are asymptomatic
                        category = 'S'
                        self.cases[max(0,ceil(inf_time)-1),2] += 1
                        
                        if self.test_campaign_date is not None:
                            #see if case is during a testing campaign
                            if inf_time <self.test_campaign_date:
                                detect_prob = self.qs
                            else:
                                detect_prob = min(0.95,self.qs*self.test_campaign_factor)
                        else:
                            detect_prob = self.qs
                        if detection_rv < detect_prob:
                            #case detected
                            isdetected=1
                            # Laura
                            # if parent is undetected, assign a new time to action 
                            if self.people[parent_key].detected==0:
                                action_time = detect_time + next(self.get_action_time)
                            if detect_time < self.cases.shape[0]:
                                self.observed_cases[max(0,ceil(detect_time)-1),2] += 1

                    else:
                        category = 'A'
                        self.cases[max(0,ceil(inf_time)-1),1] += 1
                        #detect_time = 0
                        if self.test_campaign_date is not None:
                        #see if case is during a testing campaign
                            if inf_time <self.test_campaign_date:
                                detect_prob = self.qa
                            else:
                                detect_prob = min(0.95,self.qa*self.test_campaign_factor)
                        else:
                            detect_prob=self.qa
                        if detection_rv < detect_prob:
                            #case detected
                            isdetected=1
                            #detect_time = inf_time + next(self.get_detect_time)
                            # Laura 
                            # Get absolute action time, 
                            # if parent is not detected, assign an action time 
                            # action_time = self.people[parent_key].detection_time + 
                            # 2* draw from distrubtion
                            if self.people[parent_key].detected==0:
                                action_time = detect_time + 2*next(self.get_action_time)
                            if detect_time < self.cases.shape[0]:
                                self.observed_cases[max(0,ceil(detect_time)-1),1] += 1

                    # Laura 
                    #add new infected to queue
                    # contact trace 2=1 day before parent's detection time
                    if self.people[parent_key].detected==1:
                        #only check contact tracing if parent was detected

                        if inf_time < self.people[parent_key].detection_time - self.DAYS:
                            
                            self.infected_queue.append(len(self.people))
                            
                        #elif  (self.people[parent_key].detection_time - DAYS) < inf_time < (self.people[parent_key].action_time):
                        # elif ((self.people[parent_key].detection_time - DAYS) < inf_time) and (inf_time < (self.people[parent_key].action_time)):   
                        elif inf_time < self.people[parent_key].action_time:   
                            
                            x_rn = random()
                            if x_rn <= self.p_c:
                                #case caught in contact tracing
                                #inherit action time
                                action_time = self.people[parent_key].action_time
                                self.infected_queue.append(len(self.people))
                            
                        # else assign new time to action.
                        # need to add if de
                            else:
                                action_time = detect_time + next(self.get_action_time)
                                self.infected_queue.append(len(self.people))

                                
                    else:
                        #parent undetected
                        self.infected_queue.append(len(self.people))
                        

                    #add person to tracked people
                    # Laura # add action_time when recording
                    self.people[len(self.people)] = Person(parent_key, inf_time, detect_time,isdetected, 
                    category,action_time=action_time)
                    #Laura
            #Laura
            #record actual number of secondary cases
            prop_cases_prevented = (num_offspring -actual_offspring_counter)/num_offspring
            self.secondary_cases.append(prop_cases_prevented)
            if travel:
                #for parent, check their cross border travel
                if self.cross_border_state is not None:
                    #Run cross border sim across children
                    inf_time = self.people[parent_key].infection_time

                    detect_time = self.people[parent_key].detection_time

                    if self.people[parent_key].infection_time>self.forecast_date:
                        #check it is after forecast date but before
                        #end date
                        if ceil(inf_time)-1 < self.cases.shape[0]:
                            #check if travel
                            #symptomatic people here
                            pre_symp_time = inf_time
                            while pre_symp_time < detect_time:
                                travel_rv = random()
                                if travel_rv<self.travel_prob:
                                    #travelled
                                    ## did they infect?
                                    #symptomatic
                                    self.cross_border_sim(parent_key,ceil(pre_symp_time))

                                #can travel more than once
                                pre_symp_time +=1 #move forward a day
                                if pre_symp_time>self.cases.shape[0]:
                                    break
                            if detect_time==0:
                                #asymptomatics
                                if self.people[parent_key].category=='A':
                                    for pre_symp_time in child_times:
                                        if pre_symp_time< self.cases.shape[0] -1:
                                            #only care if still within forecast time
                                            travel_rv = random()
                                            if travel_rv<self.travel_prob:
                                            #travelled
                                            ## did they infect?
                                                self.cross_border_sim(parent_key,ceil(pre_symp_time))

                                            #remove case from original state?
        return None
    
    def cases_detected(self,new_cases):
        """
        Given a tuple of new_cases generated, return the number of cases detected
        """
        #Number of detected cases in each class is Binomial with p = q_j

        i_detected = binom.rvs(n=new_cases[0],p=self.qi)
        a_detected = binom.rvs(n=new_cases[1],p=self.qa)
        s_detected = binom.rvs(n=new_cases[2],p=self.qs)

        return i_detected, a_detected, s_detected
    
    def import_arrival(self,period,size=1):
        """
        Poisson likelihood of arrivals of imported cases, with a Gamma
        prior on the mean of Poisson, results in a posterior predictive
        distribution of imported cases a Neg Binom
        """
        a = self.a_dict[self.state][period]
        b = self.b_dict[period]
        if size==1:
            return nbinom.rvs(a, 1-1/(b+1))
        else:
            return nbinom.rvs(a, 1-1/(b+1),size=size)

    def simulate(self, end_time,sim,seed,DAYS=2, p_c =0.8, t_a_shape = 2/1, t_a_scale=1 ):
        """
        Simulate forward until end_time
        """
        from collections import deque
        from math import ceil
        import gc
        np.random.seed(seed)
        self.num_of_sim = sim

        self.DAYS = DAYS
        self.p_c = p_c
        self.t_a_shape = t_a_shape
        self.t_a_scale = t_a_scale

        #generate storage for cases
        self.cases = np.zeros(shape=(end_time, 3),dtype=float)
        self.observed_cases = np.zeros_like(self.cases)
        
        self.observed_cases[0,:] = self.initial_state.copy()

        #Initalise undetected cases and add them to current
        self.initialise_sim()
        #number of cases after end time
        self.cases_after = 0 #gets incremented in generate new cases

        #Record day 0 cases
        self.cases[0,:] = self.current.copy() 
        # Generate imported cases
        num_days={
            1: 6,
            2: 8,
            3: 4,
            4: 5,
            5: 22,#min(end_time - self.quarantine_change_date -7, 24 ),
            6: max(0,end_time- 6-8-4-5-22),
        }
        qi = {
            1:self.qi *self.qua_qi_factor,
            2:self.qi,
            3:self.qi,
            4:0.95,
            5:0.98,
            6:0.98,
        }
        new_imports = []
        unobs_imports =[]
        for period in range(1,7): 
            ##ADDED continue TO SKIP THIS FOR LB
            continue
            ######
            obs_cases = self.import_arrival(
                period=period, size=num_days[period])
            #generate undetected people
            #if obs_cases includes 0.... then add one for nbinom
            nbinom_var = [o+1 if o ==0 else o for o in obs_cases ]
            unobs = nbinom.rvs(nbinom_var, p=qi[period])#modify qi here
            unobs_imports.extend(unobs)
            new_imports.extend(obs_cases + unobs)

        for day, imports in enumerate(new_imports):
            ##ADDED continue TO SKIP THIS FOR LB
            continue
            #######
            
            self.cases[day,0] = imports
            for n in range(imports):
                #Generate people
                if n - unobs_imports[day]>=0:
                    #number of observed people
                    new_person = Person(0,day,day +next(self.get_detect_time),1,'I')
                    self.people[len(self.people)] = new_person
                    if new_person.detection_time <= end_time:
                        self.observed_cases[max(0,ceil(new_person.detection_time)-1),0] +=1
                else:
                    #unobserved people
                    new_person = Person(0,day,0,0,'I')
                    self.people[len(self.people)] = new_person
                    if day <= end_time:
                        self.cases[max(0,day-1), 0] +=1
         #####

        #Create queue for infected people
        self.infected_queue = deque()
        #Assign people to infected queue
        for key, person in self.people.items():
            #add to the queue
            self.infected_queue.append(key)
            #Record their times
            if person.infection_time> end_time:
                #initial undetected cases have slim chance to be infected 
                #after end_time
                if person.category!='I':
                    #imports shouldn't count for extinction counts
                    self.cases_after +=1
                    print("cases after at initialisation")
            #else:
            #    if person.category=='S':
            #        self.cases[max(0,ceil(person.infection_time)),2] +=1
            #        if (person.detection_time < end_time) & (person.detection_time!=0):
            #            self.observed_cases[max(0,ceil(person.detection_time)), 2] +=1
            #    elif person.category=='I':
                    #Imports recorded on creation in sim
            #        continue
            #    elif person.category=='A':
            #        self.cases[max(0,ceil(person.infection_time)),1] +=1
            #        if (person.detection_time < end_time) & (person.detection_time!=0):
            #            self.observed_cases[max(0,ceil(person.detection_time)), 1] +=1
            #    else:
                    print("ERROR: not right category")

        #Record initial inferred obs including importations.
        self.inferred_initial_obs = self.observed_cases[0,:].copy() 
        #print(self.inferred_initial_obs, self.current)


        # General simulation through time by proceeding through queue
        # of infecteds
        n_resim = 0
        self.bad_sim = False
        reinitialising_window = 0
        self.daycount= 0
        while len(self.infected_queue)>0:
            day_end = self.people[self.infected_queue[0]].infection_time
            if day_end < self.forecast_date:
                if self.inf_backcast_counter> self.max_backcast_cases:
                    print("Sim "+str(self.num_of_sim
                    )+" in "+self.state+" has > "+str(self.max_backcast_cases)+" cases in backcast. Ending")
                    self.num_too_many+=1
                    self.bad_sim = True
                    break
            else:
                #check max cases for after forecast date
                if self.inf_forecast_counter>self.max_cases:
                    #hold value forever
                    if day_end < self.cases.shape[0]-1:
                        self.cases[ceil(day_end):,2] = self.cases[ceil(day_end)-2,2]

                        self.observed_cases[ceil(day_end):,2] = self.observed_cases[ceil(day_end)-2,2]
                    else:
                        self.cases_after +=1
                
                    print("Sim "+str(self.num_of_sim
                        )+" in "+self.state+" has >"+str(self.max_cases)+" cases in forecast period.")
                    self.num_too_many+=1
                    break

            
            ## stop if parent infection time greater than end time
            if self.people[self.infected_queue[0]].infection_time >end_time:
                self.infected_queue.popleft()
                print("queue had someone exceed end_time!!")
            else:
                
                #take approproate Reff based on parent's infection time
                curr_time = self.people[self.infected_queue[0]].infection_time
                if type(self.Reff)==int:
                    Reff = 1
                    print("using flat Reff")
                elif type(self.Reff)==dict:
                    while True:
                        #sometimes initial cases infection time is pre
                        #Reff data, so take the earliest one
                        try:
                            Reff = self.choose_random_item(self.Reff[ceil(curr_time)-1])
                        except KeyError:
                            if curr_time>0:
                                print("Unable to find Reff for this parent at time: %.2f" % curr_time)
                                raise KeyError
                            curr_time +=1
                            continue
                        break
                #generate new cases with times
                parent_key = self.infected_queue.popleft()
                #recorded within generate new cases
                self.generate_new_cases(parent_key,Reff=Reff,k = self.k) 
        #self.people.clear()
        if self.bad_sim ==False:
            #Check simulation for discrepancies
            for day in range(end_time):
                #each day runs through self.infected_queue
                
                ##ADDED continue TO SKIP THIS FOR LB
                continue
                #######
                
                missed_outbreak = self.data_check(day) #True or False
                if missed_outbreak:
                    self.daycount +=1                
                    if self.daycount >= reinitialising_window:
                        n_resim +=1
                        #print("Local outbreak in "+self.state+" not simulated on day %i" % day)
                        #cases to add
                        #treat current like empty list
                        self.current[2] = max(0,self.actual[day] - sum(self.observed_cases[day,1:])) 
                        self.current[2] += max(0,self.actual[day-1] - sum(self.observed_cases[day-1,1:]))
                        self.current[2] += max(0,self.actual[day-2] - sum(self.observed_cases[day-2,1:]))

                        #how many cases are symp to asymp
                        prob_symp_given_detect = self.qs*self.ps/(
                        self.qs*self.ps + self.qa*(1-self.ps)
                        )
                        num_symp = binom.rvs(n=int(self.current[2]), 
                            p=prob_symp_given_detect)
                        #distribute observed cases over 3 days
                         #Triangularly
                        self.observed_cases[max(0,day),2] += num_symp//2
                        self.cases[max(0,day),2] += num_symp//2

                        self.observed_cases[max(0,day-1),2] += num_symp//3
                        self.cases[max(0,day-1),2] += num_symp//3

                        self.observed_cases[max(0,day-2),2] += num_symp//6
                        self.cases[max(0,day-2),2] +=num_symp//6

                        #add asymptomatic
                        num_asymp = self.current[2] - num_symp
                        self.observed_cases[max(0,day),2] += num_asymp//2
                        self.cases[max(0,day),2] += num_asymp//2

                        self.observed_cases[max(0,day-1),2] += num_asymp//3
                        self.cases[max(0,day-1),2] += num_asymp//3

                        self.observed_cases[max(0,day-2),2] += num_asymp//6
                        self.cases[max(0,day-2),2] +=num_asymp//6

                        self.initialise_sim(curr_time=day)
                        #print("Reinitialising with %i new cases "  % self.current[2] )

                        #reset days to zero
                        self.daycount = 0

                if n_resim> 10:
                    print("This sim reinitilaised %i times" % n_resim)
                    self.bad_sim = True
                    n_resim = 0
                    break
                #Each check of day needs to simulate the cases before moving
                # to next check, otherwise will be doubling up on undetecteds
                while len(self.infected_queue)>0:
                    day_end = self.people[self.infected_queue[0]].infection_time
                    
                    #check for exceeding max_cases
                    if day_end <self.forecast_date:
                        if self.inf_backcast_counter > self.max_backcast_cases:
                            print("Sim "+str(self.num_of_sim
                            )+" in "+self.state+" has > "+str(self.max_backcast_cases)+" cases in backcast. Ending")
                            self.num_too_many+=1
                            self.bad_sim = True
                            break
                    else:
                        if self.inf_forecast_counter> self.max_cases:
                            
                            self.cases[ceil(day_end):,2] = self.cases[ceil(day_end)-2,2]

                            self.observed_cases[ceil(day_end):,2] = self.observed_cases[ceil(day_end)-2,2]
                            print("Sim "+str(self.num_of_sim
                                )+" in "+self.state+" has >"+str(self.max_cases)+" cases in forecast period.")
                            self.num_too_many+=1
                            break
                    ## stop if parent infection time greater than end time
                    if self.people[self.infected_queue[0]].infection_time >end_time:
                        personkey =self.infected_queue.popleft()
                        print("queue had someone exceed end_time!!")
                    else:
                        #take approproate Reff based on parent's infection time
                        curr_time = self.people[self.infected_queue[0]].infection_time
                        if type(self.Reff)==int:
                            Reff = 2
                        elif type(self.Reff)==dict:
                            while True:
                                #sometimes initial cases infection time is pre
                                #Reff data, so take the earliest one
                                try:
                                    Reff = self.choose_random_item(self.Reff[ceil(curr_time)-1])
                                except KeyError:
                                    if curr_time>0:
                                        print("Unable to find Reff for this parent at time: %.2f" % curr_time)
                                        raise KeyError
                                    curr_time +=1
                                    continue
                                break
                        #generate new cases with times
                        parent_key = self.infected_queue.popleft()
                        self.generate_new_cases(parent_key,Reff=Reff,k=self.k)
                        #missed_outbreak = max(1,missed_outbreak*0.9)
                else:
                    #pass in here if while queue loop completes 
                    continue
                #only reach here if while loop breaks, so break the data check
                break
                
        #LB needs people recorded, do not clear this attribute
        #self.people.clear()
        gc.collect()
        if self.bad_sim:
            #return NaN arrays for all bad_sims
            self.metric = np.nan
            self.cumulative_cases = np.empty_like(self.cases)
            self.cumulative_cases[:] = np.nan
            return (self.cumulative_cases,self.cumulative_cases, {
                'qs':self.qs,
                'metric':self.metric,
                'qa':self.qa,
                'qi':self.qi,
                'alpha_a':self.alpha_a,
                'alpha_s':self.alpha_s,
                #'accept':self.accept,
                'ps':self.ps,
                'bad_sim':self.bad_sim,
                # Laura add
                'Model_people':len(self.people),
                'cases_after':self.cases_after,
                'travel_seeds': self.cross_border_seeds[:,self.num_of_sim],
                'travel_induced_cases'+str(self.cross_border_state):self.cross_border_state_cases,
                'num_of_sim':self.num_of_sim,
            }
            )
        else:
            #good sim

            ## Perform metric for ABC
            self.get_metric(end_time)

            return (
                self.cases.copy(), 
                self.observed_cases.copy(), {
                'qs':self.qs,
                'metric':self.metric,
                'qa':self.qa,
                'qi':self.qi,
                'alpha_a':self.alpha_a,
                'alpha_s':self.alpha_s,
                #'accept':self.metric>=0.8,
                'ps':self.ps,
                'bad_sim':self.bad_sim,
                # Laura add
                'Model_people':len(self.people),
                'secondary_cases':self.secondary_cases,
                'generation_times': self.generation_times,
                'cases_after':self.cases_after,
                'travel_seeds': self.cross_border_seeds[:,self.num_of_sim],
                'travel_induced_cases'+str(self.cross_border_state):self.cross_border_state_cases[:,self.num_of_sim],
                'num_of_sim':self.num_of_sim,
            }
            ) 

    def simulate_then_reset(self, *args, **kwargs):
        """
        Run simulate then reset the simulation
        """
        cases_array, observed_cases_array, params = self.simulate(*args, **kwargs)
        self.reset_to_start(self.initial_people)
        return cases_array, observed_cases_array, params


    def simulate_many(self, end_time, n_sims):
        """
        Simulate multiple times 
        """
        self.end_time = end_time
        # Read in actual cases from NNDSS
        self.read_in_cases()
        import_sims = np.zeros(shape=(end_time, n_sims), dtype=float)
        import_sims_obs = np.zeros_like(import_sims)
        

        import_inci = np.zeros_like(import_sims)
        import_inci_obs = np.zeros_like(import_sims)

        asymp_inci = np.zeros_like(import_sims)
        asymp_inci_obs = np.zeros_like(import_sims)

        symp_inci = np.zeros_like(import_sims)
        symp_inci_obs = np.zeros_like(import_sims)

        bad_sim = np.zeros(shape=(n_sims),dtype=int)

        #ABC parameters
        metrics = np.zeros(shape=(n_sims),dtype=float)
        qs = np.zeros(shape=(n_sims),dtype=float)
        qa = np.zeros_like(qs)
        qi = np.zeros_like(qs)
        alpha_a = np.zeros_like(qs)
        alpha_s = np.zeros_like(qs)
        accept = np.zeros_like(qs)
        ps = np.zeros_like(qs)


        #extinction prop
        cases_after = np.empty_like(metrics) #dtype int
        self.cross_border_seeds = np.zeros(shape=(end_time,n_sims),dtype=int)
        self.cross_border_state_cases = np.zeros_like(self.cross_border_seeds)
        self.num_bad_sims = 0
        self.num_too_many = 0
        for n in range(n_sims):
            if n%(n_sims//10)==0:
                print("{} simulation number %i of %i".format(self.state) % (n,n_sims))
            
            inci, inci_obs, param_dict = self.simulate(end_time, n,n)
            if self.bad_sim:
                bad_sim[n] = 1
                print("Sim "+str(n)+" of "+self.state+" is a bad sim")
                self.num_bad_sims +=1
            else:
                #good sims
                ## record all parameters and metric
                metrics[n] = self.metric
                qs[n] = self.qs
                qa[n] = self.qa
                qi[n] = self.qi
                alpha_a[n] = self.alpha_a
                alpha_s[n] = self.alpha_s
                accept[n] = int(self.metric>=0.8)
                cases_after[n] = self.cases_after
                ps[n] =self.ps
            
            

            import_inci[:,n] = inci[:,0]
            asymp_inci[:,n] = inci[:,1]
            symp_inci[:,n] = inci[:,2]

            import_inci_obs[:,n] = inci_obs[:,0]
            asymp_inci_obs[:,n] = inci_obs[:,1]
            symp_inci_obs[:,n] = inci_obs[:,2]

        #Apply sim metric here and record
        #dict of arrays n_days by sim columns
        results = {
            'imports_inci': import_inci,
            'imports_inci_obs': import_inci_obs,
            'asymp_inci': asymp_inci,
            'asymp_inci_obs': asymp_inci_obs,
            'symp_inci': symp_inci,
            'symp_inci_obs': symp_inci_obs,
            'total_inci_obs': symp_inci_obs + asymp_inci_obs,
            'total_inci': symp_inci + asymp_inci,
            'all_inci': symp_inci + asymp_inci + import_inci,
            'bad_sim': bad_sim,
            'metrics': metrics,
            'accept': accept,
            'qs':qs,
            'qa':qa,
            'qi':qi,
            'alpha_a':alpha_a,
            'alpha_s':alpha_s,
            'cases_after':cases_after,
            'travel_seeds': self.cross_border_seeds,
            'travel_induced_cases'+str(self.cross_border_state):self.cross_border_state_cases,
            'ps':ps,
        }
        
        self.results = self.to_df(results)
        print("Number of bad sims is %i" % self.num_bad_sims)
        print("Number of sims in "+self.state\
            +" exceeding "+\
                str(self.max_cases//1000)+"k cases is "+str(self.num_too_many))
        return self.state,self.results
    

    def to_df(self,results):
        """
        Put results into a pandas dataframe and record as h5 format
        """
        import pandas as pd

        df_results = pd.DataFrame()
        n_sims = results['symp_inci'].shape[1]
        days = results['symp_inci'].shape[0]

        sim_vars=['bad_sim','metrics','qs','qa','qi',
        'accept','cases_after','alpha_a','alpha_s','ps'] 

        for key, item in results.items():
            if key not in sim_vars:
                df_results = df_results.append(
                    pd.DataFrame(
                        item.T,index=pd.MultiIndex.from_product([
                            [key], range(n_sims)],
                            names=['Category', 'sim']
                            )
                        )
                    )
        df_results.columns = pd.date_range(start = self.start_date,
                            periods=days #num of days
                        )
        df_results.columns = [col.strftime('%Y-%m-%d') for 
                    col in df_results.columns]
        #Record simulation variables 
        for var in sim_vars:       
            df_results[var] = [results[var][sim] for cat,sim in df_results.index]

        print("Saving results for state "+self.state)
        if self.forecast_R is None:
            df_results.to_parquet(
                "./results/"+self.state+self.start_date.strftime(
                    format='%Y-%m-%d')+"sim_results"+str(n_sims)+"days_"+str(days)+".parquet",
                    )
        else:
            df_results.to_parquet(
                "./results/"+self.state+self.start_date.strftime(
                    format='%Y-%m-%d')+"sim_"+self.forecast_R+str(n_sims)+"days_"+str(days)+".parquet",
                    )

        return df_results


    def data_check(self,day):
        """
        A metric to calculate how far the simulation is from the actual data
        """
        try:
            actual_3_day_total = 0
            for i in range(3):
                actual_3_day_total += self.actual[max(0,day-i)]
            threshold = 10*max(1,sum(
                self.observed_cases[
                    max(0,day-2):day+1,2] + self.observed_cases[
                        max(0,day-2):day+1,1]
                )
            )
            if  actual_3_day_total > threshold:
                return min(3,actual_3_day_total/threshold)
            else:
                #no outbreak missed
                return False

        except KeyError:
            #print("No cases on day %i" % day)
            return False
        
    def get_metric(self,end_time,omega=0.2):
        """
        Calculate the value of the metric of the current sim compared 
        to NNDSS data
        """

        ##missing dates
        #Deprecated now (DL 03/07/2020)
        #missed_dates = [day for day in range(end_time) 
        #    if day not in self.actual.keys()]

        self.actual_array = np.array([self.actual[day]
        #if day not in missed_dates else 0 
        for day in range(end_time) ])

        #calculate case differences
        #moving windows
        sim_cases =self.observed_cases[
            :len(self.actual_array),2] + \
                self.observed_cases[:
                len(self.actual_array),1] #include asymp cases.
        
        #convolution with 1s should do cum sum 
        window = 7
        sim_cases = np.convolve(sim_cases,
            [1]*window,mode='valid')
        actual_cum = np.convolve(self.actual_array,
            [1]*window,mode='valid')
        cases_diff = abs(sim_cases - actual_cum)
        
        #if sum(cases_diff) <= omega * sum(self.actual_array):
            #cumulative diff passes, calculate metric

            #sum over days number of times within omega of actual
        self.metric = sum(
            np.square(cases_diff)#,np.maximum(omega* actual_cum,7)
            )
        
        self.metric = self.metric/(end_time-window) #max is end_time

        return None
    def read_in_cases(self):
        """
        Read in NNDSS case data to measure incidence against simulation
        """
        import pandas as pd
        from datetime import timedelta
        import glob
        
        if self.cases_file_date is None:
            import glob, os

            list_of_files = glob.glob(self.datapath+'COVID-19 UoM*.xlsx') 
            path = max(list_of_files, key=os.path.getctime)
            print("Using file "+path)
        else:
            path = self.datapath+"COVID-19 UoM "+self.cases_file_date+"*.xlsx"

        for file in glob.glob(path):
            df = pd.read_excel(file,
                       parse_dates=['SPECIMEN_DATE','NOTIFICATION_DATE','NOTIFICATION_RECEIVE_DATE','TRUE_ONSET_DATE'],
                       dtype= {'PLACE_OF_ACQUISITION':str})
        if len(glob.glob(path))!=1:
            print("There are %i files with the same date" %len(glob.glob(path)))
        
            if len(glob.glob(path)) >1:
                print("Using an arbritary file")
        df = df.loc[df.STATE==self.state]

        #Set imported cases, local cases have 1101 as first 4 digits
        df.PLACE_OF_ACQUISITION.fillna('00038888',inplace=True) #Fill blanks with simply unknown

        df['date_inferred'] = df.TRUE_ONSET_DATE
        df.loc[df.TRUE_ONSET_DATE.isna(),'date_inferred'] = df.loc[df.TRUE_ONSET_DATE.isna()].NOTIFICATION_DATE - timedelta(days=5)
        df.loc[df.date_inferred.isna(),'date_inferred'] = df.loc[df.date_inferred.isna()].NOTIFICATION_RECEIVE_DATE - timedelta(days=6)

        df['imported'] = df.PLACE_OF_ACQUISITION.apply(lambda x: 1 if x[-4:]=='8888' and x != '00038888' else 0)
        df['local'] = 1 - df.imported
                
        if self.state=='VIC':
            #data quality issue
            df.loc[df.date_inferred=='2002-07-03','date_inferred'] = pd.to_datetime('2020-07-03')
            df.loc[df.date_inferred=='2002-07-17','date_inferred'] = pd.to_datetime('2020-07-17')
        df = df.groupby(['date_inferred'])[['imported','local']].sum()
        df.reset_index(inplace=True)
        df['date'] = df.date_inferred.apply(lambda x: x.dayofyear) -self.start_date.dayofyear
        df = df.sort_values(by='date')

        self.max_cases = max(500000,10*sum(df.local.values) + sum(df.imported.values))
        self.max_backcast_cases = max(100,4*sum(df.local.values) + sum(df.imported.values))
        #self.max_cases = max(self.max_cases, 1000)
        df = df.set_index('date')
        #fill missing dates with 0 up to end_time
        df = df.reindex(range(self.end_time), fill_value=0)
        self.actual = df.local.to_dict()
       
        return None

    def p_travel(self):
        """
        given a state to go to, what it probability of travel? 
        """
        ##Pop from Rob's work
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
        T_volume_ij = {
            'NSW':{
                'ACT':3000,
                #'NT':?,
                'SA':5300,
                'VIC':26700,
                'QLD':14000,
                'TAS':2500,
                'WA':5000,
            },
            'VIC':{
                'ACT':3600,
                'NSW':26700,
                'SA':12700,
                'QLD':11000,
                'TAS':5000,
                'WA':6000,
                #'NT':,
            },
            'SA':{
                'ACT':1200,
                'NSW':5300,
                'QLD':2500,
                #'TAS':,
                'VIC':12700,
                'WA':2000,
            },
            
        }
        #air and car travel from VIC to SA divided by pop of VIC
        try:
            p = T_volume_ij[
                    self.state][self.cross_border_state
                    ]/(pop[self.state]+pop[self.cross_border_state])
        except KeyError:
            print("Cross border state not implemented yet")
            raise KeyError
        return p
    def cross_border_sim(self,parent_key,day:int):
        """
        Simulate a cross border interaction between two states, where 
        export_state is the state in which a case is coming from.

        Need to feed in people, can be blank in attributes

        Feed in a time series of cases? Read in the timeseries?
        """
        import pandas as pd
        import numpy as np
        from math import ceil

        from collections import deque
        Reff = self.choose_random_item(self.Reff_travel[day])
        self.travel_infected_queue = deque()
        self.travel_people = {}
        #check parent category   
        if self.people[parent_key].category=='S':
            num_offspring = nbinom.rvs(self.k, 1- self.alpha_s*Reff/(self.alpha_s*Reff + self.k))
        elif self.people[parent_key].category=='A':
            num_offspring = nbinom.rvs(n=self.k, p = 1- self.alpha_a*Reff/(self.alpha_a*Reff + self.k))
        else:
            #Is imported
            if self.R_I is not None:
                #if splitting imported from local, change Reff to R_I 
                Reff = self.choose_random_item(self.R_I)
            if self.people[parent_key].infection_time < self.quarantine_change_date:
                #factor of 3 times infectiousness prequarantine changes

                num_offspring = nbinom.rvs(n=self.k, p = 1- self.qua_ai*Reff/(self.qua_ai*Reff + self.k))
            else:
                num_offspring = nbinom.rvs(n=self.k, p = 1- self.alpha_i*Reff/(self.alpha_i*Reff + self.k))
        if num_offspring >0:
            #import was successful, generate first gen offspring
            num_sympcases = self.new_symp_cases(num_offspring)
            for new_case in range(num_offspring):
                inf_time = next(self.get_inf_time) + self.people[parent_key].infection_time
                if inf_time < day +1:
                    #record seeded
                    if ceil(inf_time) > self.cases.shape[0]:
                        #new infection exceeds the simulation time, not recorded
                        self.travel_cases_after = self.travel_cases_after + 1
                    else:
                        self.cross_border_seeds[day-1,self.num_of_sim] += 1
                        #successful only if day trip
                        detection_rv = 1 #random() zeroed to just see all travel for now
                        detect_time = 0 #give 0 by default, fill in if passes
                            
                        recovery_time = 0 #for now not tracking recoveries
                        
                        if new_case <= num_sympcases-1: #minus 1 as new_case ranges from 0 to num_offspring-1 
                            #first num_sympcases are symnptomatic, rest are asymptomatic
                            category = 'S'
                            #record all cases the same for now
                            self.cross_border_state_cases[max(0,ceil(inf_time)-1),self.num_of_sim] += 1
                            
                            if detection_rv < self.qs:
                                #case detected
                                detect_time = inf_time + next(self.get_detect_time)
                                if detect_time < self.cases.shape[0]:
                                    print("placeholder")
                                    #self.observed_cases[max(0,ceil(detect_time)-1),2] += 1

                        else:
                            category = 'A'
                            self.cross_border_state_cases[max(0,ceil(inf_time)-1),self.num_of_sim] +=1
                            detect_time = 0
                            if detection_rv < self.qa:
                                #case detected
                                detect_time = inf_time + next(self.get_detect_time)
                                if detect_time < self.cases.shape[0]:
                                    print("placeholder")
                                    #self.observed_cases[max(0,ceil(detect_time)-1),1] += 1

                        #add new infected to queue
                        self.travel_infected_queue.append(len(self.people))

                        #add person to tracked people
                        self.travel_people[len(self.people)] = Person(parent_key, inf_time, detect_time,recovery_time, category)



                
                    
        while len(self.travel_infected_queue) >0:
            parent_key = self.travel_infected_queue.popleft()
            inf_day = ceil(self.travel_people[parent_key].infection_time)
            Reff = self.choose_random_item(self.Reff_travel[inf_day])
            self.generate_travel_cases(parent_key, Reff)
    
        return None
            

    def generate_travel_cases(self,parent_key,Reff):
        """
        Generate and record cases in cross border state 
        """
        from math import ceil
        #check parent category   
        if self.travel_people[parent_key].category=='S':
            num_offspring = nbinom.rvs(self.k, 1- self.alpha_s*Reff/(self.alpha_s*Reff + self.k))
        elif self.travel_people[parent_key].category=='A':
            num_offspring = nbinom.rvs(n=self.k, p = 1- self.alpha_a*Reff/(self.alpha_a*Reff + self.k))
        else:
            #Is imported
            if self.R_I is not None:
                #if splitting imported from local, change Reff to R_I 
                Reff = self.choose_random_item(self.R_I)
                num_offspring = nbinom.rvs(n=self.k, p = 1- self.alpha_i*Reff/(self.alpha_i*Reff + self.k))

        if num_offspring >0:
            num_sympcases = self.new_symp_cases(num_offspring)
            for new_case in range(num_offspring):
                inf_time = next(self.get_inf_time) + self.travel_people[parent_key].infection_time
                if inf_time < self.cases.shape[0]:
                    #record case
                    self.cross_border_state_cases[max(0,ceil(inf_time)-1),self.num_of_sim] += 1
                    detection_rv = 1 #random() zeroed to just see all travel for now
                    detect_time = 0 #give 0 by default, fill in if passes
                        
                    recovery_time = 0 #for now not tracking recoveries
                    
                    if new_case <= num_sympcases-1: #minus 1 as new_case ranges from 0 to num_offspring-1 
                        #first num_sympcases are symnptomatic, rest are asymptomatic
                        category = 'S'
                        #record all cases the same for now
                        self.cross_border_state_cases[max(0,ceil(inf_time)-1),self.num_of_sim] += 1
                        
                        if detection_rv < self.qs:
                            #case detected
                            detect_time = inf_time + next(self.get_detect_time)
                            if detect_time < self.cases.shape[0]:
                                print("placeholder")
                                #self.observed_cases[max(0,ceil(detect_time)-1),2] += 1

                    else:
                        category = 'A'
                        self.cross_border_state_cases[max(0,ceil(inf_time)-1),self.num_of_sim] +=1
                        detect_time = 0
                        if detection_rv < self.qa:
                            #case detected
                            detect_time = inf_time + next(self.get_detect_time)
                            if detect_time < self.cases.shape[0]:
                                print("placeholder")
                                #self.observed_cases[max(0,ceil(detect_time)-1),1] += 1

            
                    #add new infected to queue
                    self.travel_infected_queue.append(len(self.people))

                    #add person to tracked people
                    self.travel_people[len(self.people)] = Person(parent_key, inf_time, detect_time,recovery_time, category)
                else:
                    #new infection exceeds the simulation time, not recorded
                    #not added to queue
                    self.travel_cases_after = self.travel_cases_after + 1   
        return None
    
    def reset_to_start(self,people):
        """
        Reset forecast object back to initial conditions and reinitialise
        """
        ##Laura
        import gc,copy
        self.people.clear()
        gc.collect()
        self.people = copy.deepcopy(people)
        return None
