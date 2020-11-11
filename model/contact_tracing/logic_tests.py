import unittest
from ct_sim_class import Forecast, Person
import numpy as np
import pandas as pd
from numpy.random import beta, gamma

class TestGenerateCases(unittest.TestCase):
    """
    test whether the appropriate number of cases occur
    """

    def setUp(self):
        DAYS_list = (-3,-2,-1,0,1,2)
        sim =1
        DAYS = DAYS_list[1] #select right day from list
        p_c = 0.75
        ##########
        #PARAMETERS TO CHANGE
        #########

        #time to isolation gamma parameters
        t_a_offset = 0 #number of days minimum to isolation
        t_a_shape = 1
        t_a_scale = 1

        t_p_shape = 1
        t_p_scale = 1
        t_p_offset = 0.5    
            
        t_t_shape = 1
        t_t_scale = 0.5
        t_t_offset =0.5

        t_n_shape = 1
        t_n_scale = 0.5
        t_n_offset = 0

        generations_traced = 2
        test_capacity = 2000000
        trace_capacity = 200000

        sim_undetected = False
        #number of iterations
        n = 20000
        time_end = 30
        end_time = time_end
        state = 'NSW'
        case_file_date = None #'24Jul'
        #Reff_file_date = '2020-07-20'
        Reff_file_date = '2020-08-06'
        #Number of initial, detected asymptomatic and symptomatic cases respectively
        initial_cases = [1,5]

        local_detection = {
                    'NSW':0.5, #0.8 #0.2 #0.556,#0.65,
                }

        a_local_detection = {
                    'NSW':0.1,#0.556,#0.65,
                }

        qi_d = {
                    'NSW':0.95,#0.758,
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



        print("Simulating state " +state)


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
        Model = Forecast(
            current[state],
        state,start_date,people,
        alpha_i= 1, k =0.1,gam_list=gam,
        qs_list=qs_prior,qi_list=qi_prior,qa_list=qa_prior,
        qua_ai=1,qua_qi_factor=1,qua_qs_factor=1,
        forecast_R =forecast_type, R_I = R_I,forecast_date=forecast_date,
        cross_border_state=None,cases_file_date=case_file_date,
        ps_list = ps_prior,Reff_file_date=Reff_file_date
        )

        Model.end_time = time_end
        Model.cross_border_seeds = np.zeros(shape=(time_end,n),dtype=int)
        Model.cross_border_state_cases = np.zeros_like(Model.cross_border_seeds)

        Model.num_bad_sims = 0
        Model.num_too_many = 0

        from heapq import heappush, heappop
        from math import ceil
        import gc
        np.random.seed(1)
        Model.num_of_sim = sim
        Model.end_time = end_time

        Model.DAYS = DAYS
        Model.p_c = p_c

        # Laura new times
        ## present times
        Model.t_p_offset = t_p_offset
        Model.t_p_shape = t_p_shape
        Model.t_p_scale = t_p_scale
        ## test times
        Model.t_t_shape = t_t_shape
        Model.t_t_scale = t_t_scale
        Model.t_t_offset = t_t_offset
        ## notification times
        Model.t_n_shape = t_n_shape
        Model.t_n_scale = t_n_scale
        Model.t_n_offset = t_n_offset
        ## action times for contacts
        Model.t_a_shape = t_a_shape
        Model.t_a_scale = t_a_scale
        Model.t_a_offset = t_a_offset

        Model.generations_traced = generations_traced

        Model.tests_todo=0
        Model.tracing_todo=0
        Model.test_capacity = test_capacity
        Model.trace_capacity = trace_capacity // Model.generations_traced
        #generate storage for cases
        Model.cases = np.zeros(shape=(end_time, 3),dtype=float)
        Model.observed_cases = np.zeros_like(Model.cases)

        Model.observed_cases[0,:] = Model.initial_state.copy()
            #Model.read_in_cases()
        Model.max_backcast_cases = 1000000
        Model.max_cases = Model.max_backcast_cases
        Model.t_plus_n = np.zeros(Model.end_time)
        Model.n_COP = np.zeros(Model.end_time)

        #Initalise undetected cases and add them to current
        Model.initialise_sim(sim_undetected=sim_undetected)
        #number of cases after end time
        Model.cases_after = 0 #gets incremented in generate new cases

        #Record day 0 cases
        Model.cases[0,:] = Model.current.copy() 
        #Create queue for infected people
        Model.infected_queue = []


        self.Model = Model
    

    def tearDown(self):
        self.Model.reset_to_start(self.Model.initial_people)


    def test_numcases(self):
        self.Model.generate_new_cases(1,10, 100,travel=False)
        self.assertEqual(len(self.Model.infected_queue), 7, 
        msg="Number of offspring generated has changed")


        ##Person 6 should have inherited action time of 1 as they were traced
        #self.assertAlmostEqual(
        #    self.Model.people[6].action_time,
        #    self.Model.people[1].action_time,
        #    msg="Traced offspring did not inherit action time" )
        
        
        #self.Model.generate_new_cases(6, 5,100, travel=False)
        #self.assertEqual(len(self.Model.infected_queue), 4,
        #msg="Offspring not isolated and continued to infect others")

    def test_isolation(self):
        parent = 5
        self.Model.generate_new_cases(parent,51,100,travel=False )

        #See if there are any cases after index cases isolation
        any_after = False
        for key, person in self.Model.people.items():
            if person.parent == parent:
                if person.infection_time > 0:
                    any_after= True
            if any_after==True:
                print("{} occurred after initial cases notification time".format(key))
                break
        self.assertFalse(any_after,msg="Offspring occured after isolation")

        #Find an offspring who was traced and 
        # has a tracing window that overlaps with before they were notified
        # and isolated
        for i in range(parent, len(self.Model.people.keys())):
            if self.Model.people[i].detected==2:
                if self.Model.people[i].symp_onset_time-self.Model.DAYS< self.Model.people[parent].action_time:
                    offspring = i
                    break
        else:
            print("No offspring was not detected and then traced")

        self.Model.generate_new_cases(offspring,10,100,travel=False )
        #check action time is inherited past one generation
        for i in self.Model.people.keys():
            if self.Model.people[i].parent==offspring:
                if self.Model.people[i].detected==3:
                    self.assertGreaterEqual(
                        self.Model.people[i].action_time, 
                        self.Model.people[offspring].action_time,
                        msg = "Action time must increase over generations"
                        )
                
        #check in third generation no cases occured after isolation
        for key, person in self.Model.people.items():
            if person.parent == offspring:
                if person.infection_time > self.Model.people[offspring].action_time:
                    any_after= True
            if any_after==True:
                print("{} occurred after initial cases isolation time".format(key))
                break
        self.assertFalse(any_after,msg="Second gen offspring occured after isolation")

if __name__ =="__main__":
    unittest.main() 