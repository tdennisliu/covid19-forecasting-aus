from itertools import product
import pandas as pd

DAYS = (-3,-2)
detect = (0.3,0.5,0.8)
gen_traced = (1,2)
initial_cases = (1,2,10)
#Reff_factor = (0.55, 0.75)


inputs = product(DAYS, gen_traced, detect, initial_cases)

csv = list(inputs)
#extra_0 = list(product([0], [0],detect,initial_cases ))

#csv.extend(extra_0)

csv = pd.DataFrame(list(csv))

csv.to_csv("./model/contact_tracing/inputs.csv",header=False, index=False)