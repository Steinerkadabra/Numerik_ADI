import matplotlib.pyplot as plt
import numpy as np
import heat_equation_solver as hes


#
# T_ADI = np.loadtxt('results-1/T_profile_final.txt')
# T_explicit =np.loadtxt('results/T_profile_final.txt')
#
# print(np.max(T_explicit-T_ADI))


fun = hes.heat_equation(0.025, 101, 0.0001, solve = True, solver = 'explicit', output_dir='ADI_51_min')