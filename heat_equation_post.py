import matplotlib.pyplot as plt
import numpy as np

T_ADI = np.loadtxt('results-1/T_profile_final.txt')
T_explicit =np.loadtxt('results/T_profile_final.txt')

print(np.max(T_explicit-T_ADI))