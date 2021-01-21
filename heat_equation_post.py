import matplotlib.pyplot as plt
import numpy as np
import heat_equation_solver as hes
import os


#
#run multiple different timesteps
max = 1e-5
min = 1e-6
list_explicit = 10**np.linspace(np.log10(min), np.log10(max), 10)

max = 1e-3
min = 1e-5
list_ADI = 10**np.linspace(np.log10(min), np.log10(max), 10)

for t in range(len(list_ADI)):
    fun = hes.heat_equation(0.025, 1000, list_ADI[t], solve = True, solver = 'ADI', output_dir=f'results/ADI_1000-{t}')
    # fun = hes.heat_equation(0.025, 100, list_explicit[t], solve = True, solver = 'explicit', output_dir=f'results/explicit_100-{t}')


analytic = hes.heat_equation(0.025, 500, 0.001, solve = True, solver = 'ADI', output_dir=f'analytic')
# analytic.produce_grid()
analytic.solve_grid.get_analytic(nFourier = 1001)


T_anal =  analytic.solve_grid.T_analytic


time_explicit = []
err_explicit = []
time_ADI = []
err_ADI = []

for i in range(10):
    fig, ax = plt.subplots(1)
    # explicit = np.loadtxt(f'results/explicit_100-{i}/T_profile_final.txt')
    # explicit2 = np.loadtxt(f'results/explicit_100-{i}/Meta_data.txt')
    # err_explicit.append(hes.resid(T_anal, explicit))
    # time_explicit.append(explicit2[3])
    ADI = np.loadtxt(f'results/ADI_500-{i}/T_profile_final.txt')
    ADI2 = np.loadtxt(f'results/ADI_500-{i}/Meta_data.txt')
    err_ADI.append(hes.resid(T_anal, ADI))
    time_ADI.append(ADI2[3])

    p = ax.imshow(np.array(T_anal)-np.array(ADI))

    fig.colorbar(p, ax=ax)
    plt.show()
    # ax[0].imshow(T_anal)
    # ax[1].imshow(explicit)
    # ax[2].imshow(ADI)
    # plt.show()
    # plt.close()





fig, ax  = plt.subplots( 1)
ax.plot(time_explicit, err_explicit, 'ko')
ax.plot(time_ADI, err_ADI, 'go')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()





