import matplotlib.pyplot as plt
import numpy as np
import heat_equation_solver as hes
import os

# hes.heat_equation(0.025, 500, 1e5, solve = True, solver = 'explicit', output_dir=f'shit')
#
#run multiple different timesteps
max = 1e-6
min = 5e-8
list_explicit = 10**np.linspace(np.log10(min), np.log10(max), 5)
#
max = 5e-3
min = 1e-6
list_ADI = 10**np.linspace(np.log10(min), np.log10(max), 100)


max = 1e-4
min = 2e-6
list_implicit = 10**np.linspace(np.log10(min), np.log10(max), 5)

# for t in range(len(list_explicit)):
#     fun = hes.heat_equation(0.0001, 500, list_explicit[t], solve = True, solver = 'explicit', output_dir=f'results/explicit_500')
# #
# for t in range( len(list_ADI)):
#     fun = hes.heat_equation(0.0001, 500, list_ADI[t], solve = True, solver = 'ADI', output_dir=f'results/ADI_500')

# for t in range( len(list_implicit)):
#     fun = hes.heat_equation(0.0001, 1000, list_implicit[t], solve = True, solver = 'implicit_steindl', output_dir=f'results/implicit_1000')


# analytic = hes.heat_equation(0.0001, 1000, 0.00001, solve = True, solver = 'ADI', output_dir=f'analytic')
# # analytic.produce_grid()
# analytic.solve_grid.get_analytic(nFourier = 101)
# np.savetxt('analytics_1000.txt', analytic.solve_grid.T_analytic)
# T_anal =  analytic.solve_grid.T_analytic
T_anal = np.loadtxt('analytics_1000.txt')

time_explicit = []
err_explicit = []
time_ADI = []
err_ADI = []
time_implicit = []
err_implicit = []

for i in range(5):
    # fig, ax = plt.subplots(1)
    explicit = np.loadtxt(f'results/explicit_1000-{i}/T_profile_final.txt')
    explicit2 = np.loadtxt(f'results/explicit_1000-{i}/Meta_data.txt')
    err_explicit.append(hes.resid(T_anal, explicit))
    time_explicit.append(explicit2[3])

for i in range(136):
    try:
        ADI = np.loadtxt(f'results/ADI_1000-{i}/T_profile_final.txt')
    except:
        continue
    ADI2 = np.loadtxt(f'results/ADI_1000-{i}/Meta_data.txt')
    err_ADI.append(hes.resid(T_anal, ADI))
    time_ADI.append(ADI2[3])

for i in range(5):
    # fig, ax = plt.subplots(1)
    implicit = np.loadtxt(f'results/implicit_1000-{i}/T_profile_final.txt')
    implicit2 = np.loadtxt(f'results/implicit_1000-{i}/Meta_data.txt')
    err_implicit.append(hes.resid(T_anal, implicit))
    time_implicit.append(implicit2[3])

# analytic = hes.heat_equation(0.0001, 500, 0.00001, solve = True, solver = 'ADI', output_dir=f'analytic')
# # analytic.produce_grid()
# analytic.solve_grid.get_analytic(nFourier = 101)
# np.savetxt('analytics_500.txt', analytic.solve_grid.T_analytic)
# T_anal =  analytic.solve_grid.T_analytic

T_anal = np.loadtxt('analytics_500.txt')

time_explicit_500 = []
err_explicit_500 = []
time_ADI_500 = []
err_ADI_500 = []
time_implicit_500 = []
err_implicit_500 = []



for i in range(5):
    # fig, ax = plt.subplots(1)
    explicit = np.loadtxt(f'results/explicit_500-{i}/T_profile_final.txt')
    explicit2 = np.loadtxt(f'results/explicit_500-{i}/Meta_data.txt')
    err_explicit_500.append(hes.resid(T_anal, explicit))
    time_explicit_500.append(explicit2[3])

for i in range(5):
    # fig, ax = plt.subplots(1)
    implicit = np.loadtxt(f'results/implicit_500-{i}/T_profile_final.txt')
    implicit2 = np.loadtxt(f'results/implicit_500-{i}/Meta_data.txt')
    err_implicit_500.append(hes.resid(T_anal, implicit))
    time_implicit_500.append(implicit2[3])

for i in range(1, 101):
    ADI = np.loadtxt(f'results/ADI_500_{i}/T_profile_final.txt')
    ADI2 = np.loadtxt(f'results/ADI_500_{i}/Meta_data.txt')
    err_ADI_500.append(hes.resid(T_anal, ADI))
    time_ADI_500.append(ADI2[3])

    # p = ax.imshow(np.array(T_anal)-np.array(ADI))

    # fig.colorbar(p, ax=ax)
    # plt.show()
    # ax[0].imshow(T_anal)
    # ax[1].imshow(explicit)
    # ax[2].imshow(ADI)
    # plt.show()
    # plt.close()



from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='r', lw=4),
                Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='g', lw=4),
                Line2D([0], [0], color='k', marker = 'o', ls = '--', lw=1),
                Line2D([0], [0], color='k', marker = 'o', ls = '--', mfc ='None',lw=1)]




fig, ax  = plt.subplots( 1)
ax.plot(time_explicit, err_explicit, 'ro--')
ax.plot(time_ADI, err_ADI, 'go')
ax.plot(time_explicit_500, err_explicit_500, 'ro--', mfc = 'None')
ax.plot(time_ADI_500, err_ADI_500, 'go', mfc = 'None')
ax.plot(time_implicit_500, err_implicit_500, 'bo--', mfc = 'None')
ax.plot(time_implicit, err_implicit, 'bo--',)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('solver time (s)')
ax.set_ylabel('max resid at t= 0.0001')
ax.legend(custom_lines, ['explicit', 'implicit', 'ADI', 'N = 1000', 'N=500'])
# plt.show()
plt.savefig('work-precision.pdf')





