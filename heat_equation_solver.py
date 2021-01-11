import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

def analytic_solution(x, y, t):
    sol = 0
    for n in range(1, 101)[::2]:
        for m in range(1, 101)[::2]:
            sol += 16/(m*n*np.pi**2)*np.sin(n*np.pi*x)*  np.sin(m*np.pi*y) * np.exp(-(m**2 + n**2)*np.pi**2*t)
            # print(m, n, 8/(m*n*np.pi**2)*np.sin(n*np.pi*x) * np.sin(n*np.pi*y) * np.exp(-(m**2 + n**2)*np.pi**2*t)  )
    return sol


def thomas_algorithm(a_i: list, b_i: list, c_i: list, d_i: list) -> list:
    """
    Solve a system of linear equation defined with a tridiagonal matrix by the use of the Thomas algorithm.
    :param a_i: lower diagonal elements.
    :param b_i: diagonal elements.
    :param c_i: upper diagonal elements.
    :param d_i: right hand side.
    :return: solution to the system of linear equation
    """
    cp_i = [c_i[0]/b_i[0]]
    dp_i  =[d_i[0]/b_i[0]]
    for k in range(1, len(a_i)):
        cp_i.append(c_i[k] / (b_i[k] - a_i[k]*cp_i[k-1]))
        dp_i.append((d_i[k] - a_i[k]*dp_i[k-1]) / (b_i[k] - a_i[k]*cp_i[k-1]))

    x_i = [0.0 for k in range(len(a_i))]
    x_i[len(a_i)-1]= dp_i[len(a_i)-1]

    for k in range(len(a_i)-2, -1, -1):
        x_i[k] = dp_i[k]-cp_i[k]*x_i[k+1]

    return x_i

def trans(M: list) -> list:
    """
    Transpose the matrix defined as list.
    :param M: matrix.
    :return: transposed matrix.
    """
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


class T_profile:
    def __init__(self, T: list, time: float):
        """
        temperature profile for a given timestep.
        :param T: temperature profule.
        :param time: time of evolution.
        """
        self.time = time
        self.vals = T

class grid:
    """
    object that holds the grid on wich we solve the heat equation.
    """
    def __init__(self, N: int, Tinit: list = None ):
        """

        :param N: number of steps to divide the grid in x and y direction, respectively.
        :param Tinit: initial temperature profile. If nothing given, we take the test case from the paper.
        """
        self.N = N
        self.dxy = 1/(N-1)
        self.position_x = [ self.dxy*(k%N) for k in range( self.N**2) ]
        self.position_y = [ self.dxy*int(k/N) for k in range( self.N**2) ]
        if Tinit == None:
            T = [[ 1 for k in range( self.N ) ] for l in range( self.N ) ]
        else:
            T = Tinit
        for i in range(N):
            T[i][0] = 0
            T[i][N-1] = 0
            T[0][i] = 0
            T[N-1][i] = 0
        self.T = T_profile(T, 0)
        self.T_history = [self.T]


class heat_equation:
    """
    class to solve the heat equation with different solver.
    """
    def __init__(self, T_end: float, N: int, dt: float, solver: str=  'explicit', output_dir: str = 'results',
                 show_plots: bool =False, show_steps: int = 1,  save_plots: bool = False, save_steps: int = 1,
                 solve: bool = True, show_at_end: bool= False, save_at_end: bool = False, save_video: bool = False, Tinit: list = None):
        """
        initiate the solver class.
        :param T_end: final time.
        :param N: number of subdivision in x in y direction.
        :param dt: timestep to be taken.
        :param solver: solver to to use, Default: explicit, One of: explicit, ADI.
        :param output_dir: directory to save the results in, Default: results,
        :param show_plots: if True, plots are shown during the solving, Default: False.
        :param show_steps: if show_plots is true, plots are shown in this step interval, Default: 1.
        :param save_plots:if True, plots are saved during the solving, Default: False.
        :param save_steps:if show_plots is true, plots are saved in this step interval, Default: 1.
        :param solve: if true, the heat equation will be solved after initiating the class, Default: True.
        :param show_at_end: if True, show a plot at the end of the solving process, Default: false.
        :param save_at_end: if True, save a plot at the end of the solving process, Default: false.
        :param save_video: if plots are saved, and this is True, produce a video of all saved plots.
        :param Tinit: initial temperature profile. If nothing given, we take the test case from the paper.
        """
        self.T_end = T_end
        self.N = N
        self.dxy = 1/(N-1)
        self.dt = dt
        self.solver = solver
        if self.dt >= 1/(4*N**2) and self.solver == 'explicit':
            print('timestep to small. Explicit solver not stable')
            self.dt = 1/(4*N**2)
            print(f'set timestep to max stable value: {self.dt}')
        self.rho = self.dxy**2/self.dt
        self.grid = None
        self.t0 = 0
        self.Tinit = Tinit
        self.step = 0
        self.num_steps = int(T_end/self.dt)
        self.fig_number = 1
        self.output_dir = output_dir
        if os.path.isdir(self.output_dir):
            go = True
            extra =1
            while go:
                if not os.path.isdir(self.output_dir + f'-{extra}'):
                    self.output_dir =self.output_dir + f'-{extra}'
                    go = False
                else:
                    extra = extra + 1
        os.mkdir(self.output_dir)
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.show_steps  = show_steps
        self.save_steps = save_steps

        if solve:
            self.produce_grid()
            if self.save_plots or self.show_plots:
                self.plot()
                if self.save_plots and self.step % self.save_steps == 0:
                    plt.savefig(self.output_dir + '/t_Profile_' + '{:04d}'.format(self.fig_number) + '.png')
                    self.fig_number = self.fig_number + 1
                if self.show_plots and self.step % self.show_steps == 0:
                    plt.show()
                plt.close()
            self.solve_time = time.time()
            self.solve()
            self.solve_time = -(self.solve_time - time.time())
            if save_at_end or show_at_end:
                self.plot()
                if save_at_end and self.step % self.save_steps == 0:
                    plt.savefig(self.output_dir + '/final_solution.png')
                if show_at_end and self.step % self.show_steps == 0:
                    plt.show()
                plt.close()
            if save_video:
                self.produce_video()
            self.save_results()


    def produce_video(self):
        print('Producing video')
        os.system(f'ffmpeg -framerate 25 -i {self.output_dir}//t_Profile_%04d.png -vf format=yuv420p  {self.output_dir}//output.mp4')

    def produce_grid(self):
        print('Initialising Grid')
        self.solve_grid = grid(self.N, self.Tinit)

    def solve(self):
        print('Start solving')
        for i in tqdm(range(self.num_steps)):
            if self.solver == 'explicit':
                self.take_step_explicit(save_plot=self.save_plots, show_plot=self.show_plots)
            if self.solver == 'ADI':
                self.take_step_ADI(save_plot=self.save_plots, show_plot=self.show_plots)

    def save_results(self):
        np.savetxt(self.output_dir + '/T_profile_final.txt', np.array(self.solve_grid.T.vals) )
        np.savetxt(self.output_dir + '/Meta_data.txt', np.array([self.N, self.dt, self.dxy, self.solve_time]))

    def take_step_explicit(self, save_plot = False, show_plot = False):
        self.step = self.step + 1
        Told = self.solve_grid.T.vals.copy()
        T = self.solve_grid.T.vals.copy()
        for i in range(1, self.N-1):
            for j in range(1, self.N-1):
                T[i][j] = Told[i][j] + (Told[i-1][j] + Told[i+1][j] + Told[i][j+1]+ Told[i][j-1]- 4* Told[i][j])/self.rho
        self.solve_grid.T = T_profile(T, self.dt*self.step)
        self.solve_grid.T_history.append(self.solve_grid.T)
        if save_plot or show_plot:
            self.plot()
            if save_plot and self.step %self.save_steps == 0:
                plt.savefig(self.output_dir + '/t_Profile_' + '{:04d}'.format(self.fig_number) + '.png')
                self.fig_number = self.fig_number + 1
            if show_plot and self.step %self.show_steps == 0:
                plt.show()
            plt.close()

    def take_step_ADI(self, save_plot=False, show_plot=False):
        self.step = self.step + 1
        Told = self.solve_grid.T.vals.copy()

        T_new = self.solve_grid.T.vals.copy()

        if self.step%2 == 0:
            for i in range(1, self.N - 1):
                T_i = Told[i].copy()
                a = [1.0 for k in range(self.N-2)]
                b = [-(2+ self.rho) for k in range(self.N-2)]
                c = [1.0 for k in range(self.N-2)]
                d = [-T_i[k-1] + (2- self.rho)* T_i[k]- T_i[k+1] for k in range(1, self.N-1)]
                T_new_i = thomas_algorithm(a, b, c, d)
                T_new_i.insert(0, 0)
                T_new_i.append(0)
                T_new[i] = T_new_i.copy()
        else:
            Told = trans(T_new.copy())

            for i in range(1, self.N - 1):
                T_i = Told[i].copy()
                a = [1.0 for k in range(self.N-2)]
                b = [-(2+ self.rho) for k in range(self.N-2)]
                c = [1.0 for k in range(self.N-2)]
                d = [-T_i[k-1] + (2- self.rho)* T_i[k]- T_i[k+1] for k in range(1, self.N-1)]
                T_new_i = thomas_algorithm(a, b, c, d)
                T_new_i.insert(0, 0)
                T_new_i.append(0)
                T_new[i] = T_new_i.copy()
            T_new = trans(T_new.copy())

        self.solve_grid.T = T_profile(T_new, self.dt*self.step)
        self.solve_grid.T_history.append(self.solve_grid.T)
        if save_plot or show_plot:
            self.plot()
            if save_plot and self.step %self.save_steps == 0:
                plt.savefig(self.output_dir + '/t_Profile_' + '{:04d}'.format(self.fig_number) + '.png')
                self.fig_number = self.fig_number + 1
            if show_plot and self.step %self.show_steps == 0:
                plt.show()
            plt.close()


    def plot(self):
        fig, ax = plt.subplots(figsize = (5,5))
        ax.imshow(self.solve_grid.T.vals, extent = [0,1, 0, 1], vmin = 0, vmax = 1)
        ax.set_title(f'step: {self.step}     time :' + '{:06.5f}'.format(self.dt * self.step, 7))

# sol = analytic_solution(0.1, 0.5,0)
# print(sol)
fun = heat_equation(0.025, 51, 0.00001, solve = True, solver = 'ADI', output_dir='ADI_51_min')
# print(fun.solve_grid.T.vals)

T_anal = [[1 for k in range(fun.N)] for l in range(fun.N)]


for j in range(fun.N**2):
        T_anal[j%fun.N][int(j/fun.N)] = analytic_solution(fun.solve_grid.position_x[j], fun.solve_grid.position_y[j], 0.025)
        # print(fun.solve_grid.position_x[j], fun.solve_grid.position_y[j])

resid =  [[1 for k in range(fun.N)] for l in range(fun.N)]
for j in range(fun.N):
    for k in range(fun.N):
        resid[j][k] = T_anal[j][k] - fun.solve_grid.T.vals[j][k]

print(fun.solve_grid.T.vals)
print(T_anal)
print(resid)


fig, ax = plt.subplots(3,1, figsize=(10, 10))
ax[0].imshow(fun.solve_grid.T.vals, extent=[0, 1, 0, 1], vmin=0, vmax=2)
ax[0].set_title(f'ADI solver')
ax[1].imshow(T_anal, extent=[0, 1, 0, 1], vmin=0, vmax=2)
ax[1].set_title(f'analytic solver')
ax[2].imshow(resid, extent=[0, 1, 0, 1], vmin=-0.05, vmax=0.05)
ax[2].set_title(f'residuals')

plt.show()




