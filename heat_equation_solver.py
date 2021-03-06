import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import scipy.sparse as scs
import scipy.sparse.linalg as linalg

def resid(T1, T2):
    return np.max(np.max(((np.array(T1)-np.array(T2))**2)))

def analytic_solution(x, y, t, nFourier=101):
    """
    Calculate the analytic solution for the test problem. (Sum would need to go infinity of course).
    :param x: x position
    :param y: y position
    :param t: time
    :return: solution for the temperature of position (x,y) at time t.
    """
    ns = np.arange(1, nFourier, 2) * np.pi
    fexp = np.exp(-ns**2 * t)
    coeff = 4/ns * fexp

    fx = coeff * np.sin(ns*x)
    fy = coeff * np.sin(ns*y)

    sol = np.einsum('i,j->', fx, fy)

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
        # self.T_history = [self.T] This is not a good idea for larger gridsizes
        self.T_analytic = None

    def get_analytic(self, nFourier = 101):
        self.T_analytic = [[1 for k in range(self.N)] for l in range(self.N)]
        for j in range(self.N ** 2):
            self.T_analytic[j % self.N][int(j / self.N)] = analytic_solution(self.position_x[j],
                                                                  self.position_y[j], self.T.time, nFourier = nFourier)


class heat_equation:
    """
    class to solve the heat equation with different solver.
    """
    def __init__(self, T_end: float, N: int, dt: float, solver: str=  'explicit', output_dir: str = 'results',
                 show_plots: bool =False, show_steps: int = 1,  save_plots: bool = False, save_steps: int = 1,
                 solve: bool = True, show_at_end: bool= False, save_at_end: bool = False, save_video: bool = False, Tinit: list = None, v = 1):
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
        :param v: error reduction parameter
        """
        self.T_end = T_end
        self.N = N
        self.dxy = 1/(N-1)
        self.solver = solver
        if dt >= 1/(4*N**2) and self.solver == 'explicit':
            print('timestep to small. Explicit solver not stable')
            dt = 1/(4*N**2)
            print(f'set timestep to max stable value: {dt}')

        self.num_timesteps = int(T_end/dt + 1)
        self.num_timesteps = self.num_timesteps + self.num_timesteps%2
        self.dt = T_end/(int(self.num_timesteps))
        self.rho = self.dxy**2/self.dt
        print('Adjust timestep to reach exact each:')
        print(f'timestep before:{dt},    timestep now: {self.dt},     num_timesteps: {self.num_timesteps}')
        self.grid = None
        self.t0 = 0
        self.Tinit = Tinit
        self.step = 0
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
        self.v = v

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

    def _compute_implicit_params(self):
        rho = self.rho
        N = self.num_timesteps
        alpha = (4+rho-np.sqrt(8*rho+rho**2+4*np.pi**2/(N**2)))/8
        K = 1-((4+rho)*np.sqrt(8*rho+rho**2+4*np.pi**2/(N**2)) - (8*rho+rho**2))/8
        eta = self.v/np.log10(1.0/K)
        return alpha, round(eta)

    def solve(self):
        print('Start solving')
        if self.solver == 'explicit':
            for i in tqdm(range(self.num_timesteps)):
                self.take_step_explicit(save_plot=self.save_plots, show_plot=self.show_plots)
        if self.solver == 'ADI':
            for i in tqdm(range(self.num_timesteps)):
                self.take_step_ADI(save_plot=self.save_plots, show_plot=self.show_plots)
        if self.solver == 'implicit_steindl':
            self.get_A_implicit_steindl()
            for i in tqdm(range(self.num_timesteps)):
                self.take_step_implicit_steindl(save_plot=self.save_plots, show_plot=self.show_plots)
        if self.solver == 'implicit':
            alpha, eta = self._compute_implicit_params()
            print("alpha = "+str(alpha))
            print("eta = "+str(eta))
            for i in tqdm(range(self.num_timesteps)):
                self.take_step_implicit(alpha, eta, save_plot=self.save_plots, show_plot=self.show_plots)
        print(f'Finished at age of: {self.solve_grid.T.time}')


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
        # self.solve_grid.T_history.append(self.solve_grid.T)
        if save_plot or show_plot:
            self.plot()
            if save_plot and self.step %self.save_steps == 0:
                plt.savefig(self.output_dir + '/t_Profile_' + '{:04d}'.format(self.fig_number) + '.png')
                self.fig_number = self.fig_number + 1
            if show_plot and self.step %self.show_steps == 0:
                plt.show()
            plt.close()

    def take_step_implicit(self, alpha, eta, save_plot = False, show_plot = False):
        self.step = self.step + 1
        Told = self.solve_grid.T.vals.copy()
        T = self.solve_grid.T.vals.copy()
        for n in range(eta):
            T, Told = Told, T #deliberately not copying but swapping names
            for i in range(1, self.N-1):
                for j in range(1, self.N-1):
                    T[i][j] = Told[i][j] + alpha * (Told[i-1][j] + Told[i+1][j] + Told[i][j-1] + Told[i][j+1] - (4+self.rho)*Told[i][j] + self.rho*Told[i][j])

        self.solve_grid.T = T_profile(T, self.dt*self.step)
            # self.solve_grid.T_history.append(self.solve_grid.T)
        if save_plot or show_plot:
            self.plot()
            if save_plot and self.step %self.save_steps == 0:
                plt.savefig(self.output_dir + '/t_Profile_' + '{:04d}'.format(self.fig_number) + '.png')
                self.fig_number = self.fig_number + 1
            if show_plot and self.step %self.show_steps == 0:
                plt.show()
            plt.close()

    def get_A_implicit_steindl(self):
        n = (self.N-2) ** 2
        ones = np.ones(n)
        holesU = np.ones(n)
        holesL = np.ones(n)
        holesU[::self.N-2] = 0
        holesL[self.N-3::self.N-2] = 0
        data = np.array([ones, holesL, -(4 + self.rho)*ones, holesU, ones])
        offsets = np.array([-self.N+2, -1, 0, 1, self.N-2])
        sparse = scs.dia_matrix((data, offsets), shape=(n, n))

        self.A_implicit_steindl = sparse


    def take_step_implicit_steindl(self, save_plot=False, show_plot=False):
        self.step = self.step + 1
        T = self.solve_grid.T.vals.copy()
        b = np.zeros((self.N-2) ** 2)
        for i in range((self.N-2)):
            for j in range((self.N-2)):
                b[i + (self.N-2) * j] = -self.rho * T[i+1][j+1]

        self.A_implicit_steindl = scs.csr_matrix(self.A_implicit_steindl)
        T_new_vector = linalg.spsolve(self.A_implicit_steindl, b)

        for i in range( (self.N-2) ):
            for j in range( (self.N-2) ):
                T[i+1][j+1] = T_new_vector[i + (self.N-2) * j]

        self.solve_grid.T = T_profile(T, self.dt * self.step)
        # self.solve_grid.T_history.append(self.solve_grid.T)
        if save_plot or show_plot:
            self.plot()
            if save_plot and self.step % self.save_steps == 0:
                plt.savefig(self.output_dir + '/t_Profile_' + '{:04d}'.format(self.fig_number) + '.png')
                self.fig_number = self.fig_number + 1
            if show_plot and self.step % self.show_steps == 0:
                plt.show()
                plt.close()


    def take_step_ADI(self, save_plot=False, show_plot=False):
        self.step = self.step + 1
        Told = self.solve_grid.T.vals.copy()

        T_new = self.solve_grid.T.vals.copy()

        if self.step%2 == 0:
            a = [1.0 for k in range(self.N-2)]
            b = [-(2+ self.rho) for k in range(self.N-2)]
            c = [1.0 for k in range(self.N-2)]
            for i in range(1, self.N - 1):
                T_i = Told[i].copy()
                d = [-T_i[k-1] + (2- self.rho)* T_i[k]- T_i[k+1] for k in range(1, self.N-1)]
                T_new_i = thomas_algorithm(a, b, c, d)
                T_new_i.insert(0, 0)
                T_new_i.append(0)
                T_new[i] = T_new_i.copy()
        else:
            Told = trans(T_new.copy())
            a = [1.0 for k in range(self.N-2)]
            b = [-(2+ self.rho) for k in range(self.N-2)]
            c = [1.0 for k in range(self.N-2)]
            for i in range(1, self.N - 1):
                T_i = Told[i].copy()
                d = [-T_i[k-1] + (2- self.rho)* T_i[k]- T_i[k+1] for k in range(1, self.N-1)]
                T_new_i = thomas_algorithm(a, b, c, d)
                T_new_i.insert(0, 0)
                T_new_i.append(0)
                T_new[i] = T_new_i.copy()
            T_new = trans(T_new.copy())

        self.solve_grid.T = T_profile(T_new, self.dt*self.step)
        # self.solve_grid.T_history.append(self.solve_grid.T)
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




# # dfs
# fun = heat_equation(0.002, 300, 0.00001, solver = 'implicit_steindl', show_at_end=True)
# fun.solve_grid.get_analytic()
# resid =  [[1 for k in range(fun.N)] for l in range(fun.N)]
# for j in range(fun.N):
#     for k in range(fun.N):
#         resid[j][k] = fun.solve_grid.T_analytic[j][k] - fun.solve_grid.T.vals[j][k]
#
# fig, ax = plt.subplots(1,3, figsize=(13, 3))
#
# p1 = ax[0].imshow(fun.solve_grid.T.vals, extent=[0, 1, 0, 1], vmin=0, vmax=1)
# fig.colorbar(p1, ax=ax[0])
# ax[0].set_title(f'implicit solver')
#
# p2 = ax[1].imshow(fun.solve_grid.T_analytic, extent=[0, 1, 0, 1], vmin=0, vmax=1)
# fig.colorbar(p2, ax=ax[1])
# ax[1].set_title(f'analytic solution')
#
# p3 = ax[2].imshow(resid, extent=[0, 1, 0, 1], vmin=-0.01, vmax=0.01)
# fig.colorbar(p3, ax=ax[2])
# ax[2].set_title(f'residuals')
#
# plt.show()


##main and plotting for implicit
# fun = heat_equation(T_end=0.025, N=101, dt=0.0001, solve = True, solver = 'implicit', output_dir='implicit_plot', v = 1)
# fun.solve_grid.get_analytic()

# resid =  [[1 for k in range(fun.N)] for l in range(fun.N)]
# for j in range(fun.N):
    # for k in range(fun.N):
        # resid[j][k] = fun.solve_grid.T_analytic[j][k] - fun.solve_grid.T.vals[j][k]


# fig, ax = plt.subplots(1,3, figsize=(13, 3))

# p1 = ax[0].imshow(fun.solve_grid.T.vals, extent=[0, 1, 0, 1], vmin=0, vmax=1)
# fig.colorbar(p1, ax=ax[0])
# ax[0].set_title(f'implicit solver')

# p2 = ax[1].imshow(fun.solve_grid.T_analytic, extent=[0, 1, 0, 1], vmin=0, vmax=1)
# fig.colorbar(p2, ax=ax[1])
# ax[1].set_title(f'analytic solution')

# p3 = ax[2].imshow(resid, extent=[0, 1, 0, 1], vmin=-0.1, vmax=0.1)
# fig.colorbar(p3, ax=ax[2])
# ax[2].set_title(f'residuals')
# plt.savefig('implicit.jpg')
# # plt.show()

# print("maxerr:")
# print(np.max(np.abs(resid)))




