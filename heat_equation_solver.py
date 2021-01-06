import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

def thomas_algorithm(a_i, b_i, c_i, d_i):
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

def trans(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


class T_profile:
    def __init__(self, T, time):
        self.time = time
        self.vals = T

class grid:
    def __init__(self, N, Tinit = False):
        self.N = N
        self.dxy = 1/(N-1)
        self.position_x = [ self.dxy*(k%N) for k in range( self.N**2) ]
        self.position_y = [ self.dxy*int(k/N) for k in range( self.N**2) ]
        if not Tinit:
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
    def __init__(self, T_end, N, dt, solver = 'explicit', output_dir = 'results', show_plots =False, show_steps = 1,  save_plots = False, save_steps = 1,  solve = True, show_at_end= False, save_at_end = False, save_video = False, Tinit = None):
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
        # os.system(f'ffmpeg -r 1/30 -i {self.output_dir}/t_Profile_%04d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {self.output_dir}/out.mp4')
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

    def take_step_ADI_old(self, save_plot = False, show_plot = False):
        Told = self.solve_grid.T.vals.copy()
        T_new = self.solve_grid.T.vals.copy()
        for i in range(1, self.N-1):
            T_i = Told[i].copy()

            D = [(2-self.rho)*T_i[1] - T_i[2]]
            w = [-(2+self.rho)]
            b  = [1/w[0]]
            g = [D[0]/w[0]]

            for j in range(1, self.N-2):
                w.append(-(2+self.rho) - b[j-1])
                b.append(1/w[j])
                D.append(-T_i[j-1] + (2-self.rho)*T_i[j] - T_i[j+1])
                g.append((D[j]-g[j-1])/w[j])

            T_new_i = [g[-1]]
            for k in range(self.N - 3, 0, -1):
                T_new_i.append(g[k]- b[k]*T_new_i[-1])


            T_new_i = T_new_i[::-1]

            T_new_i.insert(0, 0)
            T_new_i.append(0)
            T_new[i] = T_new_i.copy()

        Told = T_new

        for j in range(1, self.N - 1):
            T_j = trans(Told.copy())[j]


            D = [(2 - self.rho) * T_j[1] - T_j[2]]
            w = [-(2 + self.rho)]
            b = [1 / w[0]]
            g = [D[0] / w[0]]

            for i in range(1, self.N - 2):
                w.append(-(2 + self.rho) - b[i-1])
                b.append(1 / w[i])
                D.append(-T_j[i - 1] + (2 - self.rho) * T_j[i] - T_j[i + 1])
                g.append((D[i] - g[i-1]) / w[i])

            T_new_j = [g[-1]]
            for k in range(self.N - 3, 0, -1):
                T_new_j.append(g[k] - b[k] * T_new_j[-1])

            T_new_j = T_new_j[::-1]
            T_new_j.insert(0, 0)
            T_new_j.append(0)
            T_new[j] = T_new_j.copy()
        T_new = trans(T_new)

        self.solve_grid.T = T_profile(T_new, self.dt*self.step)
        self.solve_grid.T_history.append(self.solve_grid.T)


    def plot(self):
        fig, ax = plt.subplots(figsize = (5,5))
        ax.imshow(self.solve_grid.T.vals, extent = [0,1, 0, 1], vmin = 0, vmax = 1)
        ax.set_title(f'step: {self.step}     time :' + '{:06.5f}'.format(self.dt * self.step, 7))



fun = heat_equation(0.1, 51, 0.001, solve = True, solver = 'ADI', output_dir='ADI_51_min', save_plots=True, save_video=True)





