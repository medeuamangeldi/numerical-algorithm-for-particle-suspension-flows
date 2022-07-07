# To work with arrays and optimization
import numpy as np
from scipy.optimize import root
from scipy import integrate

# To read outside data
import pandas as pd

# Plotting
from pylab import *
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
from labellines import labelLine, labelLines
from matplotlib.lines import Line2D

# Other
import copy

# We create a Class object "particle_solve" that will store all inputs and methods. This is more useful than just a function as it makes reusability simpler.
# For instance, for every set of parameters, we can create the objects that stores everything related to the inputs. We then can use those objects for analysis, plotting etc.

class particle_solve:    
    # radius is in meters, pressure gradient dpdz is in Pa/m, power_law is either True or False
    # alpha = K_c/K_Eta
    
    def __init__(self, radius, alpha, dpdz, phi_avg, power_law=False, sep=False, options = {}):
        self.R, self.alpha, self.dpdz, self.avg_phi, self.sep, self.power_law = radius, alpha, dpdz, phi_avg, sep, power_law

        if power_law: 
            self.phi_m, self.n = options['phi_m'], options['n']

        if self.sep:
            def visc_func(s, phi, gamma_dot_w):
                return options['eta_r'](phi)*options['eta_s'](s, gamma_dot_w)
            self.visc_func = visc_func
            self.eta_r, self.eta_s = options['eta_r'], options['eta_s']
        else:
            self.visc_func = options['visc_func']
        
            # The default boolean parameter of "power_law" is False. If True, we create new variables "phi_m" amd "n". 
            # These are also defined in "visc_func", but since the numerical approach is different for power_law model, we need to define these parameters outside of the function.
    """        
    Define the methods
    """
    def eta_w(self, sr_w, phi_w):  #viscosity function at the wall
        return self.visc_func(1, phi_w, sr_w) 

    def nond_visc(self, s, phi, sr_w, phi_w): #nondimensiolized viscosity function
        return self.visc_func(s, phi, sr_w)/self.eta_w(sr_w, phi_w)

    def find_sr_w(self, sr_w, phi_w):  #residual function to find the shear rate at the wall by root finding method
        return self.eta_w(sr_w, phi_w)*sr_w - self.dpdz/2*self.R
    
    def solve_sr_w(self, phi_w): # find the root of the resiudal equation to get the wall shear rate

        sr_w_guess = 0.01

        sol_sr_w = root(lambda sr_w: self.find_sr_w(sr_w, phi_w), [sr_w_guess], method = 'lm').x[0]

        return sol_sr_w
    
    def solve(self, N, phi_w): #solve for phi and r (phi and s if power law) given wall particle volume fraction (phi_w)
        
        self.sr_w = self.solve_sr_w(phi_w)
        
        if self.power_law:
            r_vector = np.concatenate((np.linspace(0, 0.001, N), np.linspace(0.0011, 1, N))) # this is the trick: we create 100 bins between 0 and 0.001 and another 1000 between 0.0011 and 1;
            s_vector = np.linspace(0,1, 2*N)
        else:
            s_vector = np.concatenate((np.linspace(0, 0.001, N), np.linspace(0.0011, 1, N))) # if not power law used, this trick is applied to the array "s" (nond shear rate)
            r_vector = np.linspace(0,1, 2*N)

        N = 2*N

        if self.power_law:
            phi_vector = np.linspace(self.phi_m, phi_w, N)
        else:
            phi_vector = np.ones(N)*phi_w

        result_vector = np.zeros(2)

        j = N-2

        while j > 0:
            def equation(phi):
                
                if self.power_law:
                    s1 = r_vector[j]**(1/(1-self.alpha))*(phi/phi_w)**(self.alpha/(1-self.alpha))
                    s2 = r_vector[j]**(1/self.n)*((self.phi_m-phi)/(self.phi_m-phi_w))**(1.82/self.n)

                    return s1-s2
                    
                else:
                    r1 = s_vector[j]**(1-self.alpha)*phi_w**self.alpha/(phi**self.alpha)
                    r2 = self.nond_visc(s_vector[j], phi, self.sr_w, phi_w)*s_vector[j]
                    
                    return r1-r2
            
            result_vector[1] = root(lambda phi: equation(phi), [phi_vector[j+1]], method = 'lm').x[0] #root finding using "lm" method (Levenberg-Marquardt). This finds phi.
            
            if self.power_law:
                result_vector[0] = r_vector[j]**(1/(1-self.alpha))*(result_vector[1]/phi_w)**(self.alpha/(1-self.alpha)) # Here we use that phi value above to find the value of "r" by back-substituting to one of the equations.
                s_vector[j] = result_vector[0]

            else:
                result_vector[0] = s_vector[j]**(1-self.alpha)*phi_w**self.alpha/(result_vector[1]**self.alpha) # if power law is used, we back-substitute to find the "s". 
                r_vector[j] = result_vector[0]
            
            phi_vector[j] = result_vector[1]
            
            j -= 1
        
        self.r_sol = r_vector
        self.phi_sol = phi_vector
        self.s = s_vector

        return self.r_sol, self.phi_sol, self.s, self.sr_w
    
    def solver(self, N):
        
        interval = np.zeros(2)
        interval[1] = self.avg_phi*1.7

        tol = 1E-12

        err = 1

        i = 0

        while err >= tol:
            mid = (interval[0] + interval[1])/2
            
            r, phi, s, srw = self.solve(N, mid)

            est_avg_phi = integrate.simpson(np.multiply(phi, r), r, even = "avg")*2

            err = abs(self.avg_phi - est_avg_phi)
            i += 1
            
            M = 500
            if i > M:
                print(f'Maximum iterations of {M} has been reached.')
                break
            elif err < tol:
                u = np.zeros(len(s))

                for i in range(len(r)):
                    u[i] = integrate.simpson(s[i:], r[i:], even = "avg")

                u_avg = integrate.simpson(np.multiply(u, r), r, even = "avg")*2

                # generate: radius, particle volume fraction, nond shear rate, nond velocity, nond avg velocity, wall shear rate, wall particle volume fraction
                # here each data is extracted depending on their variable name. For instance: .r: radius; .phi: phi values; .s: s values; .u: nondimensional velocity; .u_avg: nondimensional average velocity;
                # .srw: wall shear rate; .phi: wall particle volume fraction; .R: maximum radius; etc.
                self.r, self.phi, self.s, self.u, self.u_avg, self.srw, self.phiw = r, phi, s, u, u_avg, srw, mid

            else:
                if est_avg_phi > self.avg_phi:
                    interval[1] = mid
                else:
                    interval[0] = mid