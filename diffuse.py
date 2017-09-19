""" Module for solving diffusion problems
"""

#import sys
import numpy as np
from scipy import optimize
from scipy.integrate import ode
import matplotlib.pyplot as plt

def MoL_rhs_sphere2solution (t, y, args):
    """ Method of lines for solving 1D diffusion from sphere
    into solution; Note that the spheres have the same size (radius)
    and there can be multiple spheres (particles)
    this is the right hand size (rhs) of the differential equations
    Args:
        t - time
        y - current values
        args - a list containing:
            D - diffusivity (m^2/s)
            R - radius of the sphere (m)
            K_ps - partition coefficient between particle (sphere solid) and solution
            V_solution - volume of the solution
            N_particle - number of particles
    """
    D, R, K_ps, V_solution, N_particle = args
    
    n = len(y)
    dydt = np.zeros(y.shape)
    dr = R/(n-1)
    
    flux = np.zeros(y.shape) # flux from the lower-labelled mesh to higher labelled one
    for i in range(n-1):
        if i == n-2:
            flux[i] = D * (y[i] - y[i+1]*K_ps) / (0.5*dr)
        else:
            flux[i] = D * (y[i] - y[i+1]) / dr
            
    for i in range(n):
        r_i0 = i*dr
        r_i = (i+1) * dr
        if i == 0:
            v_i = 4/3*np.pi*r_i**3
            dydt[i] = -flux[i] * (4*np.pi*r_i**2) / v_i
        elif i == n-1:
            v_i = V_solution
            dydt[i] = flux[i-1] * (4*np.pi*r_i0**2) * N_particle / V_solution
        else:
            v_i = 4/3*np.pi * (r_i**3 - r_i0**3)
            dydt[i] = flux[i-1] * (4*np.pi*r_i0**2) / v_i  -  flux[i] * (4*np.pi*r_i**2) / v_i
    
    #dydt[-1] = 0
    return dydt
    #print(flux)
    

def MoL_rhs_sphere2solution_psd (t, y, args):
    """ Method of lines for solving 1D diffusion from spheres
    into solution; Note that the spheres are not of the same size 
    but their radius has a distribution (particle size distribution, psd)
    this is the right hand size (rhs) of the differential equations
    Args:
        t - time
        y - current values, concentration of spatial positions in each bin,
            the last entry is for concentration in solution
        args - a list containing:
            D - diffusivity (m^2/s)
            R - radius of the sphere (m), an array of average R in each psd bin
            K_ps - partition coefficient between particle (sphere solid) and solution
            V_solution - volume of the solution
            N_particle - number of particles, an array in each psd bin
    """
    
    D, R, K_ps, V_solution, N_particle = args
    
    n_bins = len(R)
    n_y_total = len(y)
    n_y_per_bin = int( ( n_y_total-1 ) / n_bins )
    
    dydt = np.zeros(n_y_total)
    y_tmp = np.zeros(n_y_per_bin+1)
    
    for i in range(n_bins):
        st = i*n_y_per_bin
        ed = (i+1)*n_y_per_bin
        idx = np.append( np.arange(st, ed), -1 )
        #print(st, ed)
        #print(idx)
        
        y_tmp = y[idx]
        
        paras_i = (D, R[i], K_ps, V_solution, N_particle[i])
        dydt[idx] = dydt[idx] + MoL_rhs_sphere2solution (t, y_tmp, paras_i)
    
    return dydt

        
def MoL_sphere2solution (t_range):
    
    N_meshes = 50
    
    mass = 10e-3 # kg
    density = 1e3 # kg/m3
    bulk_volume = mass / density

    
    D = 1e-11
    R = 1e-3
    K_ps = 1/ ( 0.1 / (density*1e-3) ) # 0.1 is the reported apparent partition coefficient
    V_solution = 0.3e-3
    
    each_volume = 4/3*np.pi*R**3    
    N_particle = bulk_volume/each_volume
    

    
    #N_particle = 1
    #print(N_particle)
    
    paras = (D, R, K_ps, V_solution, N_particle)
    
    n_t = len(t_range)
    y0 = np.zeros(N_meshes)
    y0[:N_meshes-1] = 1
    #y0[0] = 1
    y = np.zeros((n_t, N_meshes))
    alyt_frac = np.zeros(n_t)
    y[0,:] = y0
    
    for i in range(n_t-1):
        
        # numerical solution
        t_start = t_range[i]
        t_end = t_range[i+1]
    
        r = ode(MoL_rhs_sphere2solution).set_integrator('vode', method='bdf', 
               nsteps=5000, with_jacobian=True,atol=1e-10,rtol=1e-8)
        r.set_initial_value(y0, t_start).set_f_params(paras)
        r.integrate( r.t + t_end-t_start )
        y0 = r.y
        y[i+1,:] = y0
        
        # analytical solution
        alyt_frac[i+1] = Alyt_sphere2solution (t_end, paras)

    #plt.plot(np.transpose(y))
    plt.plot(t_range[:100], y[:100,-1]/y[-1, -1], label='num')
    plt.plot(t_range[:100], alyt_frac[:100], label='alyt')
    plt.legend()
    plt.show()
    return y


def MoL_sphere2solution_psd (t_range):
    
    N_meshes = 50
    
    # mass = 10e-3 # kg
    density = 1e3 # kg/m3
    # bulk_volume = mass / density

    
    D = 1e-11
    R = np.array([1e-4, 1e-3])
    K_ps = 1/ ( 0.1 / (density*1e-3) ) # 0.1 is the reported apparent partition coefficient
    V_solution = 0.3e-3
    
    # each_volume = 4/3*np.pi*R**3    
    N_particle = np.array([100, 2000])
        
    paras = (D, R, K_ps, V_solution, N_particle)
    
    n_t = len(t_range)
    y0 = np.zeros(N_meshes*2+1)
    y0[:N_meshes*2] = 1
    #y0[0] = 1
    y = np.zeros((n_t, N_meshes*2+1))
    alyt_frac = np.zeros(n_t)
    y[0,:] = y0
    
    for i in range(n_t-1):
        
        # numerical solution
        t_start = t_range[i]
        t_end = t_range[i+1]
    
        r = ode(MoL_rhs_sphere2solution_psd).set_integrator('vode', method='bdf', 
               nsteps=5000, with_jacobian=True,atol=1e-10,rtol=1e-8)
        r.set_initial_value(y0, t_start).set_f_params(paras)
        r.integrate( r.t + t_end-t_start )
        y0 = r.y
        y[i+1,:] = y0
        
        # analytical solution
        #alyt_frac[i+1] = Alyt_sphere2solution (t_end, paras)

    #plt.plot(np.transpose(y))
    plt.plot(t_range[:100], y[:100,-1]/y[-1, -1], label='num')
    #plt.plot(t_range[:100], alyt_frac[:100], label='alyt')
    plt.legend()
    plt.show()
    return y


def Alyt_sphere2solution (t, args, alpha_ubnd=20, N_alpha=-1):
    """ Analytical solution of diffusion from spheres into solution
    Args:
        t - time
        args - a list containing:
            D - diffusivity (m^2/s)
            R - radius of the sphere (m)
            K_ps - partition coefficient between particle (sphere solid) and solution
            V_solution - volume of the solution
        alpha_ubnd - the upper bound for solving alpha's, the coefficients in the Crank's solution
        N_alpha - number of alpha's to use to approximate the infinite series; -1 means use all calculated
    Return:
        c / c_infty, i.e. the ratio of current concentration to that at equilibrium in solution
    """
    D, R, K_ps, V_solution, N_particle = args
    
    each_volume = 4/3*np.pi*R**3 
    V_particle = each_volume * N_particle
    F = V_solution/V_particle/K_ps
    # print(F)
    
    a = optimize.root(crank_series, np.linspace(0, alpha_ubnd, 100), F)
    alpha = unique(a.x)
    if N_alpha < 0  or  N_alpha > len(alpha):
        alpha = alpha[1:] # remove the first entry which is 0
    else:
        alpha = alpha[1:N_alpha] # remove the first entry which is 0
    #print(alpha)

    s = 0
    for i in range(len(alpha)):
        a1 = 6*F*(1+F) / ( (F*alpha[i])**2 + 9*F + 9 )
        a2 = np.exp(-alpha[i]**2*D*t/R**2)        
        s = s + a1*a2
        
    return 1-s
    


def crank_series(alpha, F):
    """ Solving a non-linear algebraic equation required
    in Crank's analytical solution for diffusion from spheres
    """
    return np.tan(alpha) - 3*alpha/(3+F*alpha**2)

def unique(narray, tol=1e-2):
    """ Find unique real values"""
    
    rlt = np.unique(np.sort(narray))
    
    idx = np.array( [-1]*len(rlt) )
    n_unique = 0
    for i in range(len(rlt)) :
        c = rlt[i]        
        if i == 0  or  abs(c - rlt[idx[n_unique-1]]) > tol:
            idx[n_unique] = i
            n_unique = n_unique+1 
    
    return rlt[idx[:n_unique]]
            
    
    
    