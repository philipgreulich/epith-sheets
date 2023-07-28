#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 08:52:22 2022

@author: phil
"""


import numpy as np
import math
import random as rn
import scipy.stats as st
import sys
from matplotlib import pyplot as plt
from copy import deepcopy as dcopy

# L=10
# runtime=100
# timepts=2
# spins=np.zeros((L,L))

# copyrate = 1.0
# fliprate = copyrate
# J=0.1

#------------ Function to call simulation run ---------------------------------
def simrun(J0,copyrate,fliprate,L,runtime,ntimepts=1,rev_fate=0,interaction="ising",disorder=0,sigma_J=0,nstates=2):
    
    # J0 = initial 'J', i.e. the coupling constant
    # copyrate = rate of symmetric divisions (neighbour copying events, according to voter model)
    # fliprate = rate of attempted spin flips (according to Glauber dynamics in Ising model)
    # runtime = total runtime of simulation. Time unit = 1/(total event rate)
    # ntimepoints =  number of timepoints to record intermediate results
    # rev_fate = 0: irreversible cell fate, chosen at division, i.e. copy event,  rev_fate = 1, cell fate reversibly switching with flip rate (spin flips)
    # interaction: form of interaction function: ="ising" logistic interaction function (as in Ising model), = "hill" Hill-shaped interaction function, with Hill coefficient = 1
    
    print("J = ", J0)
    print("copyrate = ", copyrate)
    print("fliprate = ", fliprate)
    print("L = ", L)
    
    
    # initialise random seed and spin lattice
    rn.seed()
    spins_t=[]
    
    Omega = fliprate + copyrate # total rate, determines tme unit and Monte Carlo step (MCS) update time.
    runtime = runtime/Omega; # runtime in unit of 1/Omega
    
    # compute equidistance time points (number = ntimepts)
    timepts = np.zeros(ntimepts)
    for i in range(ntimepts):
        timepts[i] = runtime*(i+1)/ntimepts

    spin_values = [-1,1]
    spins = np.random.choice(spin_values,size=(L,L))

        
    J = np.zeros((L,L))
    N = L*L
    for i in range(L):
        for j in range(L):
            if disorder == 1:
                J[i,j] = rn.normalvariate(J0,sigma_J)
            else:
                J[i,j] = J0

    
    def nbsum(spins,J1,i1,j1,J2=0,i2=-2,j2=-2):

        if J2==0:
            J2=J1
            
        i1p = i1+1 if i1<L-1 else i1+1-L
        j1p = j1+1 if j1<L-1 else j1+1-L
        i2p = i2+1 if i2<L-1 else i2+1-L
        j2p = j2+1 if j2<L-1 else j2+1-L
        

        if i2 == -2 and j2 == -2:
            nb = np.array([spins[i1-1,j1],spins[i1p,j1],spins[i1,j1-1],spins[i1,j1p]])
            return(sum(nb))
        else:
            if (nstates > 2):
                print("error: irreversible mode not supported for nstates > 2")
                sys.exit()
            nb1 = [spins[i1-1,j1],spins[i1p,j1],spins[i1,j1-1],spins[i1,j1p]]
            nb2 = [spins[i2-1,j2],spins[i2p,j2],spins[i2,j2-1],spins[i2,j2p]]
            return(sum(nb1)+sum(nb2))
        
    def prob_up(spins,J1,i1,j1,J2=0,i2=-2,j2=-2):
        if interaction == "ising":
            dE = nbsum(spins,J1,i,j,J2,i2,j2)
            return(np.exp(J1*dE)/(np.exp(J1*dE) + np.exp(-J1*dE)))
        elif interaction == "hill":
            dE = nbsum(spins,J1,i,j,J2,i2,j2)
            if rev_fate == 1:
                return(1/2*(1+J1*dE/(1 + np.abs(J1*dE))))
            elif rev_fate == 0:
                return(1/2*(1+J1*dE/(1 + np.abs(J1*dE))))
            else: print("error: rev_fate not well defined.")
        else: print("error: 'interaction' not well defined.")
            
    
    t=0
    for k in range(ntimepts):
        while t < timepts[k]: 
            Omega = fliprate + copyrate
            for n in range(N):    
                rn.random()
                r1 = rn.randrange(N)
                i = r1 // L
                j = r1 % L
                spin=spins[i,j]
                r2 = rn.random()
                
                ip = i+1 if i<L-1 else i+1-L
                jp = j+1 if j<L-1 else j+1-L
                
                
                if rev_fate == 1:
                    
                    r2 = rn.random()
                    P_up = prob_up(spins,J[i,j],i,j);
                    if r2 < copyrate/Omega/4:
                        spins[ip,j] = spin
                    elif r2 < copyrate/Omega/2:
                        spins[i,jp] = spin
                    elif r2 < 3*copyrate/Omega/4:
                        spins[i-1,j] = spin
                    elif r2 < copyrate/Omega:
                        spins[i,j-1] = spin
                    elif r2 < (copyrate+P_up*fliprate)/Omega:
                        spins[i,j] = 1
                    elif r2 < (copyrate+fliprate)/Omega:
                        spins[i,j] = -1
                    
                        
                elif rev_fate == 0:
                    
                    if nstates > 2:
                        print("error: irreversible mode not supported for nstates > 2")
                        
                    index = math.floor(r2*4)
                    i_n = i + int(np.cos(np.pi/2*index))
                    j_n = j + int(np.sin(np.pi/2*index))
                    
                    i_n = i_n if i_n < L else i_n-L
                    j_n = j_n if j_n < L else j_n-L
                    
                    spin_n = np.array([i_n,j_n])
                    rn.random()
                    
                    if spins[i_n,j_n] != spins[i,j]:
                        P_up = prob_up(spins,J[i,j],i,j,J[i_n,j_n],i_n,j_n)
                        if rn.random() < P_up:
                            spins[i,j] = 1
                            spins[i_n,j_n] = 1
                        else:
                            spins[i,j] = -1
                            spins[i_n,j_n] = -1
            
            if rev_fate == 0:
                t = t+1.0
            elif rev_fate == 1:
                t = t+1/Omega
            else:
                print("error: rev_fate not well defined")
                

        spins_t.append(dcopy(spins))
       

    # print("spins_t = \n",spins_t)
    return spins_t,J
                    
                    


        