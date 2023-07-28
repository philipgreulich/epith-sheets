import numpy as np
import pandas as pd
import math
import sys
import data_ana_functions as da
import itertools
import json
# import potts_voter_sim_disorder as sim
import ising_hill_voter_sim_disorder as sim
from matplotlib import pyplot as plt





#### meta parameters ########
batchmode=0     # batchmode=0: local run. batchmode=1: job array submission, running over parameters. batchmode=2: job array submission, running over replicas 
disordered = 0
var=1           # How many parameters are varied. If var=1 there is no loop over the second parameter and the default value is chosen
ntimepts=10
rev_fate = 0
intact="hill"   # intact="ising": interaction with neighbours according to Ising update, intact="hill": interaction with neighbours according to Hill function.
fileindex_multiplier = 100
nstates=2



#### Model parameters ############## 
L=80
runtime=3200    # Attention !!! Adjusted below if L is varied .

# default parameters (those which are not varied)

pars={}

J = -2.0         # if those parameters are given below as "str_par_var1/2" they are over-ridden by the iteration loop
copyrate = 1.0
fliprate = 1.0
sigma_J = 1.0           # only relevant if disorder = 1

str_par_var1 = "J"        # See positions in list "parameters below". "q" means probability of divisions in reversible model, q=copyrate/(copyrate + fliprate)
str_par_var2 = "L"        # only relevant if var >= 2


# First parameter start, end, and interval. Attention!!! start_i and end _i overriden by input from batchscript for batchmode=1. start_j and end_j remain relevant.
start_i = -2        # Start of first parameter step i when run locally
end_i = 2        #  End of first parameter step i when run locally. 
par_1_intv = 0.3    # interval between subsequent parameters for parameter 1. Actual parameter values are i*par_1_intv 

start_j = 1     # Start of second parameter step j. Also valid when run via batchscript
end_j = 1       # End of second parameter step j. Also valid when run via batchscript
par_2_intv = 20   # End of second parameter step j. Also valid when run via batchscript

pars = {}
q = copyrate/(copyrate + fliprate)


pars["fileindex_multiplier"] = fileindex_multiplier
pars["L"] = L
pars["runtime"] = runtime
pars["ntimepts"] = ntimepts
pars["J"] = J
pars["sigma_J"] = sigma_J
pars["copyrate"] = copyrate
pars["fliprate"] = fliprate
pars["start_i"] = start_i
pars["end_i"] = end_i
pars["par_1_intv"] = par_1_intv
pars["start_j"] = start_j
pars["end_j"] = end_j
pars["par_2_intv"] = par_2_intv
pars["batchmode"] = batchmode
pars["q"] = q
pars["str_par_var1"] = str_par_var1      
pars["str_par_var2"] = str_par_var2


######## Main routine starts ########################33


if batchmode==1:        # if in batchmode, parameters are taken as job array parameters (e.g. on Iridis)
    start_i = int(sys.argv[1])
    end_i = start_i
    print("Attention!! batchmode=1 overrides start_i and end_i")
elif batchmode==2:
    fileindex0=str(int(sys.argv[1]))

arr_mag=[]
arr_mag_norm=[]
arr_mag_stag=[]
arr_mag_stag_norm=[]
arr_cluster=[]
arr_cluster_norm =[]


fig, ax = plt.subplots()
ax.set_prop_cycle(color=['black','crimson','red','orange','yellow', 'green', 'cyan', 'blue', 'violet'])

fileprefix_mag_t = "mags/mag_t_"
fileprefix_mag = "mags/mag_"
fileprefix_mag_plain = "mags/mag_plain_"
fileprefix_mag_norm = "mags/mag_norm_"

fileprefix_mag_stag_t = "mags/mag_stag_t_"
fileprefix_mag_stag = "mags/mag_stag_"
fileprefix_mag_stag_plain = "mags/mag_stag_plain_"
fileprefix_mag_stag_norm = "mags/mag_stag_norm_"

fileprefix_Js = "mags/Js_dis_"
fileprefix_spins = "mags/spins_"

#### Fitting routine: loop over parameters alpha = i*par_1_intv and beta = j*par_2_intv
for i in range(start_i,end_i + 1):
    alpha = i*par_1_intv
    for j in range(start_j,end_j + 1):
        beta = j*par_2_intv
        
        if var > 0:        # if parameters are supposed to be varied.
            pars[str_par_var1] = alpha               
        if var == 2:
            pars[str_par_var2] = beta
        
        if str_par_var1 == "q":
            if alpha < 0:
                print("error: q < 0\n")
            elif alpha > 1:
                print("error: q > 1\n")
            elif alpha==0:
                pars["copyrate"] = 0
                pars["fliprate"] = 1.0
            else:
                pars["fliprate"] = copyrate*(1-alpha)/alpha
                
            pars["q"] = alpha
            
        elif str_par_var2 == "q" and var==2:
            if beta < 0:
                print("error: q < 0\n")
            elif beta > 1:
                print("error: q > 1\n")
            elif beta == 0:
                pars["copyrate"] = 0
                pars["fliprate"] = 1.0
            else:
                pars["fliprate"] = copyrate*(1-beta)/beta
                
            pars["q"] = beta
 
        
        arr_mag=[]
        arr_mag_norm=[]
        arr_mag_stag=[]
        arr_mag_stag_norm=[]
        

            
        print("pars[copyrate] = ", pars["copyrate"])
        print("pars[fliprate] = ", pars["fliprate"])
        
        if (str_par_var1 == "L" ):
            pars["runtime"] = int(alpha*alpha)
        elif (str_par_var2 == "L" ) and (var==2):
            pars["runtime"] = int(beta*beta)
        
        print("new run")     
# Here the simulation function is called. This needs to be be adjusted to any particular model (arguments and return values only relevant for particular model):  
        if disordered == 0:
            spins_t,Js = sim.simrun(pars["J"],pars["copyrate"],pars["fliprate"],pars["L"],pars["runtime"],ntimepts=ntimepts,rev_fate=rev_fate,interaction=intact,nstates=nstates)
        else:
            spins_t,Js = sim.simrun(pars["J"],pars["copyrate"],pars["fliprate"],pars["L"],pars["runtime"],ntimepts=ntimepts,rev_fate=rev_fate,interaction=intact,disorder=1,sigma_J=sigma_J,nstates=nstates)
        
        pars["copyrate"] = copyrate
        
        print("simulation completed; ",str_par_var1,"=",pars[str_par_var1],str_par_var2,"=",pars[str_par_var2])
        print("runtime = ",pars["runtime"], "\n")
            
        # plt.matshow(spins_t[ntimepts-1])
        # plt.show()
        # plt.close()
        
        # clustersizes_t = []
        # for t in range(ntimepts):
        #     clusters=da.clusters_HK(spins_t[t],L)
        #     clustersizes = da.cluster_sizes(clusters[0])
        #     if clustersizes == []:
        #         clustersizes_t.append([])
        #     else:
        #         clustersizes_t.append(clustersizes[0])
        L=len(spins_t[ntimepts-1])
        spins_checkerb = []
        for t in range(ntimepts):
            spins_checkerb_curr = np.zeros((L,L))
            for l in range(L):
                for k in range(L):
                    spins_checkerb_curr[l,k] = spins_t[t][l,k]*(-1)**(l+k)
            spins_checkerb.append(spins_checkerb_curr)
                
    
        mag_t = np.sum(spins_t,(1,2))
        mag_norm_t = mag_t/(L*L)
        mag_stag_t = np.sum(spins_checkerb,(1,2))
        mag_stag_norm_t = mag_stag_t/(L*L)
        # av_cluster_t = np.vectorize(np.mean)(clustersizes_t)
        # av_cluster_norm_t = av_cluster_t/(L*L)
    
        
        mag = mag_t[ntimepts-1]
        mag_norm = mag_norm_t[ntimepts-1]
        mag_stag = mag_stag_t[ntimepts-1]
        mag_stag_norm = mag_stag_norm_t[ntimepts-1]
        # av_cluster = av_cluster_t[ntimepts-1]
        # av_cluster_norm = av_cluster/(L*L)
        
        arr_mag.append([alpha,beta,mag])
        arr_mag_norm.append([alpha,beta,mag_norm])
        arr_mag_stag.append([alpha,beta,mag_stag])
        arr_mag_stag_norm.append([alpha,beta,mag_stag_norm])
        # arr_cluster.append([J,fliprate, av_cluster])
        # arr_cluster_norm.append([J,fliprate, av_cluster_norm])
        
        mag_t_export = np.vstack(((np.array(range(ntimepts))+1)*runtime/ntimepts,abs(mag_t)))
        mag_t_export = mag_t_export.T
        mag_stag_t_export = np.vstack(((np.array(range(ntimepts))+1)*runtime/ntimepts,abs(mag_stag_t)))
        mag_stag_t_export = mag_stag_t_export.T
        # plt.plot(mag_t_export.T[0],mag_t_export.T[1])  
        
        fileindex1 = str(int(i*par_1_intv*fileindex_multiplier))
        fileindex2 = str(int(j*par_2_intv*fileindex_multiplier))
        
        if batchmode < 2:
            
            filename_mag_t = fileprefix_mag_t + fileindex1 + "_" + fileindex2  + ".txt"
            filename_mag = fileprefix_mag + fileindex1 + "_"  + fileindex2 + ".txt"
            filename_mag_plain = fileprefix_mag_plain + fileindex1 + "_"  + fileindex2 + ".txt"
            filename_mag_norm = fileprefix_mag_norm + fileindex1 + "_"  + fileindex2 + ".txt"
            filename_mag_stag_t = fileprefix_mag_stag_t + fileindex1 + "_" + fileindex2  + ".txt"
            filename_mag_stag = fileprefix_mag_stag + fileindex1 + "_"  + fileindex2 + ".txt"
            filename_mag_stag_plain = fileprefix_mag_stag_plain + fileindex1 + "_"  + fileindex2 + ".txt"
            filename_mag_stag_norm = fileprefix_mag_stag_norm + fileindex1 + "_"  + fileindex2 + ".txt"
            filename_Js = fileprefix_Js + fileindex1 + "_"  + fileindex2 + ".txt"
            
            filename_spins = fileprefix_spins + fileindex1 + ".txt"
            # filename_cluster = fileprefix_cluster + fileindex + ".txt"
            # filename_cluster_norm = fileprefix_cluster_norm + fileindex + ".txt"
            
            np.savetxt(filename_mag_t,mag_t_export)
            np.savetxt(filename_mag,arr_mag)
            np.savetxt(filename_mag_plain,np.array([mag_t[ntimepts-1]]))
            np.savetxt(filename_mag_norm,arr_mag_norm)
            np.savetxt(filename_mag_stag_t,mag_stag_t_export)
            np.savetxt(filename_mag_stag,arr_mag_stag)
            np.savetxt(filename_mag_stag_plain,np.array([mag_stag_t[ntimepts-1]]))
            np.savetxt(filename_mag_stag_norm,arr_mag_stag_norm)
            np.savetxt(filename_Js,Js)
            np.savetxt(filename_spins,spins_t[ntimepts-1].astype(int))
    
        if batchmode == 2:

            filename_mag_t = fileprefix_mag_t + fileindex0 + "_" + fileindex1 + "_" + fileindex2  + ".txt"
            filename_mag = fileprefix_mag + fileindex0 + "_" + fileindex1 + "_"  + fileindex2 + ".txt"
            filename_mag_plain = fileprefix_mag_plain + fileindex0 + "_" + fileindex1 + "_"  + fileindex2 + ".txt"
            filename_mag_norm = fileprefix_mag_norm + fileindex0 + "_" + fileindex1 + "_"  + fileindex2 + ".txt"
            filename_mag_stag_t = fileprefix_mag_stag_t + fileindex0 + "_" + fileindex1 + "_" + fileindex2  + ".txt"
            filename_mag_stag = fileprefix_mag_stag + fileindex0 + "_" + fileindex1 + "_"  + fileindex2 + ".txt"
            filename_mag_stag_plain = fileprefix_mag_stag_plain + fileindex0 + "_" + fileindex1 + "_"  + fileindex2 + ".txt"
            filename_mag_stag_norm = fileprefix_mag_stag_norm + fileindex0 + "_" + fileindex1 + "_"  + fileindex2 + ".txt"
            filename_Js = fileprefix_Js + fileindex0 + "_" + fileindex1 + "_"  + fileindex2 + ".txt"
            
            filename_spins = fileprefix_spins + fileindex1 + ".txt"
            
            np.savetxt(filename_mag_t,mag_t_export)
            np.savetxt(filename_mag,arr_mag)
            np.savetxt(filename_mag_plain,np.array([mag_t[ntimepts-1]]))
            np.savetxt(filename_mag_norm,arr_mag_norm)
            np.savetxt(filename_mag_stag_t,mag_stag_t_export)
            np.savetxt(filename_mag_stag,arr_mag_stag)
            np.savetxt(filename_mag_stag_plain,np.array([mag_stag_t[ntimepts-1]]))
            np.savetxt(filename_mag_stag_norm,arr_mag_stag_norm)
            np.savetxt(filename_Js,Js)
            np.savetxt(filename_spins,spins_t[ntimepts-1].astype(int))
            
            

plt.savefig("time_plots.pdf")
plt.close()

with open('mags/parameters.json', 'w') as f:
    json.dump(pars, f)



    
