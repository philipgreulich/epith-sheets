import matplotlib as mpl
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
import glob

#------------------------------
#  Parameters:
#------------------------------
    
heatscale = 100     # scale for colour range in heatmaps 
# xlabel0 = r'$\alpha$' #r'$\alpha$'
# ylabel0 = r'$\beta$'
xlabel0 = r'Signalling strength $J$' #r'$\alpha$'    # x label in plots
ylabel0 = r'Order parameter $\phi,\tilde \phi$'    # y label in plots
one_plot = 1            # if all curves in one plot = 1, if plots in separate plots = 0
plot_mag = 1            # if magnetisation (order parameter) to be plotted = 1, else = 0
plot_mag_stag = 1      # if staggered magnetisation (staggered order parameter) to be plotted = 1, else = 0

# meta parameters of batchrun. Copy from batchfile. Only used for suitable batchmode
start_i0 = 0 # start value of index for first paramter, i. Only used for batchmode = 1
end_i0 = 9  # end value of index for first paramter, i. Only used for batchmode = 1

replicas0 = 80  # Number of replicas to be averaged over. Only used for batchmode = 2. Attention!!! check number of replicas in provided files (determined by external script). If too large, "file not found" error.
plot_var = 1   # which is the x-axis variale of the plot: 1 = 1st variable in filename, 2 = 2nd variable in filename (as defined in par-var_sim.py)


# end Parameters
#-------------------------------

 # Plot parameters
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markeredgewidth'] = 3
mpl.rcParams['lines.markersize'] = 13
mpl.rcParams['font.size'] = 15
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['errorbar.capsize'] = 2
mpl.rcParams['lines.color'] = 'k'
mpl.rcParams.update({'figure.autolayout': True})

#----- 
fig, ax = plt.subplots()
ax.set_prop_cycle(color=['black','crimson','red','orange','gold','yellow','green','cyan','blue','violet'])

#------------------------------
#  Functions
#------------------------------

# function that generates heatmap (landscapes), or 2D plots if required
def heatmap(a):                    
    a_df = pd.DataFrame(a)             # construct dataframe for heatmap plotting 
    a_df = a_df.sort_values(by=[0,1])
    a=np.array(a_df)
    x=[]
    y=[]
    z=[]
    yflag=0
    i=0

    while i < len(a):    # generate (x,y,z) coordinates for heatmap. a must be 2D array, i.e. list of points (x,y,z) (z is interpreted as 'heat' coordinate)
        x_curr=a[i][0]
        x.append(x_curr)
        z_x=[]
        while (i < len(a)) and (a[i][0] == x_curr):
            z_x.append(a[i][2])
            if yflag == 0:
                y.append(a[i][1])
            i=i+1
        z.append(z_x)
        yflag=1
     
    x = np.array(x)
    y = np.array(y)
    if (len(y) == 1) or (len(x) == 1):     # in this case, no heatmap and instead 2D plot
        z = np.array(z).flatten()
    else:
        z = (np.array(z)[:-1,:-1]).T
        
    z_max = z.max()
    z_min = z_max - heatscale
    
    fig, ax = plt.subplots()            # prepare plot
    if len(y) == 1:
        plt.ylim(z_min,z_max+2)
        plt.plot(x,z,'k-')
    elif len(x) == 1:
        plt.ylim(z_min,z_max+2)
        plt.plot(y,z,'k-')
    else:
        c = ax.pcolormesh(x, y, z, cmap='hot', vmin=z_min, vmax=z_max)      # object for heatmap
    # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        fig.colorbar(c, ax=ax)
    
    plt.xlabel(xlabel0)
    plt.ylabel(ylabel0)
    
    plt.savefig("heatmap.pdf")#, format='pdf')
    plt.close()




#---------------------------------
#  Read parameters data
#---------------------------------

file = glob.glob('./mags/parameters.json', recursive=True)
with open(file[0],'r') as f:
    pars = json.load(f)

fileindex_multiplier = pars["fileindex_multiplier"]
L = pars["L"]
runtime = pars["runtime"]
ntimepts = pars["ntimepts"]
J = pars["J"]
copyrate = pars["copyrate"]
fliprate = pars["fliprate"]
start_i = pars["start_i"]
end_i = pars["end_i"]
par_1_intv = pars["par_1_intv"]
start_j = pars["start_j"]
end_j = pars["end_j"]
par_2_intv = pars["par_2_intv"]
#replicas = pars["replicas"]
batchmode = pars["batchmode"]
#q = pars["q"]


replicas=1
# start_i = start_i0
if batchmode == 1:
    end_i = end_i0
if batchmode == 2:
    replicas = replicas0
  


#-------- initialise empty data arrays --------------
# 3 indices: replica index, first parameter, second parameter. here "magnetisation" = "order parameter"

mag_t=np.zeros((replicas,end_i - start_i+1,end_j-start_j+1,ntimepts))   # abs. magnetisation as function of time
mag=np.zeros((replicas,end_i - start_i+1,end_j-start_j+1))          # abs. magnetisation at end of simulation
mag_norm=np.zeros((replicas,end_i - start_i+1,end_j-start_j+1))     # normalised magnetisation at end of simulation (between 0 and 1)
mag_stag_t=np.zeros((replicas,end_i - start_i+1,end_j-start_j+1,ntimepts))   # abs. staggered magnetisation as function of time
mag_stag=np.zeros((replicas,end_i - start_i+1,end_j-start_j+1))           # abs. staggered magnetisation at end of simulation
mag_stag_norm=np.zeros((replicas,end_i - start_i+1,end_j-start_j+1))      # normalised staggered magnetisation at end of simulation (between 0 and 1)


timepts = (np.array(range(ntimepts))+1)*runtime/ntimepts    # timepoints where magnetisation is recorded

#-------------------------------------------------------
#---------- Loop to read all source files --------------

for fileindex0 in range(replicas): 
    fileindex0_str = str(fileindex0)
    
    for i in range(start_i,end_i + 1):       # loop over first parameter
        fileindex1 = str(int(i*par_1_intv*fileindex_multiplier))   # generate string for filename index
        filename_spins = "mags/spins_" + fileindex1 +".txt"
        for j in range(start_j,end_j + 1):              # loop over second parameter
            fileindex2 = str(int(j*par_2_intv*fileindex_multiplier))
            if batchmode == 2:                   # generate all indexed filenames. for batchmode = 2, 3 indices (with replica), otherwise 2 indices
                filename_mag_t = "mags/mag_t_" + fileindex0_str + "_" + fileindex1 + "_" + fileindex2 + ".txt"
                filename_mag = "mags/mag_" + fileindex0_str + "_" + fileindex1 + "_" + fileindex2 + ".txt"
                filename_mag_norm = "mags/mag_norm_" + fileindex0_str + "_" + fileindex1 + "_" + fileindex2 + ".txt"
                filename_mag_stag_t = "mags/mag_stag_t_" + fileindex0_str + "_" + fileindex1 + "_" + fileindex2 + ".txt"
                filename_mag_stag = "mags/mag_stag_" + fileindex0_str + "_" + fileindex1 + "_" + fileindex2 + ".txt"
                filename_mag_stag_norm = "mags/mag_stag_norm_" + fileindex0_str + "_" + fileindex1 + "_" + fileindex2 + ".txt"
            else:
                filename_mag_t = "mags/mag_t_" + fileindex1 + "_" + fileindex2 + ".txt"
                filename_mag = "mags/mag_" + fileindex1 + "_" + fileindex2 + ".txt"
                filename_mag_norm = "mags/mag_norm_" + fileindex1 + "_" + fileindex2 + ".txt"
                filename_mag_stag_t = "mags/mag_stag_t_" + fileindex1 + "_" + fileindex2 + ".txt"
                filename_mag_stag = "mags/mag_stag_" + fileindex1 + "_" + fileindex2 + ".txt"
                filename_mag_stag_norm = "mags/mag_stag_norm_" + fileindex1 + "_" + fileindex2 + ".txt"
            
            
            #------- generate arrays from read data ----------------------
            mag_t_curr = np.array(np.loadtxt(filename_mag_t)).T
            mag_curr = np.array(np.loadtxt(filename_mag)).T             
            mag_norm_curr = np.array(np.loadtxt(filename_mag_norm)).T
            mag_stag_t_curr = np.array(np.loadtxt(filename_mag_stag_t)).T
            mag_stag_curr = np.array(np.loadtxt(filename_mag_stag)).T             
            mag_stag_norm_curr = np.array(np.loadtxt(filename_mag_stag_norm)).T
            #spins_curr = np.array(np.loadtxt(filename_spins))
            
            i_eff = i - start_i         # offset index of i
            j_eff = j - start_j         # offset index of j
            
            mag_t[fileindex0,i_eff,j_eff] = mag_t_curr[1]
            mag[fileindex0,i_eff,j_eff] = mag_curr[2]
            mag_norm[fileindex0,i_eff,j_eff] = mag_norm_curr[2]
            mag_stag_t[fileindex0,i_eff,j_eff] = mag_stag_t_curr[1]
            mag_stag[fileindex0,i_eff,j_eff] = mag_stag_curr[2]
            mag_stag_norm[fileindex0,i_eff,j_eff] = mag_stag_norm_curr[2]
            
            if batchmode < 2:
                plt.plot(timepts,mag_t[fileindex0,i-start_i,j-start_j]/(L*L))
        
        
        # filename_spins_exp = filename_spins = "plots/spins_" + fileindex1 +".pdf"
        # plt.imshow(spins_curr,cmap="Greys")
        # plt.savefig(filename_spins_exp)
        # plt.close()
        
                

plt.xlabel("time (MCS)")
plt.ylabel("order parameter")
plt.savefig("time_plots.pdf")
plt.close()

mag_t_replav = np.sum(np.abs(mag_t),axis=0)/replicas
mag_replav = np.sum(np.abs(mag),axis=0)/replicas
mag_norm_replav = np.sum(np.abs(mag_norm),axis=0)/replicas
mag_stag_t_replav = np.sum(np.abs(mag_stag_t),axis=0)/replicas
mag_stag_replav = np.sum(np.abs(mag_stag),axis=0)/replicas
mag_stag_norm_replav = np.sum(np.abs(mag_stag_norm),axis=0)/replicas


colours=['b','c','g','y','orange','r','darkred','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k']


#---------------------------------------------------------------
#-------- Loops to generate all magnetisation plots ------------
#---------------------------------------------------------------

# -------- 1st parameter is x-axis value, 2nd parameter represents different curves ----------
if plot_var == 1:

    # Generate filenames for magnetisation plots, with indices 
    fileindex_plot_start = str(int(start_j*par_2_intv*fileindex_multiplier))
    fileindex_plot_end = str(int(end_j*par_2_intv*fileindex_multiplier))
    filename_mag = "plots/mag[par1]_par2=" + fileindex_plot_start + "_to_" + fileindex_plot_end + ".pdf"
    filename_mag_stag = "plots/mag_stag[par1]_par2=" + fileindex_plot_start + "_to_" + fileindex_plot_end + ".pdf"      
    
    plt.xlabel(xlabel0)
    plt.ylabel(ylabel0)
    
    #---- Loop for plots of abs. magnetisation ----------------
    for j in range(start_j,end_j + 1):
        i_eff = i - start_i
        j_eff = j - start_j
        mag_plot_curr = np.abs(mag_replav[:,j_eff])
        mag_stag_plot_curr = np.abs(mag_stag_replav[:,j_eff])
        fileindex_plot = str(int(j*par_2_intv*fileindex_multiplier))
        
        # plot magnetisation
        if plot_mag == 1:
            plt.plot(par_1_intv*np.arange(start_i,end_i + 1),mag_plot_curr,color=colours[j_eff])
        if plot_mag_stag == 1:
            plt.plot(par_1_intv*np.arange(start_i,end_i + 1),mag_stag_plot_curr,color=colours[j_eff],linestyle="--")

    plt.savefig(filename_mag)
    plt.close()
    
    # Generate filenames for normalised magnetisation plots, with indices 
    filename_mag = "plots/mag_norm[par1]_par2=" + fileindex_plot_start + "_to_" + fileindex_plot_end + ".pdf"
    filename_mag_stag = "plots/mag_stag_norm[par1]_par2=" + fileindex_plot_start + "_to_" + fileindex_plot_end + ".pdf"      
    
    plt.xlabel(xlabel0)
    plt.ylabel(ylabel0)
    
    #---- Loop for plots of abs. magnetisation ----------------
    for j in range(start_j,end_j + 1):
        i_eff = i - start_i
        j_eff = j - start_j
        mag_plot_curr = np.abs(mag_norm_replav[:,j_eff])
        mag_stag_plot_curr = np.abs(mag_stag_norm_replav[:,j_eff])
        fileindex_plot = str(int(j*par_2_intv*fileindex_multiplier))
        
        # plot normalised magnetisation
        if plot_mag == 1:
            plt.plot(par_1_intv*np.arange(start_i,end_i + 1),mag_plot_curr,color=colours[j_eff])
        if plot_mag_stag == 1:
            plt.plot(par_1_intv*np.arange(start_i,end_i + 1),mag_stag_plot_curr,color=colours[j_eff],linestyle="--")

    plt.savefig(filename_mag)
    plt.close()
    

        
# -------- 2nd parameter is x-axis value, 1st parameter represents different curves ---------- (comments as above)      
elif plot_var == 2:

    fileindex_plot_start = str(int(start_i*par_2_intv*fileindex_multiplier))
    fileindex_plot_end = str(int(end_i*par_2_intv*fileindex_multiplier))
    filename_mag = "plots/mag[par1]_par2=" + fileindex_plot_start + "_to_" + fileindex_plot_end + ".pdf"
    filename_mag_stag = "plots/mag_stag[par1]_par2=" + fileindex_plot_start + "_to_" + fileindex_plot_end + ".pdf"      
    
    # plt.xlabel("signalling strength J")
    # plt.ylabel("order parameter")
    for i in range(start_i,end_i + 1):
        i_eff = i - start_i
        j_eff = j - start_j
        mag_plot_curr = np.abs(mag_replav[:,i_eff])
        mag_stag_plot_curr = np.abs(mag_stag_replav[:,i_eff])
        fileindex_plot = str(int(i*par_2_intv*fileindex_multiplier))

        if plot_mag == 1:
            plt.plot(par_1_intv*np.arange(start_j,end_j + 1),mag_plot_curr,color=colours[i_eff])
        if plot_mag_stag == 1:
            plt.plot(par_1_intv*np.arange(start_j,end_j + 1),mag_stag_plot_curr,color=colours[i_eff],linestyle="--")

    plt.savefig(filename_mag)
    plt.close()
    
    
    filename_mag = "plots/mag_norm[par1]_par2=" + fileindex_plot_start + "_to_" + fileindex_plot_end + ".pdf"
    filename_mag_stag = "plots/mag_stag_norm[par1]_par2=" + fileindex_plot_start + "_to_" + fileindex_plot_end + ".pdf"      
    
    plt.xlabel(xlabel0)
    plt.ylabel(ylabel0)
    for i in range(start_i,end_i + 1):
        i_eff = i - start_i
        j_eff = j - start_j
        mag_plot_curr = np.abs(mag_norm_replav[:,i_eff])
        mag_stag_plot_curr = np.abs(mag_stag_norm_replav[:,i_eff])
        fileindex_plot = str(int(i*par_2_intv*fileindex_multiplier))

        if plot_mag == 1:
            plt.plot(par_1_intv*np.arange(start_j,end_j + 1),mag_plot_curr,color=colours[i_eff])
        if plot_mag_stag == 1:
            plt.plot(par_1_intv*np.arange(start_j,end_j + 1),mag_stag_plot_curr,color=colours[i_eff],linestyle="--")

    plt.savefig(filename_mag)
    plt.close()
        

#------------ Plot time courses of magentisation ---------------------
# for i in range(start_i,end_i + 1):
#     for j in range(start_j,end_j + 1):
#         plt.plot(timepts,mag_t_replav[i-start_i,j-start_j],color='k')
# plt.savefig("time_plots.pdf")
# plt.close()

# for i in range(start_i,end_i + 1):
#     for j in range(start_j,end_j + 1):
#         plt.plot(timepts,mag_stag_t_replav[i-start_i,j-start_j],color='k')
# plt.savefig("time_plots_stag.pdf")
# plt.close()

for i in range(start_i,end_i + 1):
    for j in range(start_j,end_j):
        plt.plot(timepts,mag_t_replav[i-start_i,j-start_j],color='k')
    plt.plot(timepts,mag_t_replav[i-start_i,end_j-start_j],color='b')
plt.savefig("time_plots.pdf")
plt.close()

for i in range(start_i,end_i + 1):
    for j in range(start_j,end_j):
        plt.plot(timepts,mag_stag_t_replav[i-start_i,j-start_j],color='k')
    plt.plot(timepts,mag_stag_t_replav[i-start_i,end_j-start_j],color='b')
plt.savefig("time_plots_stag.pdf")
plt.close()
        

        
    
    
    
    
    
