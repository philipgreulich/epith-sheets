# epith-sheets
The file "ising_hill_voter_sim_disorder.py" contains the function "simrun" to run a simulation of a two-state spin model on a square lattice. This includes models of cell fate with two cell types and regulation via lateral, nearest-neighbour signalling.
The file "par-var_sim.py" iterates over model parameters and calls the function "simrun" from "ising_hill_voter_sim_disorder" for each parameter choice. It generates output files for spins and order parameters ("magentisation") for each individual run and deposits them in a sub-folder "mags", which needs to be created before running.
The file "plots.py" generates plots for order parameters ("magnetisation") and spin configurations and deposits them in a sub-folder "plots", which needs to be created before.
