# Oscillations_toroidal_topology
### Compute $\Gamma$ on your data at https://github.com/gdisarra/topology-degree!
Code for the paper di Sarra G, Jha S, Roudi Y (2025) The role of oscillations in grid cells’ toroidal topology. PLOS Computational Biology 21(1): e1012776. https://doi.org/10.1371/journal.pcbi.1012776.

Data from Gardner et al. should be stored in the folder Toroidal_topology_grid_cell_data after download from https://figshare.com/articles/dataset/Toroidal_topology_of_population_activity_in_grid_cells/16764508

## Gamma_computation
Provides the code to compute the degree of toroidal topology ${\bf \Gamma}$ and ${\bf \Gamma}^{self}$ given a set of barcodes from persistent cohomology.

## synthetic_dataset
Code for the simulation of poisson grid cell modules (uses functions in multi_sample_funcs.py).
Code to generate figures 5,6,7 and 11

## utils
Modified version of utility functions file from https://github.com/erikher/GridCellTorus.

## jitter_spiketimes
Plots the values of ${\bf \Gamma}$ for different values of jitter (data in Gamma_jit.pkl) in experimental data. Ouputs the values of $\Delta t_C^1$ and $\Delta t_C^2$. Code to generate figures 4 and S1

## power_spectrum
Computes power spectral density of the spike train for each cell. Correlates eta-to-theta power with grid spacing, ${\bf \Gamma}$ and $\Delta t_C^1$ and $\Delta t_C^2$.
Code to generate figures 8,9 and 10
