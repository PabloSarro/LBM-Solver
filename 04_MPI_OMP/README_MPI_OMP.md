# MPI+OpenMP LBM Solver README

## 1. Setup & Environment
Once connected to the JED (or IZAR) cluster:

```bash
# Load environment
module purge
module load gcc hdf5 openmpi

# Install dependencies (for plotting)
pip install numpy pyvista h5py imageio matplotlib scipy

```

## 2. Build

```bash
make clean
make all
```

## 3. Run

```bash
# Run through SLURM job.
sbatch lbm_MPI_OMP_JED.job # If within JED
sbatch lbm_MPI_OMP_IZAR.job # If within IZAR
```