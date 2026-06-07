# CUDA LBM Solver README

## 1. Setup & Environment

```bash
# Load environment
module purge
module load gcc hdf5 cuda

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
# Example: Run on 1 GPU
./lbm_CUDA nx=1600 ny=800 steps=60000

```

## 4. Visualization & Analysis

```bash
# Generate GIFs
python3 viz/viz.py results/lbm.xdmf --field vorticity
python3 viz/viz.py results/lbm.xdmf --field ux --cmap viridis

# Calculate Strouhal number
python3 viz/strouhal.py results/probe.csv --u-in 0.05 --diameter 20

```

*Note: Ensure `.h5` files are in the same directory as the `.xdmf` file index.*