#include "lbm_CUDA.hh"

//  

#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                << " code=" << err << " \"" << cudaGetErrorString(err) \
                << "\"" << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } while (0)

// D2Q9 lattice constants. Indexing convention used throughout:
//   0: rest         5: NE
//   1: E            6: NW
//   2: N            7: SW
//   3: W            8: SE
//   4: S

// HOST CONSTANTS
const int LBM::cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
const int LBM::cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};
const double LBM::w[9] = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};
const int LBM::opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};


// DEVICE CONSTANTS
__constant__ int    cx_dev[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
__constant__ int    cy_dev[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};
__constant__ double w_dev[9]  = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};
__constant__ int opp_dev[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};


// CONSTRUCTOR.
LBM::LBM(std::size_t nx, std::size_t ny,
         double u_in, double Re,
         double cyl_x, double cyl_y, double cyl_r): 
    nx_(nx), ny_(ny), u_in_(u_in), tau_(0.0),
    f_   (9*nx_*ny_, 0.0),
    ftmp_(9*nx_*ny_, 0.0),
    solid_(nx_*ny_, 0)
{
  const double nu = u_in_ * (2.0*cyl_r)/Re;
  tau_ = 3.0*nu + 0.5;

  for (std::size_t x = 0; x < nx_; ++x) {
    solid_[idx(x, 0)]     = 1;
    solid_[idx(x, ny_-1)] = 1;
  }
  mark_obstacle(cyl_x, cyl_y, cyl_r);

  // Allocate GPU VRAM.
  cudaMalloc(&f_d_,    9*nx_*ny_*sizeof(double));
  cudaMalloc(&ftmp_d_, 9*nx_*ny_*sizeof(double));
  cudaMalloc(&solid_d_,  nx_*ny_*sizeof(uint8_t));
}

// DESTRUCTOR.
LBM::~LBM() {
  // Free GPU VRAM.
  cudaFree(f_d_);
  cudaFree(ftmp_d_);
  cudaFree(solid_d_);
}

// KERNEL 1: COLLISION, BOUNCE-BACK AND STREAM.
__global__ void lbm_fused_kernel(const double* __restrict__ f,
                                 double* __restrict__ ftmp,
                                 const uint8_t* __restrict__ solid,
                                 int nx, int ny, double inv_tau) 
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;

  int N = nx*ny;
  int idx = y*nx + x;

  // Read local populations (Perfectly coalesced)
  double local_f[9];
  for (int i = 0; i < 9; ++i) {
    local_f[i] = f[i*N + idx];
  }

  // 1. Collision
  double rho = 0.0, mx = 0.0, my = 0.0;
  for (int i = 0; i < 9; ++i) {
    rho += local_f[i];
    mx  += cx_dev[i]*local_f[i];
    my  += cy_dev[i]*local_f[i];
  }
  double ux = (rho > 0.0) ? mx / rho : 0.0;
  double uy = (rho > 0.0) ? my / rho : 0.0;
  double u2 = ux*ux + uy*uy;

  for (int i = 0; i < 9; ++i) {
    double cu  = cx_dev[i]*ux + cy_dev[i]*uy;
    double feq = w_dev[i]*rho*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);
    local_f[i] += -inv_tau*(local_f[i] - feq);
  }

  // 2. Bounce-Back
  if (solid[idx]) {
    double tmp;
    tmp = local_f[1]; local_f[1] = local_f[3]; local_f[3] = tmp;
    tmp = local_f[2]; local_f[2] = local_f[4]; local_f[4] = tmp;
    tmp = local_f[5]; local_f[5] = local_f[7]; local_f[7] = tmp;
    tmp = local_f[6]; local_f[6] = local_f[8]; local_f[8] = tmp;
  }

  // 3. Push Stream
  for (int i = 0; i < 9; ++i) {
    int nx_x = x + cx_dev[i];
    int nx_y = y + cy_dev[i];

    if ( (nx_x >= 0)&&(nx_x < nx)&&(nx_y >= 0)&&(nx_y < ny) ) {
      ftmp[i*N + (nx_y*nx + nx_x)] = local_f[i];
    }
      
    // Fill un-pushed boundaries to match baseline logic
    int sx = x - cx_dev[i];
    int sy = y - cy_dev[i];
    if (sx < 0 || sx >= nx || sy < 0 || sy >= ny) {
      ftmp[i*N + idx] = f[i*N + idx];
    }
  }
}


// KERNEL 2: BOUNDARY CONDITIONS
__global__ void lbm_boundaries_kernel(double* __restrict__ ftmp,
                                      const uint8_t* __restrict__ solid,
                                      int nx, int ny, double u_in) 
{
  int y = blockIdx.x*blockDim.x + threadIdx.x; // ==MISTAKE!!== : 1D Grid over Y-axis
  if (y >= ny) return;

  int N = nx*ny;

  // INLET (x = 0)
  int idx_in = y*nx + 0;
  if (!solid[idx_in]) {
    double rho = 1.0;
    double ux = u_in;
    double uy = 0.0;
    double u2 = ux*ux + uy*uy;
    for (int i = 0; i < 9; ++i) {
      double cu = cx_dev[i]*ux + cy_dev[i]*uy;
      ftmp[i*N + idx_in] = w_dev[i]*rho*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);
    }
  }

  // OUTLET (x = nx-1)
  if (nx > 1) {
    int idx_out = y*nx + (nx-1);
    int idx_src = y*nx + (nx-2);
    for (int i = 0; i < 9; ++i) {
      ftmp[i*N + idx_out] = ftmp[i*N + idx_src];
    }
  }
}

// Define circular obstacle and mark inside as solid.
void
LBM::mark_obstacle(double c_x, double c_y, double r)
{
  const double r2 = r * r;
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const double dx = double(x) - c_x;
      const double dy = double(y) - c_y;
      if (dx*dx + dy*dy <= r2) solid_[idx(x,y)] = 1;
    }
  }
}


// Add second obstacle.
void
LBM::add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r)
{
  if (cyl2_r > 0.0) mark_obstacle(cyl2_x, cyl2_y, cyl2_r);
}


void LBM::initialize() {
  // Initialize the host vectors exactly as you did in the baseline
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x = 0; x < nx_; ++x) { 
      const double rho = 1.0; 
      const double ux  = solid_[idx(x,y)] ? 0.0 : u_in_; 
      const double uy  = 0.0; 
      const double u2  = ux * ux + uy * uy;
      for (int i = 0; i < 9; ++i) { 
        const double cu  = cx[i]*ux + cy[i] * uy;
        f_[i*nx_*ny_ + idx(x,y)] = w[i]*rho*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2); 
      }
    }
  }

  // Upload initial states to GPU
  cudaMemcpy(f_d_, f_.data(), 9*nx_*ny_*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(solid_d_, solid_.data(), nx_*ny_*sizeof(uint8_t), cudaMemcpyHostToDevice);
}


void LBM::step() {
  dim3 threads(32, 8);
  dim3 blocks((nx_ + 31) / 32, (ny_ + 7) / 8);

  // 1. Core Physics
  lbm_fused_kernel<<<blocks, threads>>>(f_d_, ftmp_d_, solid_d_, nx_, ny_, 1.0/tau_);
  CUDA_CHECK(cudaGetLastError()); // Catches launch and execution failures

  // 2. Boundary Fixes (1D Kernel along Y)
  int blocks_y = (ny_ + 255) / 256;
  lbm_boundaries_kernel<<<blocks_y, 256>>>(ftmp_d_, solid_d_, nx_, ny_, u_in_);
  CUDA_CHECK(cudaGetLastError()); // Catches launch and execution failures

  // 3. Pointer Swap (Zero overhead)
  std::swap(f_d_, ftmp_d_);
}

void LBM::download_to_host() {
  cudaMemcpy(f_.data(), f_d_, 9*nx_*ny_*sizeof(double), cudaMemcpyDeviceToHost);
}


// Special fast probe method to avoid copying the whole grid to host
void LBM::probe_velocity(std::size_t x, std::size_t y, double& out_ux, double& out_uy) const {
  double r = 0.0, mx = 0.0, my = 0.0;
  for (int i = 0; i < 9; ++i) {
    double fi;
    cudaMemcpy(&fi, f_d_ + i*nx_*ny_ + idx(x,y), sizeof(double), cudaMemcpyDeviceToHost);
    r += fi;
    mx += cx[i]*fi;
    my += cy[i]*fi;
  }
  out_ux = (r > 0.0) ? mx / r : 0.0;
  out_uy = (r > 0.0) ? my / r : 0.0;
}


double
LBM::rho(std::size_t x, std::size_t y) const
{
  const std::size_t N = nx_*ny_;
  double r = 0.0;
  int idx_rho = idx(x,y);
  for (int i = 0; i < Q; ++i) {
    r += f_[i*N + idx_rho];
  }
  return r;
}

double
LBM::ux(std::size_t x, std::size_t y) const
{
  const std::size_t N = nx_*ny_;
  double r = 0.0, m = 0.0;
  int idx_ux = idx(x,y);
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[i*N + idx_ux];
    r += fi;
    m += cx[i]*fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::uy(std::size_t x, std::size_t y) const
{
  const std::size_t N = nx_*ny_;
  double r = 0.0, m = 0.0;
  int idx_uy = idx(x,y);
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[i*N + idx_uy];
    r += fi;
    m += cy[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::vorticity(std::size_t x, std::size_t y) const
{
  if (x == 0 || x == nx_-1 || y == 0 || y == ny_-1) return 0.0;
  return 0.5 * ((uy(x+1, y) - uy(x-1, y)) - (ux(x, y+1) - ux(x, y-1)));
}

bool
LBM::is_solid(std::size_t x, std::size_t y) const
{
  return solid_[idx(x,y)] != 0;
}
