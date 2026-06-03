#include "lbm_MPI.hh"

//  

#include <algorithm>
#include <mpi.h>

// ======================================================== //
// ====================== lbm_MPI.cc ====================== //
// ======================================================== //

// D2Q9 lattice constants. Indexing convention used throughout:
//   0: rest         5: NE
//   1: E            6: NW
//   2: N            7: SW
//   3: W            8: SE
//   4: S

// VELOCITIES
const int LBM::cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
const int LBM::cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};

// WEIGHTS FOR LOCAL EQUILIBRIUM: f_i^{eq}(rho, u)
const double LBM::w[9] = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};


// CONSTRUCTOR.
LBM::LBM(std::size_t nx, std::size_t ny,
         double u_in, double Re,
         double cyl_x, double cyl_y, double cyl_r,
         int rank, int size): 
    nx_(nx), ny_(ny), u_in_(u_in), tau_(0.0), // Number of cells, velocity and relaxation time.
    rank_(rank), size_(size),                 // MPI rank and size.
    
    // Distribute the remainder of floor(nx/size) by giving 1 to the first nx%size workers.
    nx_local_((nx / size) + (rank < int(nx % size) ? 1 : 0)),        // Number of cells in the x direction that each MPI worker will process (from 1 to nx_local).
    nx_start_((rank*(nx / size)) + std::min(rank, int(nx % size))),  // Also, store the (global) position where x starts per worker (used to globally access its position).
    f_   (9 * (nx_local_+2) * ny_, 0.0),      // Flow distributions stored per worker (+2, since we will include cols 0 and nx_local_+1 for computation in the extremes).
    ftmp_(9 * (nx_local_+2) * ny_, 0.0),
    solid_(nx_local_*ny_, 0)                  // Solid cells (no need to allocate space for the extra cols, since we don't need them for the calculations).
{

  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;

  // DEFINE UPPER AND LOWER WALLS AS SOLID CELLS.
  for (std::size_t local_x = 1; local_x <= nx_local_; ++local_x) {
    solid_[solid_idx(local_x, 0)]      = 1;
    solid_[solid_idx(local_x, ny_-1)]  = 1;
  }
  mark_obstacle(cyl_x, cyl_y, cyl_r);
  cylinders_.push_back({cyl_x, cyl_y, cyl_r}); // Store the placed cylinder position.
}

// ADD SECOND OBSTACLE.
void
LBM::add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r)
{
  if (cyl2_r > 0.0) {
    mark_obstacle(cyl2_x, cyl2_y, cyl2_r);
    cylinders_.push_back({cyl2_x, cyl2_y, cyl2_r}); // Store the placed cylinder position.
  }
}

// DEFINE CIRCULAR OBSTACLE, AND MARK THE INSIDE AS SOLID CELLS.
void
LBM::mark_obstacle(double c_x, double c_y, double r)
{
  const double r2 = r * r;
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t local_x = 1; local_x <= nx_local_; ++local_x) {
      const double global_x = double(nx_start_) + double(local_x) - 1.0; // Since nx_start_ already represents the first position (adding e.g. local_x=1 adds an extra 1, which is corrected by the -1.0 at the end)
      const double dx = global_x - c_x;
      const double dy = double(y) - c_y;
      if (dx * dx + dy * dy <= r2) {
        solid_[solid_idx(local_x, y)] = 1; // if inside circle, mark solid.
      }
    }
  }
}


void
LBM::initialize()
{
  const std::size_t N = (nx_local_+2) * ny_;
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t local_x = 1; local_x <= nx_local_; ++local_x) { // for every local cell:
      const std::size_t k = cell_idx(local_x, y);
      const double rho = 1.0; // set initial uniform density.
      const double ux  = solid_[solid_idx(local_x, y)] ? 0.0 : u_in_; // set initial horizontal velocity: u_in in fluid, 0 in solid.
      const double uy  = 0.0; // no vertical velocity.
      const double u2  = ux * ux + uy * uy;
      for (int i = 0; i < Q; ++i) { // for every direction in that cell,
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2); // compute its eq. distribution,
        f_[i*N + k] = feq; // and set it as initial distribution.
      }
    }
  }
}

// PERFORM A STEP IN THE SIMULATION.
void
LBM::step()
{
  collide();
  bounce_back();
  update_bounds();
  stream();
  if (rank_ == 0) apply_inlet();
  if (rank_ == size_-1) apply_outlet();
}

void
LBM::collide()
{
  const std::size_t N = (nx_local_+2) * ny_;
  const double inv_tau = 1.0 / tau_;

  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t local_x = 1; local_x <= nx_local_; ++local_x) { // Only fluid cells!
      const std::size_t k = cell_idx(local_x, y);
      double rho = 0.0, mx = 0.0, my = 0.0;
      for (int i = 0; i < Q; ++i) {
        const double fi = f_[i*N + k];
        rho += fi;
        mx  += cx[i]*fi;
        my  += cy[i]*fi;
      }
      const double ux = (rho > 0.0) ? mx / rho : 0.0;
      const double uy = (rho > 0.0) ? my / rho : 0.0;
      const double u2 = ux*ux + uy*uy;

      for (int i = 0; i < Q; ++i) {
        const double cu  = cx[i]*ux + cy[i]*uy;
        const double feq = w[i]*rho*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);
        f_[i*N + k] += -inv_tau * (f_[i*N + k] - feq);
      }
    }
  }
}

void
LBM::bounce_back()
{
  const std::size_t N = (nx_local_+2) * ny_;
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t local_x = 1; local_x <= nx_local_; ++local_x) {
      const std::size_t k = cell_idx(local_x, y);
      if (!solid_[solid_idx(local_x, y)]) continue;
      std::swap(f_[1*N + k], f_[3*N + k]);
      std::swap(f_[2*N + k], f_[4*N + k]);
      std::swap(f_[5*N + k], f_[7*N + k]);
      std::swap(f_[6*N + k], f_[8*N + k]);
    }
  }
}

void
LBM::update_bounds()
{
  // This was changed, so that every worker only computes on its local domain.
  const std::size_t N = (nx_local_+2) * ny_;
  
  // Send the information from domain boundaries to neighbouring workers, and only in the relevant directions!
  MPI_Datatype column; // Define the column datatype to be sent.
  MPI_Type_vector(ny_, 1, nx_local_+2, MPI_DOUBLE, &column);
  MPI_Type_commit(&column);

  int left_nbr  = (rank_ > 0)       ? rank_-1 : MPI_PROC_NULL;
  int right_nbr = (rank_ < size_-1) ? rank_+1 : MPI_PROC_NULL;

  // Exchange left-moving populations (3, 6, 7)
  // Send column x=1 to left neighbor, receive column x=nx_local_+1 from right neighbor
  for (int i : {3,6,7}) {
    MPI_Sendrecv(
      &f_[i*N + 1], 1, column, left_nbr, 0, 
      &f_[i*N + nx_local_+1], 1, column, right_nbr, 0,
      MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
  }
  
  // Exchange right-moving populations (1, 5, 8)
  // Send column x=nx_local_ to right neighbor, receive column x=0 from left neighbor
  for (int i : {1,5,8}) {
    MPI_Sendrecv(
      &f_[i*N + nx_local_], 1, column, right_nbr, 1,
      &f_[i*N + 0], 1, column, left_nbr, 1,
      MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
  }

  MPI_Type_free(&column);
}

void
LBM::stream()
{
  const std::size_t N = (nx_local_+2) * ny_;

  for (int i = 0; i < Q; ++i) {
    for (std::size_t y = 0; y < ny_; ++y) {
      for (std::size_t local_x = 1; local_x <= nx_local_; ++local_x) {
        const long sx = long(local_x) - cx[i];
        const long sy = long(y) - cy[i];

        // If the source is outside the domain:
        if (
          (sy < 0) ||                                  // below the lower wall,
          (sy >= long(ny_)) ||                         // above the upper wall,
          ((rank_ == 0)&&(sx < 1)) ||                  // at the left of the left most wall,
          ((rank_ == size_-1)&&(sx > long(nx_local_))) // at the right of the right most wall,
        ) {
          // Ignore stream.
          ftmp_[i*N + cell_idx(local_x, y)] = f_[i*N + cell_idx(local_x, y)];
        } else {
          // Else, normal stream takes place.
          ftmp_[i*N + cell_idx(local_x, y)] = f_[i*N + cell_idx(std::size_t(sx), std::size_t(sy))];
        }
      }
    }
  }
  f_.swap(ftmp_);
}

void
LBM::apply_inlet()
{
  // Apply inlet at the left edge (x=1, y=y)
  const std::size_t N = (nx_local_+2) * ny_;
  const std::size_t x_first = 1;
  for (std::size_t y = 0; y < ny_; ++y) {
    if (solid_[solid_idx(x_first, y)]) continue;
    const double rho = 1.0;
    const double ux  = u_in_;
    const double uy  = 0.0;
    const double u2  = ux*ux + uy*uy;
    for (int i = 0; i < Q; ++i) {
      const double cu  = cx[i]*ux + cy[i]*uy;
      f_[i*N + cell_idx(x_first, y)] = w[i]*rho*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);
    }
  }
}

void
LBM::apply_outlet()
{
  // Apply outlet at the right edge (x=nx_local_, y=y) (nx_local_ corresponds to right lim, since only last rank will execute this, and nx_local_ corresponds to last column within its physical boundary).
  if (nx_ < 2) return;
  const std::size_t N  = (nx_local_+2) * ny_;
  const std::size_t x_last = nx_local_;
  const std::size_t x_sec_last = nx_local_-1;
  for (std::size_t y = 0; y < ny_; ++y) {
    for (int i = 0; i < Q; ++i) {
      f_[i*N + cell_idx(x_last, y)] = f_[i*N + cell_idx(x_sec_last, y)];
    }
  }
}

double
LBM::rho(std::size_t local_x, std::size_t y) const
{
  const std::size_t N = (nx_local_+2) * ny_;
  double r = 0.0;
  for (int i = 0; i < Q; ++i) r += f_[i*N + cell_idx(local_x, y)];
  return r;
}

double
LBM::ux(std::size_t local_x, std::size_t y) const
{
  const std::size_t N = (nx_local_+2) * ny_;
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[i*N + cell_idx(local_x, y)];
    r += fi;
    m += cx[i]*fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::uy(std::size_t local_x, std::size_t y) const
{
  const std::size_t N = (nx_local_+2) * ny_;
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[i*N + cell_idx(local_x, y)];
    r += fi;
    m += cy[i]*fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}


bool LBM::is_solid_global(std::size_t global_x, std::size_t y) const {
  // Check top and bottom walls
  if (y == 0 || y == ny_-1) return true;

  // Check the (maximum two) stored cylinders
  for (const auto& cyl : cylinders_) {
    const double dx = double(global_x) - cyl.x;
    const double dy = double(y) - cyl.y;
    if (dx*dx + dy*dy <= cyl.r*cyl.r) {
      return true;
    }
  }
  return false;
}


// Function that gathers the data in a snapshot
void
LBM::gather_local_results(std::vector<double>& g_rho, std::vector<double>& g_ux, 
                          std::vector<double>& g_uy, std::vector<double>& g_vor) const 
{
  // Extract local fluid data into 1D contiguous buffers
  std::vector<double> l_rho(nx_local_*ny_), l_ux(nx_local_*ny_), l_uy(nx_local_*ny_);

  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t local_x = 1; local_x <= nx_local_; ++local_x) {
      std::size_t glob_idx = y*nx_local_ + (local_x-1);
      l_rho[glob_idx] = rho(local_x, y);
      l_ux[glob_idx]  = ux(local_x, y);
      l_uy[glob_idx]  = uy(local_x, y);
    }
  }

  // Prepare them to be sent (idea: have them all together in a single structure) : compute size and offset
  std::vector<int> counts(size_), displs(size_);
  for (int p = 0; p < size_; ++p) {
    int nx_loc = (nx_ / size_) + (p < int(nx_%size_) ? 1 : 0); // Same logic as nx_local_
    counts[p] = nx_loc*ny_;
    displs[p] = (p == 0) ? 0 : displs[p-1] + counts[p-1];
  }

  // Allocate receive buffers ONLY on rank == 0
  std::vector<double> recv_rho, recv_ux, recv_uy, recv_vor;
  if (rank_ == 0) {
    recv_rho.resize(nx_*ny_); 
    recv_ux.resize(nx_*ny_);
    recv_uy.resize(nx_*ny_);
  }

  // Gather the raw arrays across the network
  MPI_Gatherv(l_rho.data(), counts[rank_], MPI_DOUBLE, recv_rho.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(l_ux.data(),  counts[rank_], MPI_DOUBLE, recv_ux.data(),  counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(l_uy.data(),  counts[rank_], MPI_DOUBLE, recv_uy.data(),  counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Unpack vertical data (non-trivial indexing) into global grid (only on Rank 0, with straightforward indexing).
  if (rank_ == 0) {
    g_rho.assign(nx_*ny_, 0.0);
    g_ux.assign(nx_*ny_, 0.0);
    g_uy.assign(nx_*ny_, 0.0);
    g_vor.assign(nx_*ny_, 0.0);
    
    for (int p = 0; p < size_; ++p) {
      int nx_loc = (nx_ / size_) + (p < int(nx_ % size_) ? 1 : 0);        // Same logic as nx_local_
      int nx_loc_start = (p*(nx_ / size_)) + std::min(p, int(nx_ % size_)); // Same logic as nx_start_
      int offset = displs[p];
      
      for (std::size_t y = 0; y < ny_; ++y) {
        for (int x = 0; x < nx_loc; ++x) {
          std::size_t glob_idx = y*nx_ + (nx_loc_start + x);
          std::size_t gath_idx = offset + y*nx_loc + x;
          
          g_rho[glob_idx] = recv_rho[gath_idx];
          g_ux[glob_idx]  = recv_ux[gath_idx];
          g_uy[glob_idx]  = recv_uy[gath_idx];
        }
      }
    }

    // Compute vorticity globally, with the global rho, ux, uy structures
    for (std::size_t y = 0; y < ny_; ++y) {
      for (std::size_t x = 0; x < nx_; ++x) {
        std::size_t idx = y*nx_ + x;
        
        if (x == 0 || x == nx_-1 || y == 0 || y == ny_-1) {
          g_vor[idx] = 0.0;
        } else {
          double uy_right = g_uy[idx+1];   // Right element (x --> x+1)
          double uy_left  = g_uy[idx-1];   // Left  element (x --> x-1)
          double ux_up    = g_ux[idx+nx_]; // Upper element (y --> y+1)
          double ux_down  = g_ux[idx-nx_]; // Lower element (y --> y-1)
            
          g_vor[idx] = 0.5 * ((uy_right-uy_left) - (ux_up-ux_down));
        }
      }
    }
  }
}