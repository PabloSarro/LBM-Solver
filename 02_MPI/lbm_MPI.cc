#include "lbm_MPI.hh"

//  

#include <algorithm>

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

// INDICES OF OPPOSITE DIRECTIONS.
const int LBM::opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};


// CONSTRUCTOR.
LBM::LBM(std::size_t nx, std::size_t ny,
         double u_in, double Re,
         double cyl_x, double cyl_y, double cyl_r,
         int rank, int size): 
    nx_(nx), ny_(ny), u_in_(u_in), tau_(0.0), // Number of cells, velocity and relaxation time.
    rank_(rank), size_(size),                 // MPI rank and size.
    
    // Distribute the remainder of floor(nx/size) by giving 1 to the first nx%size workers.
    nx_local_((nx / size) + (rank < int(nx % size) ? 1 : 0)),           // Number of cells in the x direction that each MPI worker will process (from 1 to nx_local).
    
    f_   (9 * (nx_local_+2) * ny_, 0.0),      // Flow distributions stored per worker (+2, since we will include cols 0 and nx_local_+1 for computation in the extremes).
    ftmp_(9 * (nx_local_+2) * ny_, 0.0),
    solid_(nx_local_*ny_, 0),                 // Solid cells (no need to allocate space for the extra cols, since we don't need them for the calculations).
{
  // RELAXATION TIME: τ
      // ν = c_s^2 (τ - 1/2) with c_s^2 = 1/3, and Re = u_in * D / ν.
  const double nu = u_in_ * (2.0 * cyl_r) / Re; // D = diameter = 2 * radius.
  tau_ = 3.0 * nu + 0.5;

  // DEFINE UPPER AND LOWER WALLS AS SOLID CELLS.
  for (std::size_t local_x = 1; local_x < nx_local_+1; ++local_x) {
    solid_[idx(local_x, 0)]      = 1;
    solid_[idx(local_x, ny_-1)]  = 1;
  }
  mark_obstacle(cyl_x, cyl_y, cyl_r);
}

// ADD SECOND OBSTACLE.
void
LBM::add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r)
{
  if (cyl2_r > 0.0) mark_obstacle(cyl2_x, cyl2_y, cyl2_r); // mark the inside as solid cells.
}

// DEFINE CIRCULAR OBSTACLE, AND MARK THE INSIDE AS SOLID CELLS.
void
LBM::mark_obstacle(double c_x, double c_y, double r)
{
  const double r2 = r * r;
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t local_x = 1; local_x < nx_local_+1; ++local_x) {
      const double global_x;
      if (rank_>=(nx_%size)) {
        global_x = (nx_%size)*nx_local_ + (rank_ - (nx_%size) - 1)*nx_local_ + local_x;
      } else {
        global_x = (rank_-1)*nx_local + local_x;
      }

      const double dx = global_x - c_x;
      const double dy = double(y) - c_y;
      if (dx * dx + dy * dy <= r2) {
        solid_[idx(local_x, y)] = 1; // if inside circle, mark solid.
      }
    }
  }
}

// INITIALISE INITIAL FLOW TO EQUILIBRIUM
void
LBM::initialize()
{
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t local_x = 1; local_x < nx_local_+1; ++local_x) { // for every local cell:
      const double rho = 1.0; // set initial uniform density.
      const double ux  = solid_[idx(local_x, y)] ? 0.0 : u_in_; // set initial horizontal velocity: u_in in fluid, 0 in solid.
      const double uy  = 0.0; // no vertical velocity.
      const double u2  = ux * ux + uy * uy;
      for (int i = 0; i < Q; ++i) { // for every direction in that cell,
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2); // compute its eq. distribution,
        f_[fidx(i, local_x, y)] = feq; // and set it as initial distribution.
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
  stream();
  if (rank_ == 0) apply_inlet();
  if (rank_ == size_ - 1) apply_outlet();
}

void
LBM::collide()
{
  const std::size_t N = (nx_local_+2) * ny_;
  const double inv_tau = 1.0 / tau_;

  for (std::size_t k = 0; k < N; ++k) {
    double rho = 0.0, mx = 0.0, my = 0.0;
    for (int i = 0; i < Q; ++i) {
      const double fi = f_[i * N + k];
      rho += fi;
      mx  += cx[i] * fi;
      my  += cy[i] * fi;
    }
    const double ux = (rho > 0.0) ? mx / rho : 0.0;
    const double uy = (rho > 0.0) ? my / rho : 0.0;
    const double u2 = ux * ux + uy * uy;

    for (int i = 0; i < Q; ++i) {
      const double cu  = cx[i] * ux + cy[i] * uy;
      const double feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
      f_[i * N + k] += -inv_tau * (f_[i * N + k] - feq);
    }
  }
}

void
LBM::bounce_back()
{
  // Fullway bounce-back: in solid cells, swap each pair of opposite directions.
  // Combined with subsequent streaming this reflects populations across the
  // solid-fluid interface.
  const std::size_t N = (nx_local_+2) * ny_;
  for (std::size_t k = 0; k < N; ++k) {
    if (!solid_[k]) continue;
    std::swap(f_[1 * N + k], f_[3 * N + k]);
    std::swap(f_[2 * N + k], f_[4 * N + k]);
    std::swap(f_[5 * N + k], f_[7 * N + k]);
    std::swap(f_[6 * N + k], f_[8 * N + k]);
  }
}

void
LBM::stream()
{
  // This was changed, so that every worker only computes on its local domain.
  const std::size_t N = (nx_local_+2) * ny_;
  // Send the information from domain boundaries to neighbouring workers, and only in the relevant directions!
  // 1. Define the strided column datatype
  MPI_Datatype column_type;
  // count = ny_, blocklength = 1, stride = nx_local_ + 2
  MPI_Type_vector(ny_, 1, nx_local_ + 2, MPI_DOUBLE, &column_type);
  MPI_Type_commit(&column_type);

  int left_nbr  = (rank_ > 0) ? rank_ - 1 : MPI_PROC_NULL;
  int right_nbr = (rank_ < size_ - 1) ? rank_ + 1 : MPI_PROC_NULL;

  // 2. Exchange left-moving populations (3, 6, 7)
  // Send column x=1 to left neighbor, receive column x=nx_local_+1 from right neighbor
  for (int i : {3, 6, 7}) {
      MPI_Sendrecv(&f_[i * N + 1], 1, column_type, left_nbr, 0,
                   &f_[i * N + nx_local_ + 1], 1, column_type, right_nbr, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // 3. Exchange right-moving populations (1, 5, 8)
  // Send column x=nx_local_ to right neighbor, receive column x=0 from left neighbor
  for (int i : {1, 5, 8}) {
      MPI_Sendrecv(&f_[i * N + nx_local_], 1, column_type, right_nbr, 0,
                   &f_[i * N + 0], 1, column_type, left_nbr, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  MPI_Type_free(&column_type);

  for (int i = 0; i < Q; ++i) {
    for (std::size_t y = 0; y < ny_; ++y) {
      for (std::size_t local_x = 1; local_x < nx_local_+1; ++local_x) {
        const long sx = long(local_x) - cx[i];
        const long sy = long(y) - cy[i];
        if (sx >= 0 && sx < long(nx_) && sy >= 0 && sy < long(ny_)) {
          ftmp_[i * N + idx(local_x, y)] = f_[i * N + idx(std::size_t(sx), std::size_t(sy))];
        } else {
          ftmp_[i * N + idx(local_x, y)] = f_[i * N + idx(local_x, y)];
        }
      }
    }
  }
  f_.swap(ftmp_);
}

void
LBM::apply_inlet()
{
  // Do we need to offset the x-access here? Only rank=0 is applying this, so rank*nx_local + 0 = 0.
  const std::size_t N = (nx_local_+2) * ny_;
  const std::size_t x = 1;
  for (std::size_t y = 0; y < ny_; ++y) {
    if (solid_[idx(x, y)]) continue;
    const double rho = 1.0;
    const double ux  = u_in_;
    const double uy  = 0.0;
    const double u2  = ux * ux + uy * uy;
    for (int i = 0; i < Q; ++i) {
      const double cu  = cx[i] * ux + cy[i] * uy;
      f_[i * N + idx(x, y)] = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }
  }
}

void
LBM::apply_outlet()
{
  // Here we don't need offset either, since it's the last rank and rank*nx_local + (nx_local-1) = nx-1?
  if (nx_ < 2) return;
  const std::size_t N  = (nx_local_+2) * ny_;
  const std::size_t x  = nx_local_ + 1;
  const std::size_t xs = nx_local_;
  for (std::size_t y = 0; y < ny_; ++y) {
    for (int i = 0; i < Q; ++i) {
      f_[i * N + idx(x, y)] = f_[i * N + idx(xs, y)];
    }
  }
}

double
LBM::rho(std::size_t x, std::size_t y) const
{
  const std::size_t N = (nx_local_+2) * ny_;
  double r = 0.0;
  for (int i = 0; i < Q; ++i) r += f_[i * N + idx(x, y)];
  return r;
}

double
LBM::ux(std::size_t x, std::size_t y) const
{
  const std::size_t N = (nx_local_+2) * ny_;
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[i * N + idx(x, y)];
    r += fi;
    m += cx[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::uy(std::size_t x, std::size_t y) const
{
  const std::size_t N = (nx_local_+2) * ny_;
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[i * N + idx(x, y)];
    r += fi;
    m += cy[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::vorticity(std::size_t x, std::size_t y) const
{
  if (x == 0 || x == nx_ - 1 || y == 0 || y == ny_ - 1) return 0.0;
  return 0.5 * ((uy(x + 1, y) - uy(x - 1, y)) - (ux(x, y + 1) - ux(x, y - 1)));
}

bool
LBM::is_solid(std::size_t x, std::size_t y) const
{
  return solid_[idx(x, y)] != 0;
}
