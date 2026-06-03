#ifndef LBM_HH
#define LBM_HH

#include <cstddef>
#include <cstdint>
#include <vector>

// ======================================================== //
// ====================== lbm_MPI.hh ====================== //
// ======================================================== //

/**
 * @brief 2D Lattice Boltzmann solver, D2Q9 lattice with BGK collision.
 *
 * The solver simulates incompressible flow past one (or two) circular
 * cylinders inside a rectangular channel with no-slip top and bottom walls.
 * The inlet (x = 0) prescribes a uniform horizontal velocity; the outlet
 * (x = nx - 1) is a simple zero-gradient copy from the column to its left.
 *
 * Distributions are stored in structure-of-arrays layout:
 *   f_[ i * (nx*ny) + y*nx + x ]   for direction i in [0, 9).
 *
 * Lattice units are used throughout (dx = dt = 1, c = 1, c_s^2 = 1/3).
 */
class LBM
{
public:
  static constexpr int Q = 9;

  static const int    cx[Q];   ///< Discrete velocity x-components.
  static const int    cy[Q];   ///< Discrete velocity y-components.
  static const double w[Q];    ///< Equilibrium weights.

  /**
   * @param nx       Number of cells along x.
   * @param ny       Number of cells along y.
   * @param u_in     Inlet velocity in lattice units (must be << 1/sqrt(3)).
   * @param Re       Target Reynolds number, based on cylinder diameter.
   * @param cyl_x    Center of the (first) cylinder along x, in cell units.
   * @param cyl_y    Center of the (first) cylinder along y, in cell units.
   * @param cyl_r    Radius of the (first) cylinder, in cell units.
   * @param rank     MPI rank of the current process.
   * @param size     Total number of MPI processes.
   */

  LBM(std::size_t nx, std::size_t ny,
      double u_in, double Re,
      double cyl_x, double cyl_y, double cyl_r,
      int rank, int size);

  /// Add a second circular obstacle. No-op if r2 <= 0.
  void add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r);

  /// Set f to the equilibrium distribution with rho = 1, u = (u_in, 0)
  /// on every fluid cell, and (0, 0) on solid cells.
  void initialize();

  /// Advance the simulation by one time step.
  void step();

  // Accessors.
  double rho      (std::size_t local_x, std::size_t y) const;
  double ux       (std::size_t local_x, std::size_t y) const;
  double uy       (std::size_t local_x, std::size_t y) const;

  // Getters.
  std::size_t nx()       const { return nx_; }
  std::size_t ny()       const { return ny_; }
  std::size_t nx_local() const { return nx_local_; }
  std::size_t nx_start() const { return nx_start_; }

  double      tau()  const { return tau_; }
  double      u_in() const { return u_in_; }
  int         rank() const { return rank_; }

  // Output.cc needed functions.
  bool is_solid_global(std::size_t global_x, std::size_t y) const;
  void gather_local_results(std::vector<double>& g_rho, std::vector<double>& g_ux, 
                            std::vector<double>& g_uy, std::vector<double>& g_vor) const;

private:
  std::size_t solid_idx (std::size_t local_x, std::size_t y)  const { return y*nx_local_ + (local_x-1); } // Since for solid structure, I don't allocate any "extra space".
  std::size_t cell_idx (std::size_t local_x, std::size_t y)   const { return y*(nx_local_+2) + local_x; } // However, here I add two extreme "redundant" cols, where I don't write to, but just read.

  void mark_obstacle (double c_x, double c_y, double r);
  void collide       ();
  void bounce_back   ();
  void update_bounds ();
  void stream        ();
  void apply_inlet   ();
  void apply_outlet  ();

  std::size_t nx_, ny_;
  double u_in_;
  double tau_;

  // MPI added parameters.
  int rank_;
  int size_;
  std::size_t nx_local_;
  std::size_t nx_start_;

  std::vector<double>  f_;      ///< Current distributions, size 9*nx*ny.
  std::vector<double>  ftmp_;   ///< Scratch buffer for streaming.
  std::vector<uint8_t> solid_;  ///< 0 = fluid, 1 = solid.  Size nx*ny.

  // Store the added cylinders to the system.
  struct Cylinder { double x, y, r; };
  std::vector<Cylinder> cylinders_;
};

#endif  // LBM_HH
