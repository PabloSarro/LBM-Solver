#include "lbm_MPI_OMP.hh"
#include "output_MPI_OMP.hh"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <mpi.h>

// ======================================================== //
// ====================== main_MPI.cc ===================== //
// ======================================================== //

namespace {

using Args = std::unordered_map<std::string, std::string>;

Args
parse_args(int argc, char ** argv)
{
  Args kv;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto eq = a.find('=');
    if (eq == std::string::npos) {
      std::cerr << "Bad argument '" << a << "' (expected key=value)\n";
      std::exit(1);
    }
    kv[a.substr(0, eq)] = a.substr(eq + 1);
  }
  return kv;
}

template <typename T>
T
get(const Args & kv, const std::string & key, T def)
{
  auto it = kv.find(key);
  if (it == kv.end()) return def;
  std::istringstream iss(it->second);
  T v; iss >> v;
  return v;
}

std::string
get_string(const Args & kv, const std::string & key, const std::string & def)
{
  auto it = kv.find(key);
  return (it == kv.end()) ? def : it->second;
}

}

int
main(int argc, char ** argv)
{
  // Initialise MPI.
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  const Args kv = parse_args(argc, argv);

  // Grid + physics.
  const std::size_t nx    = get<std::size_t>(kv, "nx",    800);
  const std::size_t ny    = get<std::size_t>(kv, "ny",    400);
  const double      Re    = get<double>     (kv, "re",    100.0);
  const double      u_in  = get<double>     (kv, "u_in",  0.05);
  const std::size_t steps = get<std::size_t>(kv, "steps", 60000);

  const double cx0 = get<double>(kv, "cyl_x", double(nx) * 0.25);
  const double cy0 = get<double>(kv, "cyl_y", double(ny) * 0.50);
  const double cr0 = get<double>(kv, "cyl_r", double(ny) * 0.025);

  // Optional second cylinder. Disabled if cyl2_r <= 0.
  const double cx1 = get<double>(kv, "cyl2_x", -1.0);
  const double cy1 = get<double>(kv, "cyl2_y", -1.0);
  const double cr1 = get<double>(kv, "cyl2_r", -1.0);

  // Output.
  const std::size_t every     = get<std::size_t>(kv, "every", 500);
  const std::string out_pref  = get_string(kv, "out", "out/lbm");
  const std::string probe_csv = get_string(kv, "probe", "probe.csv");

  // Probe location: note that now only 1 rank will "own" this cell! Corrected logic below
  const std::size_t px = get<std::size_t>(kv, "probe_x", std::size_t(cx0 + 8.0 * cr0));
  const std::size_t py = get<std::size_t>(kv, "probe_y", std::size_t(cy0));

  LBM solver(nx, ny, u_in, Re, cx0, cy0, cr0, rank, size); // Added rank and size for MPI.
  if (cr1 > 0.0) solver.add_second_cylinder(cx1, cy1, cr1);
  solver.initialize();

  if (rank == 0) {
    std::cout << "LBM 2D D2Q9 BGK\n"
              << "  grid          : " << nx << " x " << ny << "\n"
              << "  Re            : " << Re << "\n"
              << "  u_in          : " << u_in << "\n"
              << "  tau           : " << solver.tau() << "\n"
              << "  cylinder      : (" << cx0 << ", " << cy0
              << "), r = " << cr0 << "\n";
    if (cr1 > 0.0)
      std::cout << "  cylinder #2   : (" << cx1 << ", " << cy1
                << "), r = " << cr1 << "\n";
    std::cout << "  steps         : " << steps << "\n"
              << "  output every  : " << every << " (0 = off)\n"
              << "  output prefix : " << out_pref << "\n"
              << "  probe at      : (" << px << ", " << py << ")\n"
              << "  probe csv     : " << probe_csv << "\n";
    std::cout << "Launching simulation with " << size << " MPI processes.\n";
  }

  std::ofstream probe;
  if (rank == 0) {
    probe.open(probe_csv);
    probe << "step,ux,uy\n";
  }

  XDMFWriter writer(out_pref, nx, ny);
  if (every > 0) writer.write_mask(solver);

  using clk = std::chrono::high_resolution_clock;
  const auto t0 = clk::now();

  for (std::size_t step = 1; step <= steps; ++step) {
    solver.step();

    // Now we need to find which MPI process contains the (px,py) value, and reduce info on probe to Rank 0
    double local_probe[2] = {0.0, 0.0};
    if (px >= solver.nx_start() && px < solver.nx_start() + solver.nx_local()) {
        std::size_t local_px = px - solver.nx_start() + 1; // Map global px to local_x
        local_probe[0] = solver.ux(local_px, py);
        local_probe[1] = solver.uy(local_px, py);
    }

    double global_probe[2] = {0.0, 0.0};
    MPI_Reduce(local_probe, global_probe, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      probe << step << ',' << global_probe[0] << ',' << global_probe[1] << '\n';
    }

    if (every > 0 && step % every == 0) {
      writer.write_snapshot(solver, double(step));
      if (rank == 0) {
        std::cout << "\r  step " << step << " / " << steps << std::flush;
      }
    }
  }

  const double dt    = std::chrono::duration<double>(clk::now() - t0).count();
  const double mlups = double(nx) * double(ny) * double(steps) / dt / 1.0e6;
  
  if (rank == 0) {
    if (every > 0) std::cout << "\n";
    std::cout << "Wall time : " << dt    << " s\n"
              << "MLUPS     : " << mlups << "\n";
  }
  
  MPI_Finalize();

  return 0;
}
