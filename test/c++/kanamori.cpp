#include "ctqmc.hpp"
#include <triqs/operators/many_body_operator.hpp>
#include <triqs/draft/hilbert_space_tools/fundamental_operator_set.hpp>
#include <triqs/gfs/local/fourier_matsubara.hpp>
#include <triqs/parameters.hpp>
#include <triqs/gfs/block.hpp>
#include <triqs/gfs/imtime.hpp>
#include <triqs/gfs/imfreq.hpp>

using namespace cthyb;
using triqs::utility::many_body_operator;
using triqs::utility::c;
using triqs::utility::c_dag;
using triqs::utility::n;
using triqs::params::parameters;
using namespace triqs::gfs;

int main(int argc, char* argv[]) {

  std::cout << "Welcome to the CTHYB solver\n";

  // Initialize mpi
  boost::mpi::environment env(argc, argv);
  int rank;
  {
    boost::mpi::communicator c;
    rank = c.rank();
  }

  // Parameters
  double beta = 10.0;
  int num_orbitals = 2;
  double mu = 1.0;
  double U = 2.0;
  double J = 0.2;
  double V = 1.0;
  double epsilon = 2.3;

  auto N = [] (std::string sn, int an) { return n(sn+'-'+std::to_string(an),0); }; 
  auto C = [] (std::string sn, int an) { return c(sn+'-'+std::to_string(an),0); }; 
  auto C_dag = [] (std::string sn, int an) { return c_dag(sn+'-'+std::to_string(an),0); }; 

  // Hamiltonian
  many_body_operator<double> H;
  for(int o = 0; o < num_orbitals; ++o){
      H += -mu*(N("up",o) + N("down",o));
  }
  for(int o = 0; o < num_orbitals; ++o){
      H += U *N("up",o)*N("down",o);
  }
  for(int o1 = 0; o1 < num_orbitals; ++o1)
  for(int o2 = 0; o2 < num_orbitals; ++o2){
      if(o1==o2) continue;
      H += (U-2*J)*N("up",o1)*N("down",o2);
  }
  for(int o1 = 0; o1 < num_orbitals; ++o1)
  for(int o2 = 0; o2 < num_orbitals; ++o2){
      if(o2>=o1) continue;
      H += (U-3*J)*N("up",o1)*N("up",o2);
      H += (U-3*J)*N("down",o1)*N("down",o2);
  }

  for(int o1 = 0; o1 < num_orbitals; ++o1)
  for(int o2 = 0; o2 < num_orbitals; ++o2){
      if(o1==o2) continue;
      H += -J*C_dag("up",o1)*C_dag("down",o1)*C("up",o2)*C("down",o2);
      H += -J*C_dag("up",o1)*C_dag("down",o2)*C("up",o2)*C("down",o1);
  }

  // quantum numbers
  std::vector<many_body_operator<double>> qn;

  // gf structure
  std::map<std::string, std::vector<int>> gf_struct; 
  for(int o = 0; o < num_orbitals; ++o){
    gf_struct["up-"+std::to_string(o)] = {0};
    gf_struct["down-"+std::to_string(o)] = {0};
  }

  // Construct CTQMC solver
  ctqmc solver(beta, gf_struct, 1000, 1000);

  // Set hybridization function
  triqs::clef::placeholder<0> om_;
  auto delta_w = gf<imfreq>{{beta, Fermion}, {1,1}};
  delta_w(om_) << V*V / (om_ - epsilon) + V*V / (om_ + epsilon);  
  for (int o = 0; o < 2*num_orbitals; ++o){
    solver.deltat_view()[o] = triqs::gfs::inverse_fourier(delta_w);
  }

  // Solve parameters
  auto p = ctqmc::solve_parameters();
  p["max_time"] = -1;
  p["random_name"] = "";
  p["random_seed"] = 123 * rank + 567;
  p["verbosity"] = 3;
  p["length_cycle"] = 50;
  p["n_warmup_cycles"] = 50;
  p["n_cycles"] = 500;

  // Solve!
  solver.solve(H, p, qn, true);
  
  // Save the results
  if(rank==0){
    H5::H5File G_file("kanamori.output.h5",H5F_ACC_TRUNC);
    for(int o = 0; o < num_orbitals; ++o) {
      std::stringstream bup; bup << "G_up-" << o;
      h5_write(G_file, bup.str(), solver.gt_view()[o]);
      std::stringstream bdown; bdown << "G_down-" << o;
      h5_write(G_file, bdown.str(), solver.gt_view()[num_orbitals+o]);
    }
  }

  return 0;

}
