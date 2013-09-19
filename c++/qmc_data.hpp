/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2013 by I. Krivenko, M. Ferrero, O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#ifndef TRIQS_CTQMC_KRYLOV_QMC_DATA_H
#define TRIQS_CTQMC_KRYLOV_QMC_DATA_H

#include "./atomic_correlators_worker.hpp"
#include <triqs/gfs/block.hpp>
#include <triqs/gfs/imtime.hpp>
#include <triqs/det_manip/det_manip.hpp>
#include <triqs/h5.hpp>
#include <triqs/utility/serialization.hpp>

namespace triqs { namespace app { namespace impurity_solvers { namespace ctqmc_krylov {

 /**
  * The data of the Monte carlo
  */
 struct qmc_data {

  // the configuration and the worker to compute the trace...
  configuration config;
  sorted_spaces sosp;
  atomic_correlators_worker atomic_corr;

  // the trace type
  typedef atomic_correlators_worker::result_t trace_t;

  typedef gfs::gf_view<gfs::imtime> delta_block_t;
  typedef gfs::gf_view<gfs::block_index,gfs::gf<gfs::imtime>> delta_t;
  
  /// This callable object adapts the Delta function for the call of the det.
  struct delta_block_adaptor {

   delta_block_t delta_block;

   delta_block_adaptor (delta_block_t const & delta_block) : delta_block(delta_block) {}
   delta_block_adaptor (delta_block_adaptor const &) = default;
   delta_block_adaptor (delta_block_adaptor &&) = default;
   delta_block_adaptor & operator=(delta_block_adaptor const &) = delete;     // forbid assignment
   delta_block_adaptor & operator=(delta_block_adaptor && a) noexcept { delta_block.rebind(a.delta_block); return *this; }

   // no need of argument_type, return_type : det_manip now synthetize everything (need to UPDATE doc).
   double operator()(std::pair<time_pt, int> const & x, std::pair<time_pt, int> const & y) const {
    double res = delta_block[delta_block.mesh().nearest_index(double(x.first - y.first))](x.second, y.second);
    return (x.first >= y.first ? res : -res); //x,y first are time_pt, the wrapping is automatic in the - operation, but need to compute the sign
   }
  };

  // The determinants
  std::vector<det_manip::det_manip<delta_block_adaptor>> dets;
  
  // Permutation prefactor
  int current_sign, old_sign;

  // The current value of the trace
  trace_t trace;

  // construction and the basics stuff. value semantics, except = ?
  qmc_data(utility::parameters const& p, sorted_spaces const & sosp, const delta_t delta):
   config (p["beta"], sosp, p["krylov_bs_use_cutoff"], p["krylov_bs_prob_cutoff"]),
   sosp(sosp),
   atomic_corr (config, sosp,
     p["krylov_gs_energy_convergence"],
     p["krylov_small_matrix_size"]
   ),
   current_sign(1), old_sign(1)
   {
    trace = atomic_corr();
    
    dets.clear();
      for (auto const& bl : delta.mesh())
          dets.push_back(det_manip::det_manip<delta_block_adaptor>(delta_block_adaptor(delta[bl]),100));
   }

  qmc_data (qmc_data const &) = default;
  qmc_data & operator =(qmc_data const & ) = delete;
  
  void update_sign() {

    int s =0;
    size_t num_blocks = dets.size();
    vector<int> n_op_with_a_equal_to(num_blocks,0),
                n_ndag_op_with_a_equal_to(num_blocks,0);

    // In this first part we compute the sign to bring the configuration to
    // d^_1 d^_1 d^_1 ... d_1 d_1 d_1   d^_2 d^_2 ... d_2 d_2   ...   d^_n .. d_n

    // loop over the operators "op" in the trace (right to left)
    for (auto op = config.oplist.begin(); op != config.oplist.end(); ++op) {

      // how many operators with an 'a' larger than "op" are there on the left of "op"?
      for (int a = op->second.block_index+1; a < num_blocks; ++a) s += n_op_with_a_equal_to[a];
      n_op_with_a_equal_to[op->second.block_index]++;

      // if "op" is not a dagger how many operators of the same a but with a dagger are there on his right?
      if (op->second.dagger)
        s += n_ndag_op_with_a_equal_to[op->second.block_index];
      else
        n_ndag_op_with_a_equal_to[op->second.block_index]++;
    }

    // Now we compute the sign to bring the configuration to
    // d_1 d^_1 d_1 d^_1 ... d_1 d^_1   ...   d_n d^_n ... d_n d^_n
    for (int block_index = 0; block_index < num_blocks; block_index++) {
      int n = dets[block_index].size();
      s +=  n*(n+1)/2;
    }

    old_sign = current_sign;
    current_sign = (s%2==0 ? 1 : -1);
  }
  
  // h5 interface for checkpointing & crash logging
  friend void h5_write(h5::group fg, std::string name, qmc_data const & obj) {
  // h5_write (fg, name, triqs::serialize(obj));
  }
  friend void h5_read(h5::group fg, std::string name, qmc_data & obj){
   //std::string s; h5_read (fg, name, s); obj = triqs::deserialize<qmc_data>(s);
  }

  template<class Archive> void serialize(Archive & ar, const unsigned int version) {
    ar & boost::serialization::make_nvp("configuration", config)
       & boost::serialization::make_nvp("dets",dets)
       & boost::serialization::make_nvp("atomic_corr",atomic_corr)
       & boost::serialization::make_nvp("old_sign",old_sign)
       & boost::serialization::make_nvp("current_sign",current_sign);
   }    

 };

}}}}
#endif
