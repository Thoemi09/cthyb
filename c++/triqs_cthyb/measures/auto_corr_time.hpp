/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2021, Simons Foundation
 *    author: N. Wentzell
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
#pragma once

#include <triqs/stat/lin_binning.hpp>

#include "../qmc_data.hpp"

#include <tuple>

namespace triqs_cthyb {

  /// Measurement auto-correlation time based on the partition function
  struct measure_auto_corr_time {
    using acc_t = triqs::stat::lin_binning<std::complex<double>>;

    measure_auto_corr_time(qmc_data const &_data, double &_auto_corr_time) : data(_data), auto_corr_time(_auto_corr_time) {}

    void accumulate(mc_weight_t sign) {
      sign_acc << sign;
      order_acc << data.config.size();
    }

    void collect_results(mpi::communicator const &comm) {
      double sign_tau = 0.0;
      std::tie(std::ignore, std::ignore, sign_tau) = sign_acc.mean_error_and_tau(comm);

      double order_tau = 0.0;
      std::tie(std::ignore, std::ignore, order_tau) = order_acc.mean_error_and_tau(comm);

      auto_corr_time = std::max(sign_tau, order_tau);
    }

    private:
    qmc_data const &data;
    double &auto_corr_time;

    // Initialize one complex log accumulator for each observable to use for the autocorrelation analysis
    acc_t sign_acc{1.0, 256, 1};
    acc_t order_acc{1.0, 256, 1};
  };

} // namespace triqs_cthyb
