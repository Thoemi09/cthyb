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
#ifndef TRIQS_CTQMC_KRYLOV_FUNDAMENTAL_OPERATOR_SET
#define TRIQS_CTQMC_KRYLOV_FUNDAMENTAL_OPERATOR_SET
#include <utility>
#include <vector>
#include <unordered_map>
#include "tuple_hasher.hpp"

namespace triqs { namespace app { namespace impurity_solvers { namespace ctqmc_krylov {

// This class defines the list of operators that are used to describe
// e.g. Fock states etc.

 template<typename ...IndexType>
class fundamental_operator_set {

  // The index of the operators
  typedef std::tuple<IndexType...> tuple_t;

  // the hash table index <-> n
  typedef std::unordered_map<tuple_t, int, tuple_hasher<tuple_t>> map_t;
  map_t map_index_n;

  public:

  fundamental_operator_set() {}

  // constructor on a vector which gives the number of alpha's for every a
  // this only makes sense if the indices are two integers
  template<typename T = tuple_t>
   fundamental_operator_set( TYPE_ENABLE_IF(std::vector<int>, std::is_same<T, std::tuple<int,int>>) const & v) {
    for(int i=0; i<v.size(); ++i) 
     for(int j=0; j<v[i]; ++j) 
      add_operator(i,j); 
   }

  // add an operator in the set
  void add_operator(IndexType const & ...ind) {
   map_index_n.insert(std::make_pair(std::make_tuple(ind...), n_operators()));
  }

  // return the dimension of the space spanned by the operators
  size_t dimension() const { return 1ull << n_operators(); } // 2^ n_ops

  // return the number of operators
  size_t n_operators() const { return map_index_n.size(); }

  // flatten (a,alpha) --> n
  template<typename ...Indices>
   size_t index_to_n(Indices const & ...ind) const { return map_index_n.at(std::make_tuple(ind...)); }

    // flatten (a,alpha) --> n
  template<typename ...Indices>
   size_t index_tuple_to_linear(std::tuple<Indices...> const & t) const { return map_index_n.at(t);}

  // iterator on the tuples
  typedef typename map_t::const_iterator const_iterator;
  const_iterator begin()  const noexcept { return map_index_n.begin(); }
  const_iterator end()    const noexcept { return map_index_n.end(); }
  const_iterator cbegin() const noexcept { return map_index_n.cbegin(); }
  const_iterator cend()   const noexcept { return map_index_n.cend(); }
};
}}}}
#endif