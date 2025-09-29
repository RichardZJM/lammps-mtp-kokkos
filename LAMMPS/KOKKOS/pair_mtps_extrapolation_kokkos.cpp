/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

//
// Contributing author, Richard Meng, Queen's University at Kingston, 10.02.25, contact@richardzjm.com
//

#include "pair_mtps_extrapolation_kokkos.h"

#include "Kokkos_StdAlgorithms.hpp"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor_kokkos.h"

#include <csignal>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template <class DeviceType>
PairMTPsExtrapolationKokkos<DeviceType>::PairMTPsExtrapolationKokkos(LAMMPS(*lmp)) :
    PairMTPExtrapolation(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

  host_flag = (execution_space == Host);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> PairMTPsExtrapolationKokkos<DeviceType>::~PairMTPsExtrapolationKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_eatom, eatom);
  memoryKK->destroy_kokkos(k_vatom, vatom);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template <class DeviceType> void PairMTPsExtrapolationKokkos<DeviceType>::init_style()
{
  if (host_flag) {
    if (lmp->kokkos->nthreads > 1)
      error->all(
          FLERR,
          "Pair style mtp/extrapolation/kk/s can currently only run on a single CPU thread.");

    PairMTPExtrapolation::init_style();
    return;
  }

  if (force->newton_pair == 0) error->all(FLERR, "Pair style MTP requires newton pair on.");

  // neighbor list request for KOKKOS
  neighflag = lmp->kokkos->neighflag;

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> &&
                           !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
  if (neighflag == FULL)
    error->all(FLERR, "Must use half neighbor list style with pair mtp/extrapolation/kk/s.");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template <class DeviceType> double PairMTPsExtrapolationKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairMTPExtrapolation::init_one(i, j);
  //Don't need to do anything with the cutoff because the MTP (and original MLIP package) only uses one cutoff for all species combos.
  return cutone;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template <class DeviceType>
void PairMTPsExtrapolationKokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairMTPExtrapolation::coeff(narg, arg);
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template <class DeviceType>
void PairMTPsExtrapolationKokkos<DeviceType>::settings(int narg, char **arg)
{
  // We may need to process in chunks to deal with memory limitations
  // For now we expect the user to specify the chunk size
  if (narg != 3 && narg != 6)
    error->all(
        FLERR,
        "Pair mtp/extrapolation/kk/s requires 3 : {potential_file} \"chunksize\" {chunksize} "
        "Or 6 arguments: {potential_file} {output_file} {selection_threshold} "
        "{break_threshold} \"chunksize\" {chunksize}.");

  if (narg == 3) {
    if (LAMMPS_NS::utils::lowercase(arg[1]) != "chunksize")
      error->all(FLERR, "Chunksize not found, please specify \"chunksize\" {chunksize}.");
    input_chunk_size = utils::inumeric(FLERR, arg[2], true, lmp);
    PairMTPExtrapolation::settings(1, arg);
  }
  if (narg == 6) {
    if (LAMMPS_NS::utils::lowercase(arg[4]) != "chunksize")
      error->all(FLERR, "Chunksize not found, please specify \"chunksize\" {chunksize}.");
    input_chunk_size = utils::inumeric(FLERR, arg[5], true, lmp);
    PairMTPExtrapolation::settings(4, arg);
  }

  // Prepare check the alpha times waves
  PairMTPsExtrapolationKokkos::prepare_waves();

  // ---------- Now we move arrays to device ----------
  // First we set up the index lists
  MemKK::realloc_kokkos(d_alpha_index_basic, "mtp/extrapolation/kk/s:alpha_index_basic",
                        alpha_index_basic_count, 4);
  MemKK::realloc_kokkos(d_alpha_index_times, "mtp/extrapolation/kk/s:alpha_index_times",
                        alpha_index_times_count, 4);
  MemKK::realloc_kokkos(d_alpha_moment_mapping, "mtp/extrapolation/kk/s:alpha_moment_mapping",
                        alpha_scalar_count);

  // Setup the learned coefficients
  int radial_coeff_count = species_count * species_count * radial_basis_size * radial_func_count;
  MemKK::realloc_kokkos(d_radial_basis_coeffs, "mtp/extrapolation/kk/s:radial_coeffs",
                        radial_coeff_count);
  MemKK::realloc_kokkos(d_species_coeffs, "mtp/extrapolation/kk/s:species_coeffs", species_count);
  MemKK::realloc_kokkos(d_linear_coeffs, "mtp/extrapolation/kk/s:linear_coeffs",
                        alpha_scalar_count);

  // We need to init these as very small views to begin with because the user might specify a very large chunk_size which is much more than inum.
  //We will resize these as needed in compute.
  MemKK::realloc_kokkos(d_moment_jacobian, "mtp/extrapolation/kk/s:moment_jacobian", 1, 1,
                        alpha_index_basic_count, 3);
  MemKK::realloc_kokkos(d_radial_jacobian, "mtp/extrapolation/kk/s:radial_jacobian", 1,
                        alpha_index_basic_count, radial_coeff_count_per_pair * species_count);
  MemKK::realloc_kokkos(d_within_cutoff, "mtp/extrapolation/kk/s:within_cutoff", 1, 1);
  MemKK::realloc_kokkos(d_moment_tensor_vals, "mtp/extrapolation/kk/s:moment_tensor_vals", 1,
                        alpha_moment_count);
  MemKK::realloc_kokkos(d_nbh_energy_ders_wrt_moments,
                        "mtp/extrapolation/kk/s:nbh_energy_ders_wrt_moments", 1,
                        alpha_moment_count);

  //Declare host arrays
  auto h_alpha_index_basic = Kokkos::create_mirror_view(d_alpha_index_basic);
  auto h_alpha_index_times = Kokkos::create_mirror_view(d_alpha_index_times);
  auto h_alpha_moment_mapping = Kokkos::create_mirror_view(d_alpha_moment_mapping);
  auto h_radial_basis_coeffs = Kokkos::create_mirror_view(d_radial_basis_coeffs);
  auto h_species_coeffs = Kokkos::create_mirror_view(d_species_coeffs);
  auto h_linear_coeffs = Kokkos::create_mirror_view(d_linear_coeffs);

  //Populate the host arrays
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < alpha_index_basic_count; i++)
      h_alpha_index_basic(i, j) = alpha_index_basic[i][j];
    for (int i = 0; i < alpha_index_times_count; i++)
      h_alpha_index_times(i, j) = alpha_index_times[i][j];
  }
  for (int i = 0; i < alpha_scalar_count; i++) {
    h_alpha_moment_mapping(i) = alpha_moment_mapping[i];
    h_linear_coeffs(i) = linear_coeffs[i];
  }
  for (int i = 0; i < radial_coeff_count; i++) h_radial_basis_coeffs(i) = radial_basis_coeffs[i];
  for (int i = 0; i < species_count; i++) h_species_coeffs(i) = species_coeffs[i];

  // Peform the copy from host to device
  Kokkos::deep_copy(d_alpha_index_basic, h_alpha_index_basic);
  Kokkos::deep_copy(d_alpha_index_times, h_alpha_index_times);
  Kokkos::deep_copy(d_alpha_moment_mapping, h_alpha_moment_mapping);
  Kokkos::deep_copy(d_radial_basis_coeffs, h_radial_basis_coeffs);
  Kokkos::deep_copy(d_species_coeffs, h_species_coeffs);
  Kokkos::deep_copy(d_linear_coeffs, h_linear_coeffs);
  // No need to deep copy the working buffers.

  //Setup the inverse active set if nbh mode or
  // Or if we are calcing the cfg grade on device, (ie. not mpi splitted)
  if (!configuration_mode || comm->nprocs == 1) {
    MemKK::realloc_kokkos(d_inverse_active_set, "mtp/extrapolation/kk/s:inverse_active_set",
                          coeff_count, coeff_count);
    auto h_inverse_active_set = Kokkos::create_mirror_view(d_inverse_active_set);
    for (int i = 0; i < coeff_count; i++)
      for (int j = 0; j < coeff_count; j++) h_inverse_active_set(i, j) = inverse_active_set[i][j];
    Kokkos::deep_copy(d_inverse_active_set, h_inverse_active_set);

    if (!configuration_mode) {    // In neighbourhood mode only, we need memory to store grades
      MemKK::realloc_kokkos(d_nbh_extrapolation_grades, "mtp/extrapolation/kk/s:inverse_active_set",
                            1);    //We will resize as needed in compute.
    }
  }

  if (configuration_mode) {
    MemKK::realloc_kokkos(d_energy_ders_wrt_coeffs, "mtp/extrapolation/kk/s:energy_der_wrt_coeffs",
                          coeff_count);
    MemKK::realloc_kokkos(d_tmp_energy_ders_wrt_coeffs,
                          "mtp/extrapolation/kk/s:tmp_energy_der_wrt_coeffs", coeff_count);
  }
}

template <class DeviceType> void PairMTPsExtrapolationKokkos<DeviceType>::prepare_waves()
{
  // Finds the size of each alpha times waves.
  // The alpha times in the MLIP-3 format are already sorted by child node.
  int wave_number = 0;
  int last_max_node = alpha_index_basic_count - 1;
  int last_max_edge = 0;

  for (int i = 0; i < alpha_index_times_count; i++) {

    if (alpha_index_times[i][0] > last_max_node || alpha_index_times[i][1] > last_max_node) {
      if (wave_number == 2)
        error->all(FLERR,
                   "Error in the alpha times indicies! Only potentials trained from the MLIP-3 "
                   "templates are currently supported in mtp/kk/s.");
      wave_sizes[wave_number++] = i - last_max_edge;
      last_max_node = alpha_index_times[i - 1][3];
      last_max_edge = i;
    }
  }
  wave_sizes[2] = alpha_index_times_count - last_max_edge;
}

template <class DeviceType> void PairMTPsExtrapolationKokkos<DeviceType>::evaluate_grades()
{
  // Transfer the latest atom data and  grades to the host if needed
  if (extrapolation_flag || max_grade >= select_threshold) {

    // Sync atom positions, id, and types to the host for MLIP-3 style writing
    // If a lammps dump is used, sync is not needed.
    if (mlip3_style) atomKK->sync(Host, X_MASK | TYPE_MASK);

    if (!configuration_mode) {                          // If nbh mode, copy nbh grades to host
      if (!configuration_mode && nbh_count < inum) {    // Allocate more memory if needed
        memory->grow(nbh_extrapolation_grades, inum, "nbh_extrapolation_grades");
        nbh_count = inum;
      }
      Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          h_nbh_extrapolation_grades(nbh_extrapolation_grades, inum);
      Kokkos::deep_copy(h_nbh_extrapolation_grades, d_nbh_extrapolation_grades);
    }

    if (mlip3_style) write_config();
  }

  // Now process the max grade against the break threshold
  if (mlip3_style && max_grade >= break_threshold && comm->me == 0) {
    std::fflush(preselected_file);    // Ensure the writing buffers are flushed before breaking.
    std::fclose(preselected_file);
    error->one(FLERR, "Exceeded Break Threshold: {:.5f}. Terminating simulation.\n", max_grade);
  }
}

// Finds the maximum number of neighbours in all neigbhourhoods. This enables use to set the size (2nd index) of the jacobian. (Copied from other potentials)
template <class DeviceType> struct FindMaxNumNeighs {
  typedef DeviceType device_type;
  NeighListKokkos<DeviceType> k_list;

  FindMaxNumNeighs(NeighListKokkos<DeviceType> *nl) : k_list(*nl) {}
  ~FindMaxNumNeighs() { k_list.copymode = 1; }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &ii, int &max_neighs) const
  {
    const int i = k_list.d_ilist[ii];
    const int num_neighs = k_list.d_numneigh(i);
    if (max_neighs < num_neighs) max_neighs = num_neighs;
  }
};

/* ----------------------------------------------------------------------
   This version is a straightforward implementation
   ---------------------------------------------------------------------- */

template <class DeviceType>
void PairMTPsExtrapolationKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{

  // If we are running on host we just use the base implementation
  if (host_flag) {
    atomKK->sync(Host, X_MASK | F_MASK | TYPE_MASK);
    PairMTPExtrapolation::compute(eflag_in, vflag_in);
    atomKK->modified(Host, F_MASK);
    return;
  }

  max_grade = 0;

  // Determine if we are doing extrapolation grade this timestep.
  bool calculate_grade_this_step = (extrapolation_flag || mlip3_style);
  if (calculate_grade_this_step) max_grade = 0;

  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag, vflag, 0);

  // reallocate per-atom arrays if necessary
  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  copymode = 1;
  int newton_pair = force->newton_pair;
  if (newton_pair == false) error->all(FLERR, "PairMTPsExtrapolationKokkos requires 'newton on'.");

  // Now, ensure the atom data is synced
  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  inum = list->inum;

  need_dup = lmp->kokkos->need_dup<DeviceType>();
  // clang-format off
  if (need_dup) {
    dup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(f);
    dup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom);
  } else {
    ndup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(f);
    ndup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
    // clang-format on
  }

  //Precalc the max neighs. This is needed to resize the jacobian.
  max_neighs = 0;
  Kokkos::parallel_reduce("PairMTPsExtrapolationKokkos::find_max_neighs", inum,
                          FindMaxNumNeighs<DeviceType>(k_list), Kokkos::Max<int>(max_neighs));

  // Handling batching
  chunk_size = MIN(input_chunk_size,
                   inum);    // chunksize is the maximum atoms per pass as defined by the user
  chunk_offset = 0;

  // Team sizes. We specify 32 for 1 warp per thread block.
  int team_size_default = 1;
  int vector_length_default = 1;
  if (!host_flag) team_size_default = 64;

  // Resize the arrays to the chunksize if needed. Do not initialize values, we do so in the loop.
  if ((int) d_moment_tensor_vals.extent(0) < chunk_size) {
    Kokkos::realloc(Kokkos::WithoutInitializing, d_moment_tensor_vals, chunk_size,
                    alpha_moment_count);
    Kokkos::realloc(Kokkos::WithoutInitializing, d_nbh_energy_ders_wrt_moments, chunk_size,
                    alpha_moment_count);
    Kokkos::realloc(Kokkos::WithoutInitializing, d_radial_jacobian, chunk_size,
                    alpha_index_basic_count, species_count * radial_coeff_count_per_pair);
  }
  // Resize the jacobian and within _cutoff if max_neighs is too large. Do not initalize; first access is write.
  if ((int) d_moment_jacobian.extent(1) < chunk_size ||
      (int) d_moment_jacobian.extent(0) < max_neighs) {
    Kokkos::realloc(Kokkos::WithoutInitializing, d_moment_jacobian, max_neighs, chunk_size,
                    alpha_index_basic_count, 3);
    Kokkos::realloc(Kokkos::WithoutInitializing, d_within_cutoff, max_neighs, chunk_size);
  }

  // Resize nbh grades to inum not chunk size. The reduces host communication need. Only 1 FP64 per nbh.
  if (!configuration_mode && (int) d_nbh_extrapolation_grades.extent(0) < inum)
    Kokkos::realloc(Kokkos::WithoutInitializing, d_nbh_extrapolation_grades, inum);

  if (calculate_grade_this_step && configuration_mode) {    // Init the coeff ders if cfg mode
    typename Kokkos::RangePolicy<DeviceType, TagPairMTPInitCoeffDers> policy_coeff_init(
        0, coeff_count);
    Kokkos::parallel_for("InitCoeffDers", policy_coeff_init, *this);
  }

  EV_FLOAT ev;

  // ========== Begin Main Computation ==========
  while (chunk_offset < inum) {    // batching to prevent OOM on device
    EV_FLOAT ev_tmp;
    if (chunk_size > inum - chunk_offset) chunk_size = inum - chunk_offset;

    // ========== Init working views as 0  ==========
    {

      // Only init data needed for extrapolation on steps it's needed
      if (calculate_grade_this_step) {
        typename Kokkos::MDRangePolicy<Kokkos::Rank<2>, DeviceType, TagPairMTPInitRadJacobian>
            policy_rad_jac_init({0, 0}, {chunk_size, alpha_index_basic_count});
        Kokkos::parallel_for("InitRadJacobian", policy_rad_jac_init, *this);
      }
      typename Kokkos::MDRangePolicy<Kokkos::Rank<2>, DeviceType, TagPairMTPInitMomentValsDers>
          policy_moment_init({0, 0}, {alpha_moment_count, chunk_size});
      Kokkos::parallel_for("InitMomentValDers", policy_moment_init, *this);
    }

    // ========== Calculate the basic alphas (Per outer-atom parallelizaton) ==========
    {
      int team_size = team_size_default;
      int vector_length = vector_length_default;
      if (!host_flag && max_neighs < 32) team_size = 32;

      // Only calculate the radial jacobian on steps extrapolation is needed
      if (calculate_grade_this_step) {
        check_team_size_for<TagPairMTPComputeAlphaBasicRad>(chunk_size * max_neighs, team_size,
                                                            vector_length);
        int radial_scratch_count = 2 * (radial_func_count + radial_basis_size);
        int dist_coords_scratch_count = 4 * max_alpha_index_basic;
        // Reduce the scratch size to the max number of neighbors
        int scratch_size = scratch_size_helper<F_FLOAT>(
            min(team_size, max_neighs) * (radial_scratch_count + dist_coords_scratch_count));
        Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeAlphaBasicRad> policy_basic_alpha_rad(
            chunk_size, team_size);
        policy_basic_alpha_rad =
            policy_basic_alpha_rad.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
        Kokkos::parallel_for("ComputeAlphaBasicRad", policy_basic_alpha_rad, *this);
      } else {
        check_team_size_for<TagPairMTPComputeAlphaBasic>(chunk_size * max_neighs, team_size,
                                                         vector_length);
        int radial_scratch_count = 2 * (radial_func_count + radial_basis_size);
        int dist_coords_scratch_count = 4 * max_alpha_index_basic;
        // Reduce the scratch size to the max number of neighbors
        int scratch_size = scratch_size_helper<F_FLOAT>(
            min(team_size, max_neighs) * (radial_scratch_count + dist_coords_scratch_count));
        Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeAlphaBasic> policy_basic_alpha(chunk_size,
                                                                                       team_size);
        policy_basic_alpha = policy_basic_alpha.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
        Kokkos::parallel_for("ComputeAlphaBasic", policy_basic_alpha, *this);
      }
    }

    // ========== Calculate the non-elementary alphas ==========
    {
      int team_size = team_size_default;
      // Best team size depends on the max number of blocks per SM. 64 is good for CC8, and CC > 9+.
      Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeAlphaTimes> policy_basic_alpha(chunk_size,
                                                                                     team_size);
      Kokkos::parallel_for("ComputeAlphaTimes", policy_basic_alpha, *this);
    }

    // ========== Set the scalar nbh ders wrt moments ==========
    {
      typename Kokkos::MDRangePolicy<Kokkos::Rank<2>, DeviceType, TagPairMTPSetScalarNbhDers>
          policy_nbh_init({0, 0}, {alpha_scalar_count, chunk_size});
      Kokkos::parallel_for("SetScalarNbhDers", policy_nbh_init, *this);
    }

    // ========== Calc the nbh ders wrt moments ==========
    {
      int team_size = team_size_default;
      // Best team size depends on the max number of blocks per SM. 64 is good for CC8, and CC > 9+.
      Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeNbhDers> policy_basic_alpha(chunk_size,
                                                                                  team_size);
      Kokkos::parallel_for("ComputeNbhDers", policy_basic_alpha, *this);
    }

    // ========== Reduce Basis Ders (Configuration mode) / Calculate Extrapolation (Neighbourhood Mode) ==========
    if (calculate_grade_this_step) {
      if (configuration_mode) {    // Configuration mode,
        //Here is quick heurustuc tuned to work okay for most problem sizes and coeff counts.
        int team_size = 1024;
        int sizes[5] = {512, 256, 128, 64, 32};
        for (int i = 0; i < 5; i++) {
          if (chunk_size >= sizes[i]) break;
          team_size = sizes[i];
        }

        // Perform the reduction across the current chunk_size
        int vector_length = vector_length_default;
        check_team_size_for<TagPairMTPReduceCoeffDers>(coeff_count, team_size, vector_length);
        int scratch_size = scratch_size_helper<F_FLOAT>(0);
        Kokkos::TeamPolicy<DeviceType, TagPairMTPReduceCoeffDers> policy_reduce_ders(coeff_count,
                                                                                     team_size);
        policy_reduce_ders = policy_reduce_ders.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
        Kokkos::parallel_for("ReduceBasisDers", policy_reduce_ders, *this);

      } else {                          // Neighbourhood mode
        F_FLOAT chunk_max_grade = 0;    // Reduce into a tmp variable

        // Rough heuristic for team size
        int team_size = 256;
        int sizes[3] = {128, 64, 32};
        for (int i = 0; i < 3; i++) {
          if (coeff_count >= sizes[i]) break;
          team_size = sizes[i];
        }

        int scratch_size = scratch_size_helper<F_FLOAT>(coeff_count);
        Kokkos::TeamPolicy<DeviceType> policy_calc_grades(chunk_size, team_size);
        policy_calc_grades.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

        Kokkos::parallel_reduce(
            "sComputeNbhGrades", policy_calc_grades,
            sComputeNbhGrades<DeviceType>(
                chunk_size, chunk_offset, d_ilist, type, species_count, radial_coeff_count,
                alpha_index_basic_count, radial_coeff_count_per_pair, alpha_scalar_count,
                coeff_count, d_nbh_energy_ders_wrt_moments, d_radial_jacobian, d_moment_tensor_vals,
                d_alpha_moment_mapping, d_inverse_active_set, d_nbh_extrapolation_grades),
            Kokkos::Max<F_FLOAT>(chunk_max_grade));

        max_grade = Kokkos::max(chunk_max_grade, max_grade);    // Get max over all chunks
      }
    }

    // ========== Compute force (and dot product with alphas to get energy if needed) ==========
    {
      int team_size = team_size_default;
      if (!host_flag && max_neighs < 32) team_size = 32;
      if (neighflag == HALF) {
        Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeForce<HALF, 1>> policy_force(chunk_size,
                                                                                     team_size);
        Kokkos::parallel_reduce(policy_force, *this, ev_tmp);
      } else if (neighflag == HALFTHREAD) {
        Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeForce<HALFTHREAD, 1>> policy_force(
            chunk_size, team_size);
        Kokkos::parallel_reduce(policy_force, *this, ev_tmp);
      }
    }

    ev += ev_tmp;
    chunk_offset += chunk_size;    // Manage halt condition
  }    // end batching while loop

  // ========== End Main Computation ==========

  if (need_dup) Kokkos::Experimental::contribute(f, dup_f);

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    if (need_dup) Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  atomKK->modified(execution_space, F_MASK);

  copymode = 0;

  // free duplicated memory
  if (need_dup) {
    dup_f = decltype(dup_f)();
    dup_vatom = decltype(dup_vatom)();
  }

  if (!calculate_grade_this_step) return;    // Done for non-extrapolation step

  // Now, we need to handle the extrapolation obtained collectivelly across chunks.
  // This will also depend on if we are split across MPI processes.
  if (configuration_mode) {     // Configuration mode
    if (comm->nprocs == 1) {    // Single Process
      // If we are sure we are running on 1 process, we can directly evaluate the cfg grade on device
      // Perform the reduction across the current chunk_size. Simple heuristic for team size.
      int team_size = 512;
      int sizes[5] = {256, 128, 64};
      for (int i = 0; i < 5; i++) {
        if (coeff_count >= sizes[i]) break;
        team_size = sizes[i];
      }

      int scratch_size = scratch_size_helper<F_FLOAT>(0);
      Kokkos::TeamPolicy<DeviceType> policy_calc_grades(coeff_count, team_size);
      policy_calc_grades.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

      Kokkos::parallel_reduce(
          "sComputeCfgGrade", policy_calc_grades,
          sComputeCfgGrade<DeviceType>(coeff_count, d_energy_ders_wrt_coeffs, d_inverse_active_set),
          Kokkos::Max<F_FLOAT>(max_grade));

      if (atom->natoms > 0)
        max_grade /= atom->natoms;    // Normalize
      else
        max_grade = 0.0;
      pvector[0] = max_grade;

    } else {    // Multiple Processes

      // On multiple procs we need to move ders to host and MPI reduce across ranks
      Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          h_energy_ders_wrt_coeffs(energy_ders_wrt_coeffs, coeff_count);
      Kokkos::deep_copy(h_energy_ders_wrt_coeffs, d_energy_ders_wrt_coeffs);

      PairMTPExtrapolation::compile_grades();
    }    // Normalize by atom count in CFG mode
  } else {    // Neighbourhood mode
    PairMTPExtrapolation::compile_grades();
  }

  evaluate_grades();    // Evaluate and write based on max grade
}

// ========== Kernels ==========

// Inits the working arrays: moments and ders, moment jacobian not needed.
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void
PairMTPsExtrapolationKokkos<DeviceType>::operator()(TagPairMTPInitMomentValsDers, const int &k,
                                                    const int &ii) const
{
  d_moment_tensor_vals(ii, k) = 0;
  d_nbh_energy_ders_wrt_moments(ii, k) = 0;
}

// Inits the radial jacobian (only called on steps with extrapolation) anddo the above
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void
PairMTPsExtrapolationKokkos<DeviceType>::operator()(TagPairMTPInitRadJacobian, const int &ii,
                                                    const int &k) const
{
  for (int ri = 0; ri < species_count * radial_coeff_count_per_pair; ri++)
    d_radial_jacobian(ii, k, ri) = 0;
}

// Inits the coeff ders (only called on steps with extrapolation and cfg mode)
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void
PairMTPsExtrapolationKokkos<DeviceType>::operator()(TagPairMTPInitCoeffDers, const int &kk) const
{
  d_energy_ders_wrt_coeffs(kk) = 0.0;
}

// Calculates the basic alphas using fused operations where possible
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMTPsExtrapolationKokkos<DeviceType>::operator()(
    TagPairMTPComputeAlphaBasic,
    const typename Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeAlphaBasic>::member_type &team)
    const
{
  // Extract the atom number
  int ii = team.league_rank();
  int thread = team.team_rank();

  // Get information about the central atom
  const int i = d_ilist[ii + chunk_offset];
  const F_FLOAT xi[3] = {x(i, 0), x(i, 1), x(i, 2)};
  const int itype = type[i] - 1;    // switch to zero indexing
  const int jnum = d_numneigh(i);
  const int array_size = Kokkos::min(team.team_size(), jnum);

  shared_double_2d s_radial_vals(team.team_scratch(0), array_size, radial_func_count);
  shared_double_2d s_radial_ders(team.team_scratch(0), array_size, radial_func_count);
  shared_double_2d s_dist_powers(team.team_scratch(0), array_size, max_alpha_index_basic);
  shared_double_3d s_coord_powers(team.team_scratch(0), array_size, max_alpha_index_basic);
  shared_double_2d s_radial_basis_vals(team.team_scratch(0), array_size, radial_basis_size);
  shared_double_2d s_radial_basis_ders(team.team_scratch(0), array_size, radial_basis_size);

  // Now we calculate the alpha basics. There might be benefits to using a parallel reduce into the array of moment values here.
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, jnum), [=](const int jj) {
    const int j = d_neighbors(i, jj) & NEIGHMASK;
    const int jtype = type[j] - 1;    // switch to zero indexing
    const F_FLOAT r[3] = {x(j, 0) - xi[0], x(j, 1) - xi[1], x(j, 2) - xi[2]};
    const F_FLOAT rsq = Kokkos::fma(r[0], r[0], Kokkos::fma(r[1], r[1], r[2] * r[2]));

    const bool valid_pair = rsq < max_cutoff_sq;
    d_within_cutoff(jj, ii) = valid_pair;

    if (!valid_pair) return;
    const F_FLOAT dist = sqrt(rsq);

    s_dist_powers(thread, 0) = s_coord_powers(thread, 0, 0) = s_coord_powers(thread, 0, 1) =
        s_coord_powers(thread, 0, 2) = 1;    // Set the constants

    // Precompute the coord and distance power
    for (int k = 1; k < max_alpha_index_basic; k++) {
      s_dist_powers(thread, k) = s_dist_powers(thread, k - 1) * dist;
      for (int a = 0; a < 3; a++)
        s_coord_powers(thread, k, a) = s_coord_powers(thread, k - 1, a) * r[a];
    }

    // Calculate the radial basis and store in shared memory
    F_FLOAT mult = 2.0 / (max_cutoff - min_cutoff);
    F_FLOAT ksi = Kokkos::fma(2.0, dist, -(min_cutoff + max_cutoff)) / (max_cutoff - min_cutoff);

    F_FLOAT temp = dist - max_cutoff;
    s_radial_basis_vals(thread, 0) = scaling * temp * temp;
    s_radial_basis_vals(thread, 1) = scaling * ksi * temp * temp;
    for (int k = 2; k < radial_basis_size; k++) {
      s_radial_basis_vals(thread, k) = Kokkos::fma(2.0 * ksi, s_radial_basis_vals(thread, k - 1),
                                                   -s_radial_basis_vals(thread, k - 2));
    }

    // Do the same with the derivatives
    s_radial_basis_ders(thread, 0) = scaling * 2.0 * temp;
    s_radial_basis_ders(thread, 1) = scaling * Kokkos::fma(mult, temp * temp, 2.0 * ksi * temp);
    for (int k = 2; k < radial_basis_size; k++) {
      F_FLOAT tmp = Kokkos::fma(mult, s_radial_basis_vals(thread, k - 1),
                                ksi * s_radial_basis_ders(thread, k - 1));
      s_radial_basis_ders(thread, k) = Kokkos::fma(2.0, tmp, -s_radial_basis_ders(thread, k - 2));
    }

    // Precompute the mu vals and ders
    int pair_offset = itype * species_count + jtype;
    for (int mu = 0; mu < radial_func_count; mu++) {
      F_FLOAT val = 0;
      F_FLOAT der = 0;
      int offset = (pair_offset * radial_basis_size * radial_func_count) + mu * radial_basis_size;

      for (int ri = 0; ri < radial_basis_size; ri++) {
        val = Kokkos::fma(d_radial_basis_coeffs(offset + ri), s_radial_basis_vals(thread, ri), val);
        der = Kokkos::fma(d_radial_basis_coeffs(offset + ri), s_radial_basis_ders(thread, ri), der);
      }

      s_radial_vals(thread, mu) = val;
      s_radial_ders(thread, mu) = der;
    }

    //Now, we loop through all the basic alphas
    for (int k = 0; k < alpha_index_basic_count; k++) {

      int mu = d_alpha_index_basic(k, 0);
      int a0 = d_alpha_index_basic(k, 1);
      int a1 = d_alpha_index_basic(k, 2);
      int a2 = d_alpha_index_basic(k, 3);

      F_FLOAT val = s_radial_vals(thread, mu);
      F_FLOAT der = s_radial_ders(thread, mu);

      // Normalize by the rank of alpha's coresponding tensor
      int norm_rank = a0 + a1 + a2;
      F_FLOAT norm_fac = 1.0 / s_dist_powers(thread, norm_rank);
      val *= norm_fac;
      der = Kokkos::fma(norm_fac, der, -norm_rank * val / dist);

      F_FLOAT pow0 = s_coord_powers(thread, a0, 0);
      F_FLOAT pow1 = s_coord_powers(thread, a1, 1);
      F_FLOAT pow2 = s_coord_powers(thread, a2, 2);
      F_FLOAT pow = pow0 * pow1 * pow2;
      Kokkos::atomic_add(&d_moment_tensor_vals(ii, k), val * pow);

      // Get the component's derivatives too
      F_FLOAT temp_jac[3] = {pow * r[0], pow * r[1], pow * r[2]};

      pow *= der / dist;
      temp_jac[0] = pow * r[0];
      temp_jac[1] = pow * r[1];
      temp_jac[2] = pow * r[2];

      if (a0 != 0)
        temp_jac[0] =
            Kokkos::fma(val * a0, s_coord_powers(thread, a0 - 1, 0) * pow1 * pow2, temp_jac[0]);
      if (a1 != 0)
        temp_jac[1] =
            Kokkos::fma(val * a1, pow0 * s_coord_powers(thread, a1 - 1, 1) * pow2, temp_jac[1]);
      if (a2 != 0)
        temp_jac[2] =
            Kokkos::fma(val * a2, pow0 * pow1 * s_coord_powers(thread, a2 - 1, 2), temp_jac[2]);

      d_moment_jacobian(jj, ii, k, 0) = temp_jac[0];
      d_moment_jacobian(jj, ii, k, 1) = temp_jac[1];
      d_moment_jacobian(jj, ii, k, 2) = temp_jac[2];
    }
  });
}

// Calculates the basic alphas with radial jacobian
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMTPsExtrapolationKokkos<DeviceType>::operator()(
    TagPairMTPComputeAlphaBasicRad,
    const typename Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeAlphaBasicRad>::member_type
        &team) const
{
  // Extract the atom number
  int ii = team.league_rank();
  int thread = team.team_rank();

  // Get information about the central atom
  const int i = d_ilist[ii + chunk_offset];
  const F_FLOAT xi[3] = {x(i, 0), x(i, 1), x(i, 2)};
  const int itype = type[i] - 1;    // switch to zero indexing
  const int jnum = d_numneigh(i);
  const int array_size = Kokkos::min(team.team_size(), jnum);

  shared_double_2d s_radial_vals(team.team_scratch(0), array_size, radial_func_count);
  shared_double_2d s_radial_ders(team.team_scratch(0), array_size, radial_func_count);
  shared_double_2d s_dist_powers(team.team_scratch(0), array_size, max_alpha_index_basic);
  shared_double_3d s_coord_powers(team.team_scratch(0), array_size, max_alpha_index_basic);
  shared_double_2d s_radial_basis_vals(team.team_scratch(0), array_size, radial_basis_size);
  shared_double_2d s_radial_basis_ders(team.team_scratch(0), array_size, radial_basis_size);

  // Now we calculate the alpha basics. There might be benefits to using a parallel reduce into the array of moment values here.
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, jnum), [=](const int jj) {
    const int j = d_neighbors(i, jj) & NEIGHMASK;
    const int jtype = type[j] - 1;    // switch to zero indexing
    const F_FLOAT r[3] = {x(j, 0) - xi[0], x(j, 1) - xi[1], x(j, 2) - xi[2]};
    const F_FLOAT rsq = Kokkos::fma(r[0], r[0], Kokkos::fma(r[1], r[1], r[2] * r[2]));

    const bool valid_pair = rsq < max_cutoff_sq;
    d_within_cutoff(jj, ii) = valid_pair;

    if (!valid_pair) return;
    const F_FLOAT dist = sqrt(rsq);

    s_dist_powers(thread, 0) = s_coord_powers(thread, 0, 0) = s_coord_powers(thread, 0, 1) =
        s_coord_powers(thread, 0, 2) = 1;    // Set the constants

    // Precompute the coord and distance power
    for (int k = 1; k < max_alpha_index_basic; k++) {
      s_dist_powers(thread, k) = s_dist_powers(thread, k - 1) * dist;
      for (int a = 0; a < 3; a++)
        s_coord_powers(thread, k, a) = s_coord_powers(thread, k - 1, a) * r[a];
    }

    // Calculate the radial basis and store in shared memory
    F_FLOAT mult = 2.0 / (max_cutoff - min_cutoff);
    F_FLOAT ksi = Kokkos::fma(2.0, dist, -(min_cutoff + max_cutoff)) / (max_cutoff - min_cutoff);

    F_FLOAT temp = dist - max_cutoff;
    s_radial_basis_vals(thread, 0) = scaling * temp * temp;
    s_radial_basis_vals(thread, 1) = scaling * ksi * temp * temp;
    for (int k = 2; k < radial_basis_size; k++) {
      s_radial_basis_vals(thread, k) = Kokkos::fma(2.0 * ksi, s_radial_basis_vals(thread, k - 1),
                                                   -s_radial_basis_vals(thread, k - 2));
    }

    // Do the same with the derivatives
    s_radial_basis_ders(thread, 0) = scaling * 2.0 * temp;
    s_radial_basis_ders(thread, 1) = scaling * Kokkos::fma(mult, temp * temp, 2.0 * ksi * temp);
    for (int k = 2; k < radial_basis_size; k++) {
      F_FLOAT tmp = Kokkos::fma(mult, s_radial_basis_vals(thread, k - 1),
                                ksi * s_radial_basis_ders(thread, k - 1));
      s_radial_basis_ders(thread, k) = Kokkos::fma(2.0, tmp, -s_radial_basis_ders(thread, k - 2));
    }

    // Precompute the mu vals and ders
    int pair_offset = itype * species_count + jtype;
    for (int mu = 0; mu < radial_func_count; mu++) {
      F_FLOAT val = 0;
      F_FLOAT der = 0;
      int offset = (pair_offset * radial_basis_size * radial_func_count) + mu * radial_basis_size;

      for (int ri = 0; ri < radial_basis_size; ri++) {
        val = Kokkos::fma(d_radial_basis_coeffs(offset + ri), s_radial_basis_vals(thread, ri), val);
        der = Kokkos::fma(d_radial_basis_coeffs(offset + ri), s_radial_basis_ders(thread, ri), der);
      }

      s_radial_vals(thread, mu) = val;
      s_radial_ders(thread, mu) = der;
    }

    //Now, we loop through all the basic alphas
    for (int k = 0; k < alpha_index_basic_count; k++) {
      int mu = d_alpha_index_basic(k, 0);
      int a0 = d_alpha_index_basic(k, 1);
      int a1 = d_alpha_index_basic(k, 2);
      int a2 = d_alpha_index_basic(k, 3);

      F_FLOAT val = s_radial_vals(thread, mu);
      F_FLOAT der = s_radial_ders(thread, mu);

      // Normalize by the rank of alpha's coresponding tensor
      int norm_rank = a0 + a1 + a2;
      F_FLOAT norm_fac = 1.0 / s_dist_powers(thread, norm_rank);
      val *= norm_fac;
      der = Kokkos::fma(norm_fac, der, -norm_rank * val / dist);

      F_FLOAT pow0 = s_coord_powers(thread, a0, 0);
      F_FLOAT pow1 = s_coord_powers(thread, a1, 1);
      F_FLOAT pow2 = s_coord_powers(thread, a2, 2);
      F_FLOAT pow = pow0 * pow1 * pow2;
      Kokkos::atomic_add(&d_moment_tensor_vals(ii, k), val * pow);

      // Update radial jacobian
      int rad_offset = Kokkos::fma(jtype, radial_coeff_count_per_pair, mu * radial_basis_size);
      for (int ri = 0; ri < radial_basis_size; ri++) {
        Kokkos::atomic_add(&d_radial_jacobian(ii, k, rad_offset + ri),
                           norm_fac * pow * s_radial_basis_vals(thread, ri));
      }

      // Get the component's derivatives too
      F_FLOAT temp_jac[3] = {pow * r[0], pow * r[1], pow * r[2]};

      pow *= der / dist;
      temp_jac[0] = pow * r[0];
      temp_jac[1] = pow * r[1];
      temp_jac[2] = pow * r[2];

      if (a0 != 0)
        temp_jac[0] =
            Kokkos::fma(val * a0, s_coord_powers(thread, a0 - 1, 0) * pow1 * pow2, temp_jac[0]);
      if (a1 != 0)
        temp_jac[1] =
            Kokkos::fma(val * a1, pow0 * s_coord_powers(thread, a1 - 1, 1) * pow2, temp_jac[1]);
      if (a2 != 0)
        temp_jac[2] =
            Kokkos::fma(val * a2, pow0 * pow1 * s_coord_powers(thread, a2 - 1, 2), temp_jac[2]);

      d_moment_jacobian(jj, ii, k, 0) = temp_jac[0];
      d_moment_jacobian(jj, ii, k, 1) = temp_jac[1];
      d_moment_jacobian(jj, ii, k, 2) = temp_jac[2];
    }
  });
}

// Calculates the non-elementary alpha from the basic alphas
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMTPsExtrapolationKokkos<DeviceType>::operator()(
    TagPairMTPComputeAlphaTimes,
    const typename Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeAlphaTimes>::member_type &team)
    const
{

  int ii = team.league_rank();

  int offset = 0;
  // Traverse all edges in the alpha times compute graph. We need to do this in waves to ensure dependencies.
  for (int i = 0; i < 3; i++) {
    int wave_size = wave_sizes[i];
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, wave_size), [=](const int kk) {
      int k = offset + kk;    // Offset for the wave
      int a0 = d_alpha_index_times(k, 0);
      int a1 = d_alpha_index_times(k, 1);
      int mult = d_alpha_index_times(k, 2);
      int a3 = d_alpha_index_times(k, 3);

      F_FLOAT val0 = d_moment_tensor_vals(ii, a0);
      F_FLOAT val1 = d_moment_tensor_vals(ii, a1);

      Kokkos::atomic_add(&d_moment_tensor_vals(ii, a3), mult * val0 * val1);
    });
    offset += wave_size;
    team.team_barrier();    // Wait for the wave to finish
  }
}

// Sets the nbh energy ders as the linear coeffs
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void
PairMTPsExtrapolationKokkos<DeviceType>::operator()(TagPairMTPSetScalarNbhDers, const int &k,
                                                    const int &ii) const
{
  d_nbh_energy_ders_wrt_moments(ii, d_alpha_moment_mapping(k)) = d_linear_coeffs(k);
}

// Calculates the nbh ders
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMTPsExtrapolationKokkos<DeviceType>::operator()(
    TagPairMTPComputeNbhDers,
    const typename Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeNbhDers>::member_type &team)
    const
{

  int ii = team.league_rank();

  int offset = alpha_index_times_count;
  // Traverse all edges in the alpha times compute graph. We need to do this in reverse waves to ensure dependencies.
  for (int i = 2; i >= 0; i--) {
    int wave_size = wave_sizes[i];
    offset -= wave_size;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, wave_size), [=](const int kk) {
      int k = kk + offset;    // Offset for the wave
      int a0 = d_alpha_index_times(k, 0);
      int a1 = d_alpha_index_times(k, 1);
      int mult = d_alpha_index_times(k, 2);
      int a3 = d_alpha_index_times(k, 3);

      F_FLOAT val0 = d_moment_tensor_vals(ii, a0);
      F_FLOAT val1 = d_moment_tensor_vals(ii, a1);
      F_FLOAT val3 = d_nbh_energy_ders_wrt_moments(ii, a3);

      Kokkos::atomic_add(&d_nbh_energy_ders_wrt_moments(ii, a1), val3 * mult * val0);
      Kokkos::atomic_add(&d_nbh_energy_ders_wrt_moments(ii, a0), val3 * mult * val1);
    });
    team.team_barrier();    // Wait for the wave to finish
  }
}

// Computes forces from jac and nbh ders
template <class DeviceType>
template <int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION void PairMTPsExtrapolationKokkos<DeviceType>::operator()(
    const TagPairMTPComputeForce<NEIGHFLAG, EVFLAG> &,    // Tag parameter added here
    const typename Kokkos::TeamPolicy<DeviceType,
                                      TagPairMTPComputeForce<NEIGHFLAG, EVFLAG>>::member_type &team,
    EV_FLOAT &ev) const
{
  // The f array is duplicated for OpenMP, atomic for GPU, and neither for Serial
  auto v_f =
      ScatterViewHelper<NeedDup_v<NEIGHFLAG, DeviceType>, decltype(dup_f), decltype(ndup_f)>::get(
          dup_f, ndup_f);
  auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG, DeviceType>>();

  const int ii = team.league_rank();
  const int i = d_ilist[ii + chunk_offset];
  const int jnum = d_numneigh(i);
  bool need_energies = EVFLAG && eflag_either;

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, jnum), [&](const int jj) {
    const int j = d_neighbors(i, jj) & NEIGHMASK;

    if (!d_within_cutoff(jj, ii)) return;

    F_FLOAT temp_force[3] = {0, 0, 0};
    for (int k = 0; k < alpha_index_basic_count; k++) {
      for (int a = 0; a < 3; a++) {
        //Calculate forces
        temp_force[a] += d_nbh_energy_ders_wrt_moments(ii, k) * d_moment_jacobian(jj, ii, k, a);
      }
    }

    // This could feasibly be done with a reduction instead, but is a marginal speedup if any
    a_f(i, 0) += temp_force[0];
    a_f(i, 1) += temp_force[1];
    a_f(i, 2) += temp_force[2];

    a_f(j, 0) -= temp_force[0];
    a_f(j, 1) -= temp_force[1];
    a_f(j, 2) -= temp_force[2];

    if (need_energies) {
      F_FLOAT r[3] = {x(j, 0) - x(j, 0), x(j, 1) - x(j, 1), x(j, 2) - x(j, 2)};
      v_tally_xyz<NEIGHFLAG>(ev, i, j, temp_force[0], temp_force[1], temp_force[2], r[0], r[1],
                             r[2]);
    }
  });

  if (need_energies) {
    const int itype = type(i) - 1;    // zero indexing
    F_FLOAT nbh_energy = 0;

    // Reduction to find the dot product of the linear coeffs and the moment tensor vals
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, alpha_scalar_count),
        [&](const int k, F_FLOAT &sum) {
          sum += d_linear_coeffs(k) * d_moment_tensor_vals(ii, d_alpha_moment_mapping(k));
        },
        nbh_energy);

    // A single team member updates the global array
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      nbh_energy += d_species_coeffs[itype];    // Essentially the reference energy
      if (eflag_global) ev.evdwl += nbh_energy;
      if (eflag_atom) d_eatom[i] = nbh_energy;
    });
  }
}

// Accumulates the coeffs der wrt cfg energy
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMTPsExtrapolationKokkos<DeviceType>::operator()(
    TagPairMTPReduceCoeffDers,
    const typename Kokkos::TeamPolicy<DeviceType, TagPairMTPReduceCoeffDers>::member_type &team)
    const
{
  /*We need to perform a reduction across all atoms for all energy ders.
There are three types of ders:
1. Radial ders
2. Species Ders
3. Moment Ders
Radials are much much more expesive than the others but there is no guarentee that there are enough
radial ders to saturate the SMs, espeically if only have 1 species. Thus, we will also issue the other reductions
in the same kernel call. We will target 1 thread block per SM, so 1024 threads per block.
It is probably  preferable to use different streams.
*/
  const int kk = team.league_rank();
  F_FLOAT reduction_result = 0;

  if (kk < radial_coeff_count) {
    //Case 1: Radial coefficients
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, chunk_size),
        [&](const int ii, F_FLOAT &sum) {
          F_FLOAT partial_sum = 0;
          const int i = d_ilist[ii + chunk_offset];
          const int itype = type[i] - 1;    // switch to zero indexing

          // We only take the deriatives based on the type of the central.
          // Since the radial array is flattened with the itype first, integer divide
          // by the width of itype * the coeffs per pair to check

          if (kk / (species_count * radial_coeff_count_per_pair) == itype) {
            const int local_index = kk % (species_count * radial_coeff_count_per_pair);
            for (int k = 0; k < alpha_index_basic_count; k++) {
              partial_sum +=
                  d_nbh_energy_ders_wrt_moments(ii, k) * d_radial_jacobian(ii, k, local_index);
            }
          }
          sum += partial_sum;
        },
        reduction_result);
  } else if (kk < radial_coeff_count + species_count) {
    //Case 2: Species coefficient
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, chunk_size),
        [&](const int ii, F_FLOAT &sum) {
          const int i = d_ilist[ii + chunk_offset];
          const int itype = type[i] - 1;    // switch to zero indexing
          F_FLOAT val = 0.0;
          if (itype == kk - radial_coeff_count) val = 1.0;
          sum += val;
        },
        reduction_result);

  } else {
    //Case 3: Basis set
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, chunk_size),
        [&](const int ii, F_FLOAT &sum) {
          sum += d_moment_tensor_vals(
              ii, d_alpha_moment_mapping(kk - species_count - radial_coeff_count));
        },
        reduction_result);
  }
  Kokkos::single(Kokkos::PerTeam(team), [&]() {
    d_energy_ders_wrt_coeffs(kk) += reduction_result;
  });
}

// Compute the extrapolation grade for all nbhs and reduce the maximum value
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void sComputeNbhGrades<DeviceType>::operator()(
    const typename Kokkos::TeamPolicy<DeviceType>::member_type &team, F_FLOAT &nbh_max_grade) const
{
  // Extract the atom number
  int ii = team.league_rank();
  if (ii >= chunk_size) return;

  const int i = d_ilist(ii + chunk_offset);
  const int itype = type(i) - 1;    // switch to zero indexing

  // Shared memory to store the candidate vector
  shared_double_1d s_candidate_vector(team.team_scratch(0), coeff_count);

  // Initialize the radial and species coeff ders
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, radial_coeff_count + species_count),
                       [&](const int k) {
                         s_candidate_vector(k) = 0.0;
                       });
  team.team_barrier();    // Barrier to ensure all vals are inited

  // First calculate the radial ders and store into shared memory
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, alpha_index_basic_count), [&](const int k) {
    int offset = itype * species_count * radial_coeff_count_per_pair;
    for (int rii = k; rii < (radial_coeff_count_per_pair * species_count) + k; rii++) {
      int ri = rii % (radial_coeff_count_per_pair * species_count);
      Kokkos::atomic_add(&s_candidate_vector(offset + ri),
                         d_nbh_energy_ders_wrt_moments(ii, k) * d_radial_jacobian(ii, k, ri));
    }
  });

  // Load the basis vals into shared memory
  int moment_offset = radial_coeff_count + species_count;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, alpha_scalar_count), [&](const int k) {
    s_candidate_vector(moment_offset + k) = d_moment_tensor_vals(ii, d_alpha_moment_mapping(k));
  });

  // Store the species der
  Kokkos::single(Kokkos::PerTeam(team), [&]() {
    s_candidate_vector(radial_coeff_count + itype) = 1;
  });

  team.team_barrier();    // Barrier to ensure all data is loaded

  // Now we can calculate the extrapolation grade with a parallel reduction
  F_FLOAT nbh_grade = 0;

  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team, coeff_count),
      [&](const int i, F_FLOAT &grade) {
        F_FLOAT current_grade = 0;
        for (int j = 0; j < coeff_count; j++) {
          current_grade += s_candidate_vector(j) * d_inverse_active_set(i, j);
        }
        current_grade = Kokkos::abs(current_grade);
        grade = (grade > current_grade) ? grade : current_grade;
      },
      Kokkos::Max<F_FLOAT, DeviceType>(nbh_grade));

  Kokkos::single(Kokkos::PerTeam(team), [&]() {
    d_nbh_extrapolation_grades(i) = nbh_grade;
    nbh_max_grade = Kokkos::max(nbh_grade, nbh_max_grade);
  });
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void sComputeCfgGrade<DeviceType>::operator()(
    const typename Kokkos::TeamPolicy<DeviceType>::member_type &team, F_FLOAT &cfg_max_grade) const
{
  // Extract row number
  int ik = team.league_rank();

  // Now we can calculate the swap grade of this row with a parallel reduction
  F_FLOAT candidate_grade = 0;
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team, coeff_count),
      [&](const int jk, F_FLOAT &grade) {
        grade = Kokkos::fma(d_energy_ders_wrt_coeffs(jk), d_inverse_active_set(ik, jk), grade);
      },
      candidate_grade);
  candidate_grade = Kokkos::abs(candidate_grade);

  Kokkos::single(Kokkos::PerTeam(team), [&]() {
    cfg_max_grade = Kokkos::max(candidate_grade, cfg_max_grade);
  });
}

// =========== Helper Functions (Also used in other Kokkos potentials)===========
template <class DeviceType>
template <int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION void PairMTPsExtrapolationKokkos<DeviceType>::v_tally_xyz(
    EV_FLOAT &ev, const int &i, const int &j, const F_FLOAT &fx, const F_FLOAT &fy,
    const F_FLOAT &fz, const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const
{
  // The vatom array is duplicated for OpenMP, atomic for GPU, and neither for Serial

  auto v_vatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG, DeviceType>, decltype(dup_vatom),
                                   decltype(ndup_vatom)>::get(dup_vatom, ndup_vatom);
  auto a_vatom = v_vatom.template access<AtomicDup_v<NEIGHFLAG, DeviceType>>();

  const E_FLOAT v0 = delx * fx;
  const E_FLOAT v1 = dely * fy;
  const E_FLOAT v2 = delz * fz;
  const E_FLOAT v3 = delx * fy;
  const E_FLOAT v4 = delx * fz;
  const E_FLOAT v5 = dely * fz;

  if (vflag_global) {
    ev.v[0] += v0;
    ev.v[1] += v1;
    ev.v[2] += v2;
    ev.v[3] += v3;
    ev.v[4] += v4;
    ev.v[5] += v5;
  }

  if (vflag_atom) {
    a_vatom(i, 0) += 0.5 * v0;
    a_vatom(i, 1) += 0.5 * v1;
    a_vatom(i, 2) += 0.5 * v2;
    a_vatom(i, 3) += 0.5 * v3;
    a_vatom(i, 4) += 0.5 * v4;
    a_vatom(i, 5) += 0.5 * v5;
    a_vatom(j, 0) += 0.5 * v0;
    a_vatom(j, 1) += 0.5 * v1;
    a_vatom(j, 2) += 0.5 * v2;
    a_vatom(j, 3) += 0.5 * v3;
    a_vatom(j, 4) += 0.5 * v4;
    a_vatom(j, 5) += 0.5 * v5;
  }
}

template <class DeviceType>
template <class TagStyle>
void PairMTPsExtrapolationKokkos<DeviceType>::check_team_size_for(int inum, int &team_size,
                                                                  int vector_length)
{
  int team_size_max;

  team_size_max = Kokkos::TeamPolicy<DeviceType, TagStyle>(inum, Kokkos::AUTO)
                      .team_size_max(*this, Kokkos::ParallelForTag());

  if (team_size * vector_length > team_size_max) team_size = team_size_max / vector_length;
}

template <class DeviceType>
template <typename scratch_type>
int PairMTPsExtrapolationKokkos<DeviceType>::scratch_size_helper(int values_per_team)
{
  typedef Kokkos::View<scratch_type *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      ScratchViewType;

  return ScratchViewType::shmem_size(values_per_team);
}    // namespace LAMMPS_NS

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairMTPsExtrapolationKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairMTPsExtrapolationKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS
