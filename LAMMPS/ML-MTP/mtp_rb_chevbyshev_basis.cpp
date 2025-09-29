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
// Contributing author, Richard Meng, Queen's University at Kingston, 22.11.24, contact@richardzjm.com
//

#include "mtp_radial_basis.h"
#include "mtp_rb_chebyshev_basis.h"

#include "error.h"
#include "memory.h"
#include "utils.h"

#include "cstring"

using namespace LAMMPS_NS;

void RBChebyshev::calc_radial_basis(double dist)
{
  double ksi = (2 * dist - (min_cutoff + max_cutoff)) / (max_cutoff - min_cutoff);

  radial_basis_vals[0] = scaling * (1 * (dist - max_cutoff) * (dist - max_cutoff));
  radial_basis_vals[1] = scaling * (ksi * (dist - max_cutoff) * (dist - max_cutoff));
  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = 2 * ksi * radial_basis_vals[i - 1] - radial_basis_vals[i - 2];
  }
}

void RBChebyshev::calc_radial_basis_ders(double dist)
{
  RBChebyshev::calc_radial_basis(dist);

  double mult = 2.0 / (max_cutoff - min_cutoff);
  double ksi = (2 * dist - (min_cutoff + max_cutoff)) / (max_cutoff - min_cutoff);

  radial_basis_ders[0] = scaling * 2 * (dist - max_cutoff);
  radial_basis_ders[1] =
      scaling * (mult * (dist - max_cutoff) * (dist - max_cutoff) + 2 * ksi * (dist - max_cutoff));
  for (int i = 2; i < size; i++) {
    radial_basis_ders[i] = 2 * (mult * radial_basis_vals[i - 1] + ksi * radial_basis_ders[i - 1]) -
        radial_basis_ders[i - 2];
  }
}