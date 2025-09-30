# lammps-mtp-kokkos

This repository is a Kokkos implementation of the Moment Tensor Potential in LAMMPS for GPU acceleration, using MLIP-3-compatible potential and configuration formats. This also includes a CPU version with additional performance optimizations over MLIP-3 with up to a $2\,\times$ speedup in real production workloads.

## Installation

For this repository, we detail a very simple installation method. In the future, we may seek to push this implementation to the LAMMPS repository.

1. Clone LAMMPS. This **requires version 24 March 2022 or newer.**

```
git clone -b stable https://github.com/lammps/lammps.git
```

2. Add source files.

- For a CPU-only install, simply copy the _contents_ of the folder `LAMMPS/ML-MTP` directly into `lammps/src`. Then, build and compile as usual.
- For a GPU install with Kokkos, copy both the _contents_ of the folder `LAMMPS/ML-MTP` and the _contents_ of the folder `LAMMPS/KOKKOS` directly into `lammps/src`. Then, build and compile as usual, following the Kokkos installation instructions found [here](https://docs.lammps.org/Speed_kokkos.html) and [here](https://docs.lammps.org/Build_extras.html#kokkos).

3.  After you have compiled the executable, you can check whether the potentials have been installed with `/path/to/lammps/executable -h`. In the output, under `* Pair styles:`, you should find the following among other pair styles. You will only see the styles ending in `/kk` if you compiled with Kokkos support.

```
mtp mtp/extrapolation mtp/extrapolation/kk mtp/kk mtp/extrapolation/small/kk mtp/small/kk
```

## Invocation

There are several variants provided with this repository. Specifically, we support active learning in both configuration and neighborhood modes on the CPU and on the GPU with two variants. These variants are a thread-parallel approach, which maps each atom to a GPU thread, requiring $\gtrsim$ 50,000 atoms for peak throughput on an A100, and a block-parallel approach, which maps each atom to a block of GPU threads, requiring $\gtrsim$ 2,000 atoms for peak throughput on an A100. The block-parallel method may attain a lower peak throughput due to additional parallelization overheads. The choice of GPU variant depends on the problem size, MTP characteristics, and the GPU hardware and is best determined through testing.

When active learning is enabled, the mode is automatically read from the potential file. The mode of a potential file is determined when the potential is trained.

The MTP is invoked in a LAMMPS script using the `pair_style` command. The `pair_coeff` command is not required for these `pair_style`s. Below are the identifiers for each of the variants.

<div align="center">

| Platform | Variant         | Inference      | Active Learning / Both Modes |
| -------- | --------------- | -------------- | ---------------------------- |
| CPU      | —               | `mtp`          | `mtp/extrapolation`          |
| GPU      | Thread-Parallel | `mtp/kk`       | `mtp/extrapolation/kk`       |
| GPU      | Block-Parallel  | `mtp/small/kk` | `mtp/extrapolation/small/kk` |

</div>

The identifier is then followed by the path to the MTP potential file. These files are backward-compatible with the MLIP-3 format. For GPU variants, a chunk size must be provided using the `chunksize` keyword to manage memory usage. If the total number of atoms exceeds the specified chunk size, the simulation proceeds in multiple chunks. For optimal performance, the chunk size should be tuned to ensure sufficient parallelism while avoiding excessive memory usage (which can lead to contention) and minimizing the occurrence of a small final chunk (which can degrade performance due to underutilization).

### Inference

Here are example invocations for inference.

```
pair_style mtp path/to/mtp/file
pair_style mtp/kk path/to/mtp/file chunksize 32768
pair_style mtp/small/kk path/to/mtp/file chunksize 32768
```

### Active Learning

In active learning, the mode, either configuration or neighborhood, is read from the MTP file. For neighborhood active learning variants, we support both LAMMPS-like and MLIP-3-like processing of extrapolation grades.

#### LAMMPS-style extrapolation grades

In the former style, the pair style is invoked much as it is in inference.

```
pair_style mtp/extrapolation path/to/mtp/file
pair_style mtp/extrapolation/kk path/to/mtp/file chunksize 32768
pair_style mtp/extrapolation/small/kk path/to/mtp/file chunksize 32768
```

A fix is then required to request extrapolation grades every `X` timesteps.

```
fix mtp_grade all pair X mtp/extrapolation extrapolation 1
fix mtp_grade all pair X mtp/extrapolation/kk extrapolation 1
fix mtp_grade all pair X mtp/extrapolation/small/kk extrapolation 1
```

The neighborhood extrapolation grades can then be accessed through the `f_mtp_grade` variable. LAMMPS's `dump` can then be used to periodically write the grades and other desired per-atom properties to a file. Notably, if the user attempts to access grades on timesteps where extrapolation is not being calculated, the values will not be up-to-date.

```
dump my_dump all custom X path/to/dump f_mtp_grade
```

#### MLIP-3-style extrapolation grades

In the MLIP-3 style, the user specifies, in order, the MTP file, the output file, the selection threshold, and the break threshold. Extrapolation is evaluated every timestep, and should the maximum grade surpass the selection threshold, the current cell is written to the output file in the MLIP-3 format. Should this maximum grade surpass the break threshold, the simulation is immediately terminated. GPU variants still require the chunk size.

```
pair_style mtp/extrapolation path/to/mtp/file \
    path/to/output 2 10
pair_style mtp/extrapolation/kk path/to/mtp/file \
    path/to/output 2 10 chunksize 32768
pair_style mtp/extrapolation/small/kk path/to/mtp/file \
    path/to/output 2 10 chunksize 32768
```

Configuration mode is only available with the MLIP-3 style.

#### Active Learning Details

In either mode, the maximum extrapolation grade at each timestep is available as a LAMMPS variable through a LAMMPS compute.

```
compute max_grade all pair mtp/extrapolation
compute max_grade all pair mtp/extrapolation/kk
compute max_grade all pair mtp/extrapolation/small/kk
```

The variable can be viewed as usual through `c_max_grade[1]`. Commonly, the user will print the grade along with other per-timestep quantities at regular intervals with the LAMMPS `thermo` command. This variable can be used with other commands such as `fix halt`. Notably, if the user attempts to access this variable on timesteps where extrapolation is not being calculated, the value will not be up-to-date.

```
thermo_style custom step c_max_grade[1]
thermo X
```

Much like some other machine learning potentials in LAMMPS, when invoking a LAMMPS script utilizing an MTP Kokkos GPU variant through the command line, additional flags are required:

```bash
-pk kokkos newton on neigh half
```

### Example

A full example script is available below:

```sh
units metal
dimension       3
boundary        p p p
atom_style      atomic

lattice         bcc 5.28
region          box block 0 3 0 3 0 3 units lattice
create_box      1 box
create_atoms    1 region box
mass 1 39.0983

pair_style mtp path/to/mtp/file
# pair_style mtp/kk path/to/mtp/file chunksize 32768
# pair_style mtp/small/kk path/to/mtp/file chunksize 32768

# pair_style mtp/extrapolation      path/to/mtp/file ./pre.cfg 10 10
# pair_style mtp/extrapolation/kk      path/to/mtp/file ./pre.cfg 10 10 chunksize 32768
# pair_style mtp/extrapolation/small/kk      path/to/mtp/file ./pre.cfg 10 10 chunksize 32768
pair_coeff      * * # Not required

run 0
velocity all create 200.0 12345 mom yes rot yes
fix 1 all nve
run 100
```

If you were to run on a single GPU, you could use the following command. Further documentation is available from the [Kokkos package page](https://docs.lammps.org/Speed_kokkos.html).

```sh
mpirun -np 1 /path/to/executable -in /path/to/script -k on g 1 -sf kk -pk kokkos newton on neigh half
```
