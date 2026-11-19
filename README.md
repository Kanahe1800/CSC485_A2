# CSC485A A2: Graph Construction and Two-Hop Reachability

## Requirements

- NVIDIA GPU with compute capability 7.5 or newer (e.g. T4)
- CUDA Toolkit (`nvcc`)
- Windows: Visual Studio Build Tools (provides `nmake` and the C++ compiler)

## Files

- `main.cu`: driver (generates the graph, runs the dense and sparse versions, times them, checks the results on the CPU)
- `dense_graph.h`: `DenseGraph` (adjacency matrix): build and two-hop kernels
- `sparse_graph.h`: `SparseGraph` (CSR): build kernels (count, scan, scatter)
- `data_generator.h`, `data_types.h`, `cuda_common.h`: provided helpers
- `Makefile`: build script for `nmake`

## Build

From the project folder, run:

    nmake

This runs:

    nvcc -O3 -std=c++20 -arch=native main.cu -o a2.exe

On Google Colab (T4), use `-arch=sm_75` instead of `-arch=native`.

To remove the build output:

    nmake clean

## Run

    a2.exe

or build and run in one step:

    nmake run

On Linux or Colab:

    ./a2.exe

## Changing the input

In `main()` in `main.cu`, set the number of nodes and the average degree, then rebuild:

    std::size_t constexpr n = 4096;
    std::size_t constexpr expected_degree = n >> 1;   // about half of all pairs connected

The graph density is about `expected_degree / n`. For a sparse graph (below 2.5%), use a small value such as `expected_degree = 64`.

## Choosing the dense two-hop version

`launch_two_hop( DenseGraph )` in `dense_graph.h` launches one of these kernels (comment out the others):

| Kernel | Version |
|---|---|
| `two_hop_reachability` | one thread per output cell |
| `two_hop_reachability_tile` | shared-memory tiling (32 x 32 tiles) |
| `two_hop_reachability_warp` | `__ballot_sync` bit packing, bitwise AND, block-wide early exit |

## Output

The program runs the dense graph first, then the sparse graph. For each one it prints:

1. `Build time`: time to build the graph from the edge list, in microseconds
2. `Reachability time`: time for the two-hop reachability step, in microseconds
3. a correctness check computed on the CPU

Example:

    Build time: <t> us
    Reachability time: <t> us
    dense two-hop OK
    Build time: <t> us
    Reachability time: <t> us
    offsets OK

- `dense two-hop OK`: the dense result matches a CPU two-hop computation (`WRONG` and the first mismatching cell are printed otherwise).
- `offsets OK`: the sparse `neighbours_start_at` array matches a CPU prefix sum of the node degrees.

The sparse two-hop reachability is not implemented, so its reachability time is close to 0.