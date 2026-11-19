/**
 * The file in which you will implement your SparseGraph GPU solutions!
 */

#include <cstddef>  // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

namespace csc485b {
namespace a2      {

/**
 * A SparseGraph is optimised for a graph in which the number of edges
 * is close to cn, for a small constanct c. It is represented in CSR format.
 */
struct SparseGraph
{
  std::size_t n; /**< Number of nodes in the graph. */
  std::size_t m; /**< Number of edges in the graph. */
  node_t * neighbours_start_at; /** Pointer to an n=|V| offset array */
  node_t * neighbours; /** Pointer to an m=|E| array of edge destinations */
};


namespace gpu {

__device__
node_t warp_exclusive_scan( node_t val )
{
    unsigned constexpr FULL_MASK = 0xFFFFFFFFu;
    int const lane = threadIdx.x & 31;
    for( int stride = 1; stride < 32; stride <<= 1 )
    {
        node_t const other = __shfl_up_sync( FULL_MASK, val, stride );
        if( lane >= stride ) val += other;
    }
    node_t const prev = __shfl_up_sync( FULL_MASK, val, 1 );
    return ( lane == 0 ) ? 0 : prev;
}

__device__
node_t block_exclusive_scan( node_t count, node_t & block_total )
{
    __shared__ node_t sum_of_warp_before[ 32 ];
    __shared__ node_t total;
    int const lane = threadIdx.x;
    int const warp = threadIdx.y;
    int const tid  = warp * 32 + lane;

    node_t v = warp_exclusive_scan( count );
    if ( lane == 31 ) sum_of_warp_before[ warp ] = v + count;
    __syncthreads();

    if ( warp == 0 )
    {
        node_t t = sum_of_warp_before[ lane ];
        sum_of_warp_before[ lane ] = warp_exclusive_scan( t );
    }
    __syncthreads();

    v += sum_of_warp_before[ warp ];
    if ( tid == 1023 ) total = v + count; // last thread knows the block total
    __syncthreads();

    block_total = total;
    __syncthreads(); // so the next call can reuse the shared arrays
    return v;
}

// 1. make histogram
__global__
void count_degrees( SparseGraph g, edge_t const * edge_list, std::size_t m )
{
    std::size_t const th_id = blockIdx.x * ( blockDim.x * blockDim.y ) + threadIdx.y * blockDim.x + threadIdx.x;
    if ( th_id < m ) atomicAdd( &g.neighbours_start_at[ edge_list[th_id].x ], 1 );
}

// 2. Exclusive Scan
__global__
void scan_offsets( SparseGraph g, node_t * copy )
{
    int const tid = threadIdx.y * blockDim.x + threadIdx.x;
    node_t carry = 0;
    for ( std::size_t base = 0; base < g.n; base += 1024 )
    {
        std::size_t const i = base + tid;
        node_t count = ( i < g.n ) ? g.neighbours_start_at[ i ] : 0;
        node_t chunk_total;
        node_t v = block_exclusive_scan( count, chunk_total );
        if ( i < g.n )
        {
            g.neighbours_start_at[ i ] = carry + v; // exclusive: start of list
            copy[ i ] = carry + v;
        }
        carry += chunk_total;
    }
}


// 3. many blocks: scatter using copy
__global__
void scatter( SparseGraph g, edge_t const * edge_list, std::size_t m, node_t * copy )
{
    std::size_t const th_id = blockIdx.x * ( blockDim.x * blockDim.y ) + threadIdx.y * blockDim.x + threadIdx.x;
    if ( th_id < m )
    {
        int x = edge_list[ th_id ].x;
        int y = edge_list[ th_id ].y;
        int slot = atomicAdd( &copy[x], 1 );
        g.neighbours[ slot ] = y;
    }
}

inline void launch_build( SparseGraph g, edge_t const * d_edges, std::size_t m )
{
    dim3 const threads( 32, 32 );
    std::size_t const blocks = ( m + 1023 ) / 1024;

    node_t * copy;
    cudaMalloc( (void**)&copy, sizeof( node_t ) * g.n );

    count_degrees <<< blocks, threads >>>( g, d_edges, m );
    scan_offsets <<< 1, threads >>>( g, copy );
    scatter <<< blocks, threads >>>( g, d_edges, m, copy );

    cudaDeviceSynchronize();
    cudaFree( copy );
}



/**
 * Constructs a SparseGraph from an input edge list of m edges.
 *
 * @pre The pointers in SparseGraph g have already been allocated.
 */
// __global__
// void build_graph( SparseGraph g, edge_t const * edge_list, std::size_t m )
// {
//     if ( blockIdx.x != 0 ) return;
    
//     __shared__ node_t copy[ 1024 ];  
//     __shared__ node_t sum_of_warp_before[ 32 ];
//     int const tid  = threadIdx.y * blockDim.x + threadIdx.x;
//     int const lane = threadIdx.x;
//     int const warp = threadIdx.y;  
//     std::size_t const th_id = blockIdx.x * ( blockDim.x * blockDim.y ) + tid;  
//     if ( tid == 0 ) printf( "m = %llu\n", (unsigned long long) g.m );
//     // Make Histogram of value x
//     for ( std::size_t e = tid; e < m; e += 1024 )
//         atomicAdd( &g.neighbours_start_at[ edge_list[e].x ], 1 );
//     __syncthreads();

//     // Exclusive scan
//     node_t count = ( tid < g.n ) ? g.neighbours_start_at[ tid ] : 0;
//     node_t v = warp_exclusive_scan(count);

//     // each warp's last lane saves the warp total
//     if ( lane == 31 ) sum_of_warp_before[ warp ] = v + count;
//     __syncthreads();
 
//     // warp 0 turns the totals into "sum of all warps before me"
//     if ( warp == 0 )
//     {
//         node_t total = sum_of_warp_before[ lane ];
//         sum_of_warp_before[ lane ] = warp_exclusive_scan( total );
//     }
//     __syncthreads();

//     // every thread adds its warp's offset
//     v += sum_of_warp_before[ warp ];

//     if ( tid < g.n ) {
//         g.neighbours_start_at[ tid ] = v;
//         copy[ tid ] = v;
//     }
//     __syncthreads();

//     // if ( th_id < g.n ) g.neighbours_start_at[th_id] = v;          // store it back
//     // copy[th_id] = v;   
//     // __syncwarp();

//     // Make Scatter using histogram
//     for ( std::size_t e = tid; e < m; e += 1024 )
//     {
//         int x = edge_list[ e ].x;
//         int y = edge_list[ e ].y;
//         int slot = atomicAdd( &copy[x], 1 );
//         g.neighbours[slot] = y;
//     }
// }

/**
  * Repopulates the adjacency lists as a new graph that represents
  * the two-hop neighbourhood of input graph g
  */
__global__
void two_hop_reachability( SparseGraph g )
{
    // IMPLEMENT ME!
    // algorithm unknown
    return;
}

inline void launch_two_hop( SparseGraph g )
{
    // not implemented :)
}

} // namespace gpu
} // namespace a2
} // namespace csc485b