/**
 * The file in which you will implement your DenseGraph GPU solutions!
 */

#include <cstddef>  // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

namespace csc485b {
namespace a2      {

/**
 * A DenseGraph is optimised for a graph in which the number of edges
 * is close to n(n-1). It is represented using an adjacency matrix.
 */
struct DenseGraph
{
	std::size_t n; /**< Number of nodes in the graph. */
	node_t * adjacencyMatrix; /** Pointer to an n x n adj. matrix */

	/** Returns number of cells in the adjacency matrix. */
	__device__ __host__ __forceinline__
	std::size_t matrix_size() const { return n * n; }
};


namespace gpu {


/**
 * Constructs a DenseGraph from an input edge list of m edges.
 *
 * @pre The pointers in DenseGraph g have already been allocated.
 */
__global__
void build_graph( DenseGraph g, edge_t const * edge_list, std::size_t m )
{
	int const tid  = threadIdx.y * blockDim.x + threadIdx.x; // 0..1023 inside the block
	std::size_t const th_id = blockIdx.x * ( blockDim.x * blockDim.y ) + tid; // global id
	if ( th_id >= m ) { return; }
	int x = edge_list[ th_id ].x;
	int y = edge_list[ th_id ].y;
	g.adjacencyMatrix[ x * g.n + y ] = 1;
	return;
}

inline void launch_build( DenseGraph g, edge_t const * d_edges, std::size_t m )
{
	dim3 const threads( 32, 32 );
	std::size_t const blocks = ( m + 1023 ) / 1024;
	build_graph<<< blocks, threads >>>( g, d_edges, m );
}

/**
  * Repopulates the adjacency matrix as a new graph that represents
  * the two-hop neighbourhood of input graph g
  */
__global__
void two_hop_reachability( DenseGraph g, node_t * result )
{
	std::size_t const i = blockIdx.y * blockDim.y + threadIdx.y;   // row
	std::size_t const j = blockIdx.x * blockDim.x + threadIdx.x;   // column
	if ( i >= g.n || j >= g.n ) return;

	// square: count paths i -> k -> j
	node_t sum = 0;
	for ( std::size_t k = 0; k < g.n; ++k )
		sum += g.adjacencyMatrix[ i * g.n + k ] * g.adjacencyMatrix[ k * g.n + j ];

	// clamp to 0/1 and remove the diagonal
	result[ i * g.n + j ] = ( sum > 0 && i != j ) ? 1 : 0;
}

__global__
void two_hop_reachability_tile( DenseGraph g, node_t * result)
{
	__shared__ node_t row[ 32 ][ 32 ];
	__shared__ node_t column[ 32 ][ 32 ];
	std::size_t const i = blockIdx.y * blockDim.y + threadIdx.y;   // row
	std::size_t const j = blockIdx.x * blockDim.x + threadIdx.x;   // column
	std::size_t num_of_rounds = ( g.n + 31 ) / 32;
	node_t sum = 0;
	for(std::size_t round = 0; round < num_of_rounds; ++round){
		row[ threadIdx.y ][ threadIdx.x ] = ( i < g.n && round * 32 + threadIdx.x  < g.n ) ? g.adjacencyMatrix[ i * g.n + round * 32 + threadIdx.x ] : 0;
		column[ threadIdx.y ][ threadIdx.x ] = ( j < g.n && round * 32 + threadIdx.y < g.n ) ? g.adjacencyMatrix[ ( round * 32 + threadIdx.y ) * g.n + j ] : 0; 
		__syncthreads();
		for ( int k = 0; k < 32; ++k )
			sum += row[ threadIdx.y ][ k ] * column[ k ][ threadIdx.x ];
		__syncthreads();
	}
	if( i < g.n && j < g.n )
		result[ i * g.n + j ] = ( sum > 0 && i != j ) ? 1 : 0;
}

__global__
void two_hop_reachability_warp( DenseGraph g, node_t * result){
	__shared__ unsigned int rowBits[ 32 ];
	__shared__ unsigned int colBits[ 32 ];
	__shared__ int all_found[ 2 ];
	std::size_t const i = blockIdx.y * blockDim.y + threadIdx.y;   // row
	std::size_t const j = blockIdx.x * blockDim.x + threadIdx.x;   // column
	std::size_t num_of_rounds = ( g.n + 31 ) / 32;
	unsigned int found = 0;
	for (std::size_t round = 0; round < num_of_rounds; ++round){
		unsigned int s = round % 2;
		unsigned int row_val = ( ( blockIdx.y * 32 + threadIdx.y ) < g.n && round * 32 + threadIdx.x < g.n ) ? g.adjacencyMatrix[ ( blockIdx.y * 32 + threadIdx.y ) * g.n + (round * 32 + threadIdx.x) ] : 0;
		unsigned int col_val = ( ( blockIdx.x * 32 + threadIdx.y ) < g.n && round * 32 + threadIdx.x < g.n ) ? g.adjacencyMatrix[ ( blockIdx.x * 32 + threadIdx.y ) * g.n + (round * 32 + threadIdx.x) ] : 0; 
		unsigned int row_val_word = __ballot_sync( 0xFFFFFFFFu, row_val != 0 );
		unsigned int col_val_word = __ballot_sync( 0xFFFFFFFFu, col_val != 0 );
		rowBits[ threadIdx.y ] = row_val_word;
		colBits[ threadIdx.y ] = col_val_word;
		all_found[ s ] = 1;
		__syncthreads();
		// https://www.geeksforgeeks.org/cpp/cpp-bitwise-operators/
		found |= ( rowBits[ threadIdx.y ] & colBits[ threadIdx.x ] ) != 0;
		if ( !found ) all_found[ s ] = 0;
		__syncthreads();
		if ( all_found[ s ] == 1 ) break;
	}
	if( i < g.n && j < g.n )
		result[ i * g.n + j ] = ( found && i != j ) ? 1 : 0;
}


inline void launch_two_hop( DenseGraph g )
{
	std::size_t const bytes = sizeof( node_t ) * g.n * g.n;
	node_t * result;
	cudaMalloc( (void**)&result, bytes );

	dim3 const threads( 32, 32 );
	dim3 const blocks( ( g.n + 31 ) / 32, ( g.n + 31 ) / 32 );   // 2D grid: one thread per cell
	// two_hop_reachability<<< blocks, threads >>>(g, result);
	// two_hop_reachability_tile<<< blocks, threads >>>( g, result );
	two_hop_reachability_warp<<< blocks, threads >>>( g, result );
	cudaMemcpy( g.adjacencyMatrix, result, bytes, cudaMemcpyDeviceToDevice );   // "in place"
	cudaFree( result );
}

} // namespace gpu
} // namespace a2
} // namespace csc485b