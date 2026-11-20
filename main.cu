/**
 * Driver for the benchmark comparison. Generates random data,
 * runs the CPU baseline, and then runs your code.
 */

#include <chrono>   // for timing
#include <iostream> // std::cout, std::endl
#include <iterator> // std::ostream_iterator
#include <vector>

#include "dense_graph.h"
#include "sparse_graph.h"

#include "data_generator.h"
#include "data_types.h"

/**
 * Runs timing tests on a CUDA graph implementation.
 * Consists of independently constructing the graph and then
 * modifying it to its two-hop neighbourhood.
 */
template < typename DeviceGraph >
void run( DeviceGraph g, csc485b::a2::edge_t const * d_edges, std::size_t m )
{
    cudaDeviceSynchronize();
    auto const build_start = std::chrono::high_resolution_clock::now();

    csc485b::a2::gpu::launch_build(g, d_edges, m);
    cudaDeviceSynchronize();
    auto const reachability_start = std::chrono::high_resolution_clock::now();

    // neither does this!
    csc485b::a2::gpu::launch_two_hop( g );

    cudaDeviceSynchronize();
    auto const end = std::chrono::high_resolution_clock::now();

    std::cout << "Build time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(reachability_start - build_start).count()
              << " us"
              << std::endl;

    std::cout << "Reachability time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - reachability_start).count()
              << " us"
              << std::endl;
}

/**
 * CPU reference for dense two-hop reachability.
 * Returns true if result[i*n + j] == 1 exactly when j is reachable from i
 * by a path of length 2 (i != j). Prints the first mismatch otherwise.
 */
bool check_dense_two_hop( std::vector< csc485b::a2::edge_t > const & edges,
                          std::size_t n,
                          std::vector< csc485b::a2::node_t > const & result )
{
    std::size_t const words = ( n + 63 ) / 64;   // 64 cells per word

    // adjacency matrix as bit rows
    std::vector< unsigned long long > adj( n * words, 0 );
    for ( auto const & e : edges )
        adj[ e.x * words + e.y / 64 ] |= 1ULL << ( e.y % 64 );

    std::vector< unsigned long long > row( words );
    for ( std::size_t i = 0; i < n; ++i )
    {
        // row i of the two-hop = OR of rows k, for every neighbour k of i
        for ( std::size_t w = 0; w < words; ++w ) row[w] = 0;
        for ( std::size_t k = 0; k < n; ++k )
            if ( ( adj[ i * words + k / 64 ] >> ( k % 64 ) ) & 1ULL )
                for ( std::size_t w = 0; w < words; ++w )
                    row[w] |= adj[ k * words + w ];

        row[ i / 64 ] &= ~( 1ULL << ( i % 64 ) );   // remove the diagonal

        // compare with the GPU result
        for ( std::size_t j = 0; j < n; ++j )
        {
            csc485b::a2::node_t const expected = ( row[ j / 64 ] >> ( j % 64 ) ) & 1ULL;
            if ( result[ i * n + j ] != expected )
            {
                std::cout << "first mismatch at (" << i << "," << j << "): expected "
                          << expected << ", got " << result[ i * n + j ] << std::endl;
                return false;
            }
        }
    }
    return true;
}

/**
 * Allocates space for a dense graph and then runs the test code on it.
 */
void run_dense( csc485b::a2::edge_t const * d_edges, std::size_t n, std::size_t m )
{
    using namespace csc485b;

    // allocate device DenseGraph
    a2::node_t * d_matrix;
    cudaMalloc( (void**)&d_matrix, sizeof( a2::node_t ) * n * n );
    cudaMemset( d_matrix, 0, sizeof( a2::node_t ) * n * n );
    a2::DenseGraph d_dg{ n, d_matrix };

    run( d_dg, d_edges, m );
    
    // check output?
    std::vector< a2::node_t > host_matrix( d_dg.matrix_size() );
    a2::DenseGraph dg{ n, host_matrix.data() };
    cudaMemcpy( dg.adjacencyMatrix, d_dg.adjacencyMatrix, sizeof( a2::node_t ) * d_dg.matrix_size(), cudaMemcpyDeviceToHost );
    // std::copy( host_matrix.cbegin(), host_matrix.cend(), std::ostream_iterator< a2::node_t >( std::cout, " " ) );

    // CPU check: matrix must have a 1 exactly where there is an edge
    std::vector< a2::edge_t > h_edges( m );
    cudaMemcpy( h_edges.data(), d_edges, sizeof( a2::edge_t ) * m, cudaMemcpyDeviceToHost );
    // std::vector< a2::node_t > expected_matrix( n * n, 0 );
    // for ( auto const& e : h_edges ) expected_matrix[ e.x * n + e.y ] = 1;
    // std::cout << ( expected_matrix == host_matrix ? "dense build OK" : "dense build WRONG" ) << std::endl;
    
    std::cout << ( check_dense_two_hop( h_edges, n, host_matrix ) ? "dense two-hop OK" : "dense two-hop WRONG" ) << std::endl;
    // clean up
    cudaFree( d_matrix );
}

/**
 * Allocates space for a sparse graph and then runs the test code on it.
 */
void run_sparse( csc485b::a2::edge_t const * d_edges, std::size_t n, std::size_t m )
{
    using namespace csc485b;

    // allocate device SparseGraph
    a2::node_t * d_offsets, * d_neighbours;
    cudaMalloc( (void**)&d_offsets,    sizeof( a2::node_t ) * n );
    cudaMemset( d_offsets, 0, sizeof( a2::node_t ) * n );
    cudaMalloc( (void**)&d_neighbours, sizeof( a2::node_t ) * m );
    a2::SparseGraph d_sg{ n, m, d_offsets, d_neighbours };

    run( d_sg, d_edges, m );

    // copy results back to host
    std::vector< a2::node_t > host_offsets( n );
    std::vector< a2::node_t > host_neighbours( m );
    cudaMemcpy( host_offsets.data(),    d_offsets,    sizeof( a2::node_t ) * n, cudaMemcpyDeviceToHost );
    cudaMemcpy( host_neighbours.data(), d_neighbours, sizeof( a2::node_t ) * m, cudaMemcpyDeviceToHost );

    // CPU check: count degrees, prefix-sum, compare
    std::vector< a2::node_t > expected( n, 0 );
    std::vector< a2::edge_t > h_edges( m );
    cudaMemcpy( h_edges.data(), d_edges, sizeof( a2::edge_t ) * m, cudaMemcpyDeviceToHost );
    for ( auto const& e : h_edges ) expected[ e.x ]++;
    a2::node_t run_sum = 0;
    for ( std::size_t i = 0; i < n; ++i ) { a2::node_t c = expected[i]; expected[i] = run_sum; run_sum += c; }
    std::cout << ( expected == host_offsets ? "offsets OK" : "offsets WRONG" ) << std::endl;


    // print them
    // std::cout << "\nneighbours_start_at: ";
    // std::copy( host_offsets.cbegin(), host_offsets.cend(), std::ostream_iterator< a2::node_t >( std::cout, " " ) );
    // std::cout << "\nneighbours: ";
    // std::copy( host_neighbours.cbegin(), host_neighbours.cend(), std::ostream_iterator< a2::node_t >( std::cout, " " ) );
    // std::cout << std::endl;
    
    // clean up
    cudaFree( d_neighbours );
    cudaFree( d_offsets );
}

int main()
{
    using namespace csc485b;

    // Create input
    std::size_t constexpr n = 4096;
    std::size_t constexpr expected_degree = n >> 1;

    a2::edge_list_t const graph = a2::generate_graph( n, n * expected_degree );
    //a2::edge_list_t const graph = a2::generate_specific_graph();
    std::size_t const m = graph.size();

    // lazily echo out input graph
    // for( auto const& e : graph )
    // {
    //     std::cout << "(" << e.x << "," << e.y << ") ";
    // }

    // allocate and memcpy input to device
    a2::edge_t * d_edges;
    cudaMalloc( (void**)&d_edges, sizeof( a2::edge_t ) * m );
    cudaMemcpyAsync( d_edges, graph.data(), sizeof( a2::edge_t ) * m, cudaMemcpyHostToDevice );
    // run your code!
    run_dense ( d_edges, n, m );
    cudaDeviceSynchronize();
    run_sparse( d_edges, n, m );

    return EXIT_SUCCESS;
}