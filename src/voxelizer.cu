#include "host_device_interface.h"
#include "device_code.h"
#include "helper_math.h"
#include "common.h"

#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
//#include <sm_11_atomic_functions.h>
//#include <sm_12_atomic_functions.h>
#include <sm_20_atomic_functions.h>
#include <sm_35_atomic_functions.h>
#include <math_functions.h>

#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#if defined(unix) || defined(__unix__) || defined(__unix)
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#endif 

#include <iostream>
#include <ctime>
#include <float.h>

namespace vox {

///////////////////////////////////////////////////////////////////////////////
/// Handles the sorting of the <em>work queue</em> by using \a thrust.
/// 
/// \throws Exception if CUDA reports an error during execution.
///
/// \tparam Node Type of \p Node used.
///
/// \param[in,out] devData Reference to all relevant, device-specific 
///                        variables.
/// \param[in] startTime Time when the program started executing.
/// \param[in] verbose Whether or not to print what is being done.
///////////////////////////////////////////////////////////////////////////////
void sortWorkQueue(
    CommonDevData const & devData,
    uint                * workQueueTriangles_gpu,
    uint                * workQueueTiles_gpu,
    clock_t	              startTime, 
    bool		          verbose )
{
    if (verbose) 
        std::cout << (clock() - startTime) << ": Calling sortWorkQueue.\n";

    thrust::device_ptr<uint> wqtri = thrust::device_pointer_cast<uint>(
        workQueueTriangles_gpu );
    thrust::device_ptr<uint> wqtil = thrust::device_pointer_cast<uint>(
        workQueueTiles_gpu );

    thrust::sort_by_key( wqtil, wqtil + devData.workQueueSize, wqtri );

    checkCudaErrors( "sortWorkQueue" );
}
///////////////////////////////////////////////////////////////////////////////
/// Handles the compacting of the <em>work queue</em> by using \a thrust. The 
/// algorithm does not just compact the <em>work queue</em> so that each \a 
/// tile is represented only once. It also replaces the <em>triangle id</em> 
/// with an \a offset into the <em>work queue</em> where the pairs involving 
/// that particular \a tile begin.
/// 
/// \throws Exception if CUDA reports an error during execution.
///
/// \tparam Node Type of \p Node used.
///
/// \param[in,out] devData Reference to all relevant, device-specific 
///                        variables.
/// \param[in] startTime Time when the program started executing.
/// \param[in] verbose Whether or not to print what is being done.
///////////////////////////////////////////////////////////////////////////////
void compactWorkQueue(
    CommonDevData & devData,
    uint          * workQueueTiles_gpu,
    uint          * tileList_gpu,
    uint          * tileOffsets_gpu,
    clock_t		    startTime, 
    bool	        verbose )
{
    if (verbose) 
        std::cout << (clock() - startTime) << ": Calling compactWorkQueue.\n";

    thrust::device_ptr<uint> wqtil = 
        thrust::device_pointer_cast<uint>(workQueueTiles_gpu);
    thrust::device_ptr<uint> iv;
    thrust::pair<thrust::device_ptr<uint>, thrust::device_ptr<uint> > ne;
    thrust::device_ptr<uint> tl;
    thrust::device_ptr<uint> to;

    tl = thrust::device_pointer_cast(tileList_gpu);
    to = thrust::device_pointer_cast(tileOffsets_gpu);
    iv = thrust::device_malloc<uint>(devData.maxWorkQueueSize);
    thrust::sequence(iv, iv + devData.maxWorkQueueSize, 0, 1);

    ne = thrust::unique_by_key_copy( wqtil, 
                                     wqtil + devData.workQueueSize, 
                                     iv, 
                                     tl, 
                                     to );

    devData.nrValidElements = uint( ne.first - tl );

    // devData.tileList_gpu = thrust::raw_pointer_cast(tl);
    // devData.tileOffsets_gpu = thrust::raw_pointer_cast(to);

    thrust::device_free(iv);

    checkCudaErrors( "compactWorkQueue" );
}
///////////////////////////////////////////////////////////////////////////////
///
///////////////////////////////////////////////////////////////////////////////
template <class Node>
void calcSurfNodeCount
    ( CommonDevData & devData
    , Node * nodes
    , clock_t startTime
    , bool verbose
    )
{
    if (verbose) 
        std::cout << (clock() - startTime) << ": Calling calcSurfNodeCount.\n";

    thrust::device_ptr<Node> nDevPtr = thrust::device_pointer_cast<Node>( nodes );

    uint count = thrust::count_if( nDevPtr
                                 , nDevPtr + devData.nrOfNodes
                                 , NodeBidEquals<Node, 1>() );

    devData.nrOfSurfaceNodes = count;

    checkCudaErrors( "calcSurfNodeCount" );
}
///////////////////////////////////////////////////////////////////////////////
/// Handles the calculating of the <em>tile overlap buffer</em> by calling the 
/// \p calculateTileOverlap() kernel. 
/// 
/// \throws Exception if CUDA reports an error during execution.
///
/// \tparam Node Type of \p Node used.
///
/// \param[in,out] devData Reference to all relevant, device-specific 
///                        variables.
/// \param[in] hostData Reference to all relevant non device-specific 
///                     variables.
/// \param[in] yzSubSpace The bounding box for the subspace that is to be 
///                       voxelized.
/// \param[in] startTime Time when the program started executing.
/// \param[in] verbose Whether or not to print what is being done.
///////////////////////////////////////////////////////////////////////////////
void calcTileOverlap(
    CommonDevData  const & devData,
    CommonHostData const & hostData,
    float          const * vertices_gpu,
    uint           const * indices_gpu,
    uint                 * tileOverlaps_gpu,
    Bounds<uint2>  const & yzSubSpace, 
    clock_t				   startTime, 
    bool				   verbose )
{
    uint blocks = devData.blocks;
    uint threadsPerBlock = devData.threads;

    if (verbose) 
        std::cout << (clock() - startTime) << ": Calling calculateTileOverlap"
                     "<<<" << blocks << ", " << threadsPerBlock << ">>> (" << 
                     yzSubSpace.min.x << ", " << yzSubSpace.min.y << ")\n";

    calculateTileOverlap
        <<< blocks, 
            threadsPerBlock >>>( vertices_gpu
                               , indices_gpu
                               , tileOverlaps_gpu
                               , hostData.nrOfTriangles
                               , devData.extMinVertex
                               , hostData.voxelLength
                               , devData.extResolution
                               , yzSubSpace );

    checkCudaErrors( "calcTileOverlap" );
}
///////////////////////////////////////////////////////////////////////////////
/// Handles the calculation of the <em>work queue</em> by calling the 
/// \p constructWorkQueue() kernel.
/// 
/// \throws Exception if CUDA reports an error during execution.
///
/// \tparam Node Type of \p Node used.
///
/// \param[in,out] devData Reference to all relevant, device-specific 
///                variables.
/// \param[in] hostData Reference to all relevant non device-specific 
///                        variables.
/// \param[in] yzSubSpace The bounding box for the subspace that is to be 
///                       voxelized.
/// \param[in] startTime Time when the program started executing.
/// \param[in] verbose Whether or not to print what is being done.
///////////////////////////////////////////////////////////////////////////////
void calcWorkQueue(
    CommonDevData  const & devData,
    CommonHostData const & hostData,
    float          const * vertices_gpu,
    uint           const * indices_gpu,
    uint                 * workQueueTriangles_gpu,
    uint                 * workQueueTiles_gpu,
    uint           const * offsetBuffer_gpu,
    Bounds<uint2>  const & yzSubSpace,
    clock_t			       startTime, 
    bool				   verbose )
{
    uint blocks = devData.blocks;
    uint threadsPerBlock = devData.threads;

    if (verbose) 
        std::cout << (clock() - startTime) << ": Calling constructWorkQueue"
                     "<<<" << blocks << ", " << threadsPerBlock << ">>> (" << 
                     yzSubSpace.min.x << ", " << yzSubSpace.min.y << ")\n";

    constructWorkQueue
        <<< blocks, 
            threadsPerBlock >>>( vertices_gpu, 
                                 indices_gpu, 
                                 workQueueTriangles_gpu, 
                                 workQueueTiles_gpu, 
                                 offsetBuffer_gpu, 
                                 hostData.nrOfTriangles, 
                                 devData.extMinVertex, 
                                 devData.firstTriangleWithTiles, 
                                 hostData.voxelLength, 
                                 devData.extResolution, 
                                 yzSubSpace );

    checkCudaErrors( "calcWorkQueue" );
}
///////////////////////////////////////////////////////////////////////////////
/// Handles the calculation of the \a voxelization by calling the 
/// \p generateVoxelization() kernel.
/// 
/// \throws Exception if CUDA reports an error during execution.
///
/// \tparam Node Type of \p Node used.
///
/// \param[in,out] devData Reference to all relevant, device-specific 
///                           variables.
/// \param[in] hostData Reference to all relevant non device-specific 
///                        variables.
/// \param[in] subSpace The bounding box for the subspace that is to be 
///                     voxelized.
/// \param[in] startTime Time when the program started executing.
/// \param[in] verbose Whether or not to print what is being done.
///////////////////////////////////////////////////////////////////////////////
void calcVoxelization(
    CommonDevData  const & devData,
    CommonHostData const & hostData,
    float          const * vertices_gpu,
    uint           const * indices_gpu,
    uint           const * workQueueTriangles_gpu,
    uint           const * workQueueTiles_gpu,
    uint           const * tileList_gpu,
    uint           const * tileOffsets_gpu,
    VoxInt               * voxels_gpu,
    Bounds<uint3>  const & subSpace, 
    clock_t                startTime, 
    bool				   verbose )
{
    uint const blocks = devData.nrValidElements;
    dim3 const threadsPerBlock(16, 4);

    if (verbose) 
        std::cout << (clock() - startTime) << ": Calling generateVoxelization"
                     "<<<" << blocks << ", (" << threadsPerBlock.x << ", " << 
                     threadsPerBlock.y << ")>>> (" << subSpace.min.x << ", " << 
                     subSpace.min.y << ", " << subSpace.min.z << ")\n";
    
    generateVoxelization
        <<< blocks, 
            threadsPerBlock >>>( vertices_gpu, 
                                 indices_gpu, 
                                 workQueueTriangles_gpu, 
                                 workQueueTiles_gpu, 
                                 tileList_gpu, 
                                 tileOffsets_gpu, 
                                 voxels_gpu, 
                                 hostData.nrOfTriangles, 
                                 devData.workQueueSize, 
                                 devData.extMinVertex, 
                                 hostData.voxelLength, 
                                 devData.left,
                                 devData.right,
                                 devData.up,
                                 devData.down,
                                 devData.extResolution, 
                                 subSpace ); 

    checkCudaErrors( "calcVoxelization" );
}
///////////////////////////////////////////////////////////////////////////////
/// Handles the conversion to the \p Node representation by calling the 
/// \p constructNodeList2() kernel.
/// 
/// \throws Exception if CUDA reports an error during execution.
///
/// \tparam Node Type of \p Node used.
///
/// \param[in,out] devData Reference to all relevant, device-specific 
///                variables.
/// \param[in] yzSubSpace The bounding box for the subspace that is to be 
///                       voxelized.
/// \param[in] startTime Time when the program started executing.
/// \param[in] verbose Whether or not to print what is being done.
///////////////////////////////////////////////////////////////////////////////
template <class Node> void calcNodeList(
    CommonDevData  const & devData,
    VoxInt         const * voxels_gpu,
    Node                 * nodes_gpu,
    Bounds<uint2>  const & yzSubSpace, 
    clock_t				   startTime, 
    bool				   verbose )
{
    uint blocks = devData.blocks;
    dim3 threadsPerBlock( devData.threads >> VOX_DIV, 32 );

    if (verbose) 
        std::cout << (clock() - startTime) << ": Calling constructNodeList"
                     "<<<" << blocks << ", (" << threadsPerBlock.x << ", " << 
                     threadsPerBlock.y << ")>>> (" << yzSubSpace.min.x << ", " 
                     << yzSubSpace.min.y << ")\n";
    
    constructNodeList2<Node>
        <<< blocks, threadsPerBlock >>>( voxels_gpu,
                                         nodes_gpu,
                                         devData.allocResolution,
                                         yzSubSpace );

    checkCudaErrors( "calcNodeList" );
}
///////////////////////////////////////////////////////////////////////////////
/// Handles the conversion to the <tt>FCC Node</tt> representation by calling 
/// the \p convertToFCCGrid() kernel.
/// 
/// \throws Exception if CUDA reports an error during execution.
///
/// \tparam Node Type of \p Node used. Can be called with \p Node types other 
///              than FCC Nodes, but kernels expecting FCC Nodes don't work 
///              properly with any other types.
///
/// \param[in,out] devData Reference to all relevant, device-specific 
///                variables.
/// \param[in] yzSubSpace The bounding box for the subspace that is to be 
///                       voxelized.
/// \param[in] gridType The type of voxelization (one of four kinds) to be 
///                     processed into the FCC grid. Can take values between 
///                     1 and 4.
/// \param[in] startTime Time when the program started executing.
/// \param[in] verbose Whether or not to print what is being done.
///////////////////////////////////////////////////////////////////////////////
template <class Node> void launchConvertToFCCGrid(
    CommonDevData const & devData, 
    VoxInt        const * voxels_gpu,
    Node                * nodes_gpu,
    Bounds<uint2> const & yzSubSpace, 
    int                   gridType,
    clock_t				  startTime, 
    bool				  verbose )
{
    uint blocks = devData.blocks;
    dim3 threadsPerBlock( devData.threads >> VOX_DIV, 32 );

    if (verbose) 
        std::cout << (clock() - startTime) << ": Calling convertToFCCGrid"
                     "<<<" << blocks << ", (" << threadsPerBlock.x << ", " << 
                     threadsPerBlock.y << ")>>> (" << yzSubSpace.min.x << ", " 
                     << yzSubSpace.min.y << ")\n";
    
    convertToFCCGrid<Node>
        <<< blocks, threadsPerBlock >>>( voxels_gpu,
                                         nodes_gpu,
                                         gridType,
                                         devData.allocResolution,
                                         yzSubSpace );

    checkCudaErrors( "launchConvertToFCCGrid" );
}
///////////////////////////////////////////////////////////////////////////////
/// Handles the calculation of <em>boundary id</em>s by calling the 
/// \p fillNodeList2() kernel.
/// 
/// \throws Exception if CUDA reports an error during execution.
///
/// \tparam Node Type of \p Node used.
///
/// \param[in,out] devData Reference to all relevant, device-specific 
///                           variables.
/// \param[in] yzSubSpace The bounding box for the \a subspace that is to be 
///                       voxelized.
/// \param[in] xSlicing \p true if slicing along the x-axis, \p false in all 
///                     other cases.
/// \param[in] startTime Time when the program started executing.
/// \param[in] verbose Whether or not to print what is being done.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> void procNodeList(
    CommonDevData const & devData, 
    Node                * nodes_gpu,
    Node                * nodesCopy_gpu,
    bool                * error_gpu,
    Bounds<uint2> const & yzSubSpace,
    bool                  xSlicing,
    SNode               * surfNodes,
    clock_t				  startTime, 
    bool				  verbose )
{
    uint blocks = devData.blocks;
    uint threadsPerBlock = devData.threads;

    if (verbose) 
        std::cout << (clock() - startTime) << ": Calling fillNodeList<<<" << 
        blocks << ", " << threadsPerBlock << ">>> (" << yzSubSpace.min.x << 
        ", " << yzSubSpace.min.y << ")\n";

    Node * nodePtr = xSlicing ? nodesCopy_gpu
                              : nodes_gpu;
    
    fillNodeList2<Node,SNode>
        <<< blocks, threadsPerBlock >>>( nodePtr, 
                                         devData.allocResolution, 
                                         yzSubSpace,
                                         error_gpu,
                                         devData.hashMap,
                                         surfNodes );

    cudaDeviceSynchronize();

    checkCudaErrors( "procNodeList" );
}
///////////////////////////////////////////////////////////////////////////////
/// Handles the calculation of <em>boundary id</em>s by calling the 
/// \p calculateFCCBoundaries() kernel.
/// 
/// \throws Exception if CUDA reports an error during execution.
///
/// \tparam Node Type of \p Node used.
///
/// \param[in,out] devData Reference to all relevant, device-specific 
///                           variables.
/// \param[in] yzSubSpace The bounding box for the \a subspace that is to be 
///                       voxelized.
/// \param[in] xSlicing \p true if slicing along the x-axis, \p false in all 
///                     other cases.
/// \param[in] startTime Time when the program started executing.
/// \param[in] verbose Whether or not to print what is being done.
///////////////////////////////////////////////////////////////////////////////
template <class Node> void launchCalculateFCCBoundaries(
    CommonDevData const & devData, 
    Node                * nodes_gpu,
    Node                * nodesCopy_gpu,
    Bounds<uint2> const & yzSubSpace,
    bool                  xSlicing,
    clock_t				  startTime, 
    bool				  verbose )
{
    uint blocks = devData.blocks;
    uint threadsPerBlock = devData.threads;

    if (verbose) 
        std::cout << (clock() - startTime) << ": Calling "
                     "calculateFCCBoundaries<<<" << blocks << ", " << 
                     threadsPerBlock << ">>> (" << yzSubSpace.min.x << ", " << 
                     yzSubSpace.min.y << ")\n";

    Node * nodePtr = xSlicing ? nodesCopy_gpu
                              : nodes_gpu;
    
    calculateFCCBoundaries<Node>
        <<< blocks, threadsPerBlock >>>( nodePtr, 
                                         devData.allocResolution, 
                                         yzSubSpace );

    checkCudaErrors( "launchCalculateFCCBoundaries" );
}
///////////////////////////////////////////////////////////////////////////////
/// Handles the calculation of a simple \a surface voxelization by calling the 
/// \p SimpleSurfaceVoxelizer() kernel.
/// 
/// \throws Exception if CUDA reports an error during execution.
///
/// \tparam Node Type of \p Node used.
///
/// \param[in,out] devData Reference to all relevant, device-specific 
///                           variables.
/// \param[in] hostData Reference to all relevant non device-specific 
///                        variables.
/// \param[in] startTime Time when the program started executing.
/// \param[in] verbose Whether or not to print what is being done.
///////////////////////////////////////////////////////////////////////////////
template <class Node> void calcSurfaceVoxelization(
    CommonDevData  const & devData, 
    CommonHostData const & hostData, 
    float          const * vertices_gpu, 
    uint           const * indices_gpu, 
    Node                 * nodes_gpu, 
    uchar          const * materials_gpu, 
    clock_t				   startTime, 
    bool				   verbose )
{
    uint blocks = devData.blocks;
    uint threadsPerBlock = devData.threads;

    if (verbose) 
        std::cout << (clock() - startTime) << 
                ": Calling SimpleSurfaceVoxelization<<<" << blocks << ", " << 
                threadsPerBlock << ">>>\n";
    
    float3 minVertex = make_float3( float(devData.minVertex.x), 
                                    float(devData.minVertex.y), 
                                    float(devData.minVertex.z) );
    float voxelLength = (float)hostData.voxelLength;

    uint3 resolution = devData.resolution.max - devData.resolution.min;

    SimpleSurfaceVoxelizer<Node>
        <<< blocks, threadsPerBlock >>>( vertices_gpu, 
                                         indices_gpu, 
                                         nodes_gpu, 
                                         materials_gpu, 
                                         hostData.nrOfTriangles, 
                                         minVertex, 
                                         voxelLength, 
                                         resolution );

    checkCudaErrors( "calcSurfaceVoxelization" );
}
///////////////////////////////////////////////////////////////////////////////
/// Handles the calculation of the <em>triangle classification</em> by calling 
/// the \p classifyTriangles() kernel, and then using \a thrust to sort and 
/// compact the results.
/// 
/// \throws Exception if CUDA reports an error during execution.
///
/// \tparam Node Type of \p Node used.
///
/// \param[in,out] devData Reference to all relevant, device-specific 
///                           variables.
/// \param[in] hostData Reference to all relevant non device-specific 
///                        variables.
/// \param[in] startTime Time when the program started executing.
/// \param[in] verbose Whether or not to print what is being done.
///////////////////////////////////////////////////////////////////////////////
void calcTriangleClassification(
    CommonDevData  const & devData, 
    CommonHostData       & hostData, 
    float          const * vertices_gpu, 
    uint           const * indices_gpu, 
    uint                 * triangleTypes_gpu,
    uint                 * sortedTriangles_gpu,
    clock_t		           startTime, 
    bool		           verbose )
{
    uint blocks = devData.blocks;
    uint threadsPerBlock = devData.threads;


    if (verbose) 
        std::cout << (clock() - startTime) << ": Calling classifyTriangles<<<" 
                  << blocks << ", " << threadsPerBlock << ">>>\n";

    // Classify the triangles by assigning them their bounding box type, 
    // dominant axis and estimated number of voxel columns to process.
    float3 modelBBMin = make_float3( float(devData.extMinVertex.x), 
                                     float(devData.extMinVertex.y),
                                     float(devData.extMinVertex.z) );

    classifyTriangles
        <<< blocks, threadsPerBlock >>>( vertices_gpu, 
                                         indices_gpu, 
                                         triangleTypes_gpu,
                                         hostData.nrOfTriangles, 
                                         modelBBMin, 
                                         float(hostData.voxelLength) );
    // Sort the triangles by classification.
    thrust::device_ptr<uint> triTypes = 
        thrust::device_pointer_cast<uint>(triangleTypes_gpu);
    thrust::device_ptr<uint> tris = thrust::device_pointer_cast<uint>(
        sortedTriangles_gpu);
    thrust::sequence( tris, tris + hostData.nrOfTriangles, 0, 1 );
    thrust::sort_by_key( triTypes
                       , triTypes + hostData.nrOfTriangles
                       , tris );

    // Calculate the number of different types of triangle.
    uint nrOf0DTris = 
        (uint)thrust::count_if( triTypes, 
                                triTypes + hostData.nrOfTriangles, 
                                is_BB< BBType_Degen, VOX_BPI - 2 >() );
    uint nrOf1DTris = 
        (uint)thrust::count_if( triTypes, 
                                triTypes + hostData.nrOfTriangles, 
                                is_BB< BBType_1D, VOX_BPI - 2 >() );
    uint nrOf2DTris = 
        (uint)thrust::count_if( triTypes, 
                                triTypes + hostData.nrOfTriangles, 
                                is_BB< BBType_2D, VOX_BPI - 2 >() );
    uint nrOf3DTris = 
        (uint)thrust::count_if( triTypes, 
                                triTypes + hostData.nrOfTriangles, 
                                is_BB< BBType_3D, VOX_BPI - 2 >() );

    // Determine start and end indices for each bounding box type.
    hostData.start1DTris = nrOf0DTris;
    hostData.end1DTris = hostData.start1DTris + nrOf1DTris;
    hostData.start2DTris = hostData.end1DTris;
    hostData.end2DTris = hostData.start2DTris + nrOf2DTris;
    hostData.start3DTris = hostData.end2DTris;
    hostData.end3DTris = hostData.start3DTris + nrOf3DTris;

    if (verbose)
        std::cout << "Deg. tris: " << nrOf0DTris << ", 1D tris: " << nrOf1DTris 
                  << ", 2D tris: " << nrOf2DTris << ", 3D tris: " << nrOf3DTris 
                  << ".\n";

    checkCudaErrors( "calcTriangleClassification" );
}
///////////////////////////////////////////////////////////////////////////////
/// Handles the calculation of the <em>surface voxelization</em> by calling the 
/// \p process1DTriangles(), \p process2DTriangles() and \p 
/// process3DTriangles() kernels.
/// 
/// \throws Exception if CUDA reports an error during execution.
///
/// \tparam Node Type of \p Node used.
///
/// \param[in,out] devData Reference to all relevant, device-specific 
///                           variables.
/// \param[in] hostData Reference to all relevant non device-specific 
///                        variables.
/// \param[in] subSpace The bounding box for the subspace that is to be 
///                     voxelized.
/// \param[in] gridType Which of the four grids to use.
/// \param[in] startTime Time when the program started executing.
/// \param[in] verbose Whether or not to print what is being done.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void calcOptSurfaceVoxelization(
    CommonDevData  const & devData, 
    CommonHostData const & hostData, 
    float          const * vertices_gpu, 
    uint           const * indices_gpu, 
    uint           const * triangleTypes_gpu,
    uint           const * sortedTriangles_gpu, 
    uchar          const * materials_gpu, 
    Node                 * nodes_gpu, 
    Bounds<uint3>  const & subSpace,
    int                    gridType,
    bool                   countVoxels,
    SNode                * surfNodes,
    clock_t                startTime, 
    bool                   verbose )
{
    uint blocks = 32;//devData.blocks;
    uint threadsPerBlock = 64;//devData.threads;

    float3 modelBBMin = make_float3( float(devData.extMinVertex.x), 
                                     float(devData.extMinVertex.y), 
                                     float(devData.extMinVertex.z) );

    // Process and voxelize triangles with a 1-dimensional bounding box.
    if (hostData.start1DTris != hostData.end1DTris) 
    {
        if (verbose) 
            std::cout << (clock() - startTime) << 
                    ": Calling process1DTriangles<<<" << blocks << ", " << 
                    threadsPerBlock << ">>> (" << subSpace.min.x << ", " << 
                    subSpace.min.y << ", " << subSpace.min.z << ")\n";

        process1DTriangles<Node, SNode>
            <<< blocks, 
                threadsPerBlock >>>( vertices_gpu, 
                                     indices_gpu, 
                                     triangleTypes_gpu,
                                     sortedTriangles_gpu, 
                                     materials_gpu, 
                                     nodes_gpu, 
                                     hostData.start1DTris,
                                     hostData.end1DTris,
                                     devData.left,
                                     devData.right,
                                     devData.up,
                                     devData.down,
                                     modelBBMin, 
                                     float(hostData.voxelLength),
                                     devData.extResolution,
                                     subSpace,
                                     gridType,
                                     countVoxels,
                                     devData.hashMap,
                                     surfNodes );

        cudaDeviceSynchronize();
    }

    checkCudaErrors( "process1DTriangles" );

    // Process and voxelize triangles with a 2-dimensional bounding box.
    if (hostData.start2DTris != hostData.end2DTris) 
    {
        if (verbose) 
            std::cout << (clock() - startTime) << 
                    ": Calling process2DTriangles<<<" << blocks << ", " << 
                    threadsPerBlock << ">>> (" << subSpace.min.x << ", " << 
                    subSpace.min.y << ", " << subSpace.min.z << ")\n";

        process2DTriangles<Node, SNode>
            <<< blocks, 
                threadsPerBlock >>>( vertices_gpu, 
                                     indices_gpu, 
                                     triangleTypes_gpu,
                                     sortedTriangles_gpu, 
                                     materials_gpu, 
                                     nodes_gpu, 
                                     hostData.start2DTris,
                                     hostData.end2DTris,
                                     devData.left,
                                     devData.right,
                                     devData.up,
                                     devData.down,
                                     modelBBMin, 
                                     float(hostData.voxelLength),
                                     devData.extResolution,
                                     subSpace,
                                     gridType,
                                     countVoxels,
                                     devData.hashMap,
                                     surfNodes );

        cudaDeviceSynchronize();
    }

    checkCudaErrors( "process2DTriangles" );

    // Process and voxelize triangles with a 3-dimensional bounding box.
    if (hostData.start3DTris != hostData.end3DTris) 
    {
        if (verbose) 
            std::cout << (clock() - startTime) << 
                    ": Calling process3DTriangles<<<" << blocks << ", " << 
                    threadsPerBlock << ">>> (" << subSpace.min.x << ", " << 
                    subSpace.min.y << ", " << subSpace.min.z << ")\n";

        process3DTriangles<Node, SNode>
            <<< blocks, 
                threadsPerBlock >>>( vertices_gpu, 
                                     indices_gpu, 
                                     triangleTypes_gpu,
                                     sortedTriangles_gpu, 
                                     materials_gpu, 
                                     nodes_gpu, 
                                     hostData.start3DTris,
                                     hostData.end3DTris,
                                     devData.left,
                                     devData.right,
                                     devData.up,
                                     devData.down,
                                     modelBBMin, 
                                     float(hostData.voxelLength),
                                     devData.extResolution,
                                     subSpace,
                                     gridType,
                                     countVoxels,
                                     devData.hashMap,
                                     surfNodes );

        cudaDeviceSynchronize();
    }

    checkCudaErrors( "calcOptSurfaceVoxelization" );
}
///////////////////////////////////////////////////////////////////////////////
/// Handles the creation of the \a padding by calling the \p zeroPadding() 
/// kernel.
/// 
/// \throws Exception if CUDA reports an error during execution.
///
/// \tparam Node Type of \p Node used.
///
/// \param[in,out] devData Reference to all relevant, device-specific 
///                           variables.
/// \param[in] xSlicing \p true if slicing along the x-axis, \p false in all 
///                     other cases.
/// \param[in] startTime Time when the program started executing.
/// \param[in] verbose Whether or not to print what is being done.
///////////////////////////////////////////////////////////////////////////////
template <class Node>
void makePaddingZero( CommonDevData const & devData,
                      Node                * nodes_gpu,
                      Node                * nodesCopy_gpu,
                      bool                  xSlicing,
                      clock_t               startTime,
                      bool                  verbose )
{
    int blocks = devData.blocks;
    int threads = devData.threads;

    if (verbose) 
        std::cout << (clock() - startTime) << ": Calling zeroPadding<<<" << 
                blocks << ", " << threads << ">>> on device " << 
                devData.dev << "\n";

    uint3 dimensions = devData.allocResolution.max -
                       devData.allocResolution.min;

    Node * nPtr = xSlicing ? nodesCopy_gpu : 
                             nodes_gpu;
    zeroPadding<<< blocks, threads >>>( nPtr, dimensions );

    checkCudaErrors( "makePaddingZero" );
}
///////////////////////////////////////////////////////////////////////////////
/// Handles the restoration of the rotated \p Nodes by calling the 
/// \p unRotateNodes() kernel.
/// 
/// \throws Exception if CUDA reports an error during execution.
///
/// \tparam Node Type of \p Node used.
///
/// \param[in,out] devData Reference to all relevant, device-specific 
///                           variables.
/// \param[in] yzSubSpace The bounding box for the subspace that is to be 
///                       voxelized.
/// \param[in] startTime Time when the program started executing.
/// \param[in] verbose Whether or not to print what is being done.
///////////////////////////////////////////////////////////////////////////////
template <class Node>
void restoreRotatedNodes( CommonDevData const & devData,
                          Node                * nodes_gpu,
                          Node                * nodesCopy_gpu,
                          Bounds<uint2> const & yzSubSpace,
                          clock_t               startTime,
                          bool                  verbose )
{
    uint blocks = devData.blocks;
    uint threadsPerBlock = devData.threads;

    if (verbose) 
        std::cout << (clock() - startTime) << ": Calling restoreRotatedNodes"
                     "<<<" << blocks << ", " << threadsPerBlock << ">>> (" << 
                     yzSubSpace.min.x << ", " << yzSubSpace.min.y << ")\n";
    
    unRotateNodes<Node>
        <<< blocks, threadsPerBlock >>>( nodes_gpu, 
                                         nodesCopy_gpu,
                                         devData.allocResolution,
                                         yzSubSpace );

    checkCudaErrors( "restoreRotatedNodes" );
}
///////////////////////////////////////////////////////////////////////////////
///
///////////////////////////////////////////////////////////////////////////////
template <class Node>
void populateHashMap
    ( CommonDevData       & devData
    , Node                * nodes_gpu
    , clock_t               startTime
    , bool                  verbose )
{
    if ( !Node::usesTwoArrays() )
        return;

    const uint blocks = 1;
    const dim3 threadsPerBlock(32, 32);

    if (verbose) 
        std::cout << (clock() - startTime) << ": Calling populateHashMap"
                     "<<<" << blocks << ", (32,32) >>>\n";

    uint3 dim = devData.allocResolution.max - devData.allocResolution.min;
    
    fillHashMap<Node>
        <<< blocks, threadsPerBlock >>>( nodes_gpu, 
                                         devData.hashMap,
                                         dim );

    checkCudaErrors( "populateHashMap" );
}

///////////////////////////////////////////////////////////////////////////////
/// The purpose of calculateTileOverlap() is to find out how many \a tiles each 
/// triangle overlaps. Knowing this makes it possible to calculate the size of 
/// the work queue, since each occurrence of a tile overlapping a triangle 
/// ends up as an entry (tile-triangle-pair) in the work queue. A tile is a 
/// square formed from 16 projected voxel centers on the yz-plane. When a tile 
/// overlaps a triangle, the yz-projection of the triangle overlaps the tile.
/// 
/// In order to find out the overlaps, the triangle has to be projected to the 
/// yz-plane and then tested for overlap. The overlap test requires calculating 
/// <em>edge functions</em> for each edge of the triangle. An edge function is 
/// a function that takes a point in (two dimensional) space and returns a 
/// positive value if the point is located on one side of the edge, a negative 
/// value if it is located on the opposite side, and zero if it is located 
/// exactly on the edge. It is possible to construct the edge functions in such 
/// a way that a positive result on all of them implies that the point is 
/// within the triangle formed by the three edges.
///
/// The overlap testing tries to evaluate the edge functions against each 
/// projected voxel center that the tile comprises of, until it succeeds on all 
/// three tests, confirming overlap between the tile and the triangle, or none 
/// of the voxel centers of the tile succeed three times, confirming no overlap 
/// between the tile and the triangle. A counter is increased each time an 
/// overlap is confirmend, and when the processing ends, this value is written 
/// to the tile overlap array using the index that corresponds to the triangle 
/// id.
///
/// In order to minimize the amount of tiles that need to be checked, the 
/// coordinates of the possibly overlapping tiles are determined from the 
/// triangle's bounding box, or more specifically, the yz-projected bounding 
/// box.
///////////////////////////////////////////////////////////////////////////////
__global__ void calculateTileOverlap
    (
    float const * vertices,      ///< [in] Vertices of the model.
    uint  const * indices,       ///< [in] Indices of the model.
    uint        * tileOverlaps,  ///< [out] Tile overlap buffer.
    uint          nrOfTriangles, ///< [in] Number of triangles in the model.
    double3       minVertex,     /**< [in] Minimum corner of the subspace's 
                                      bounding box. */
    double        voxelLength,   ///< [in] Distance between voxel centers.
    Bounds<uint3> resolution,    /**< [in] Bounding box of the voxelization 
                                           space. */
    Bounds<uint2> subSpace       /**< [in] Bounding box of the subspace, in 
                                      the yz-plane. */
    )
{
    // Loop until there are no more triangles to process.
    for( uint triangle = blockIdx.x * blockDim.x + threadIdx.x; 
         triangle < nrOfTriangles; 
         triangle += gridDim.x * blockDim.x )
    {
        // Vertices of the triangle.
        double3 verts[3];
        fetchTriangleVertices( vertices, indices, verts, triangle );

        // Calculating the bounding box.
        Bounds<double2> triBB;
        getTriangleBounds( verts, triBB );

        // Rename the distance between voxel centers for convenience.
        double d = voxelLength;

        // Transform bounding box to voxel coordinates.
        Bounds<double2> voxBB;
        getVoxelBoundsDouble( triBB, minVertex, voxBB, d );

        // Cast voxel coordinates to integers.
        Bounds<int2> voxels = { 
            make_int2( ceil( voxBB.min.x ), ceil( voxBB.min.y ) ), 
            make_int2( floor( voxBB.max.x ) + 1.0, floor( voxBB.max.y ) + 1.0 ) 
        };

        // X-component of the triangle's normal vector.
        double tN = ( verts[0].y * verts[1].z + verts[2].y * verts[0].z + 
                      verts[2].z * verts[1].y ) - ( verts[2].y * verts[1].z + 
                      verts[0].z * verts[1].y + verts[2].z * verts[0].y );

        // Shift the subSpaces to begin at the origin of resolution.
        Bounds<uint2> space = {
            make_uint2( subSpace.min.x - resolution.min.y,
                        subSpace.min.y - resolution.min.z ),
            make_uint2( subSpace.max.x - resolution.min.y, 
                        subSpace.max.y - resolution.min.z )
        };

        // Ignore degenerate or irrelevant triangles.
        if (tN == 0.0)
        {
            tileOverlaps[triangle] = 0;
            continue;
        }

        
        if ( voxels.min.x >= voxels.max.x || voxels.min.y >= voxels.max.y )
        {
            tileOverlaps[triangle] = 0;
            continue;
        }
        
        if ( ( voxels.max.x <= int(space.min.x) ) || 
             ( voxels.max.y <= int(space.min.y) ) )
        {
            tileOverlaps[triangle] = 0;
            continue;
        }
        
        if ( ( voxels.min.x >= int(space.max.x) ) || 
             ( voxels.min.y >= int(space.max.y) ) )
        {
            tileOverlaps[triangle] = 0;
            continue;
        }

        // Take the intersection between the triangle's bounding box and the 
        // bounding box of the subspace, if it is defined. Else, take the 
        // smaller bounding box.
        Bounds<uint2> commonSpace = { 
            make_uint2( max( make_int2(space.min), voxels.min ) ), 
            min( space.max, make_uint2(voxels.max) ) 
        };

        // Edgenormals & distances.
        double2 edgeNormals[3];
        double distances[3];

        if (tN > 0.0)
            for (int i = 0; i < 3; i++)
                edgeNormals[i] = make_double2( verts[i].z - verts[(i+1)%3].z, 
                                               verts[(i+1)%3].y - verts[i].y );
        else
            for (int i = 0; i < 3; i++)
                edgeNormals[i] = make_double2( verts[(i+1)%3].z - verts[i].z, 
                                               verts[i].y - verts[(i+1)%3].y );

        for (int i = 0; i < 3; i++)
            distances[i] = -( edgeNormals[i].x * verts[i].y + 
                              edgeNormals[i].y * verts[i].z );

        // Minimum and maximum tile coordinates by dividing voxel 
        // coordinates by 4.
        Bounds<uint2> tiles = { 
            make_uint2( commonSpace.min.x / 4, commonSpace.min.y / 4 ),
            make_uint2( ( commonSpace.max.x - 1 ) / 4, 
                        ( commonSpace.max.y - 1 ) / 4 ) 
        };

        // Loop over the relevant tiles, and the relevant voxels within those 
        // tiles. Once an intersection is found, record it and skip to the 
        // next tile.
        uint nrTilesOverlapped = 0;
        
        traverseTilesForOverlaps( edgeNormals, 
                                  distances, 
                                  nrTilesOverlapped, 
                                  tiles, 
                                  commonSpace, 
                                  minVertex, 
                                  d );

        // But what about the subRes splits? I think thy need some more 
        // thought.

        tileOverlaps[triangle] = nrTilesOverlapped;
    } // End for loop.
}
///////////////////////////////////////////////////////////////////////////////
/// Traverses and tests the given tiles for the \p calculateTileOverlap() 
/// function. It makes sure to keep within the specified voxel limits, since a 
/// tile can be cut in half by a subspace. Performs overlap testing and 
/// increments the overlap counter after each successful overlap. 
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
void traverseTilesForOverlaps
( 
    double2 const * edgeNormals,  ///< [in] Edge normals.
    double const * distances,     ///< [in] Distances to the edges.
    uint & nrOfTilesOverlapped,   /**< [out] Counter for the amount of 
                                       overlaps the triangle has. */
    Bounds<uint2> const & tiles,  ///< [in] The traversable tiles.
    Bounds<uint2> const & voxels, ///< [in] Voxel center restrictions.
    double3 const & minVertex,    /**< [in] Minimum corner of subspace's 
                                       bounding box. */
    double d                      ///< [in] Distance between voxel centers.
)
{
    bool skipTile = false;
    double2 p = make_double2(0.0, 0.0);
    bool testResult = true;

    for ( uint t = tiles.min.x; t <= tiles.max.x; t++ )
    {
        for ( uint u = tiles.min.y; u <= tiles.max.y; u++ )
        {
            skipTile = false;

            for ( uint i = t*4;
                  i < t*4 + 4 && i < voxels.max.x && !skipTile; 
                  i++ )
            {
                for ( uint j = u*4; 
                      j < u*4 + 4 && j < voxels.max.y && !skipTile; 
                      j++ )
                {
                    if ( i >= voxels.min.x && j >= voxels.min.y )
                    {
                        p.x = double(i) * d + minVertex.y;
                        p.y = double(j) * d + minVertex.z;

                        testResult = true;
                        for (int k = 0; k < 3; k++)
                            testResult &= ( edgeNormals[k].x * p.x + 
                                            edgeNormals[k].y * p.y + 
                                            distances[k] >= 0.0 );

                        if (testResult)
                        {
                            nrOfTilesOverlapped++;
                            skipTile = true;
                        }
                    }
                }
            }
        }
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Generates the contents of the work queue based on the results of the tile 
/// overlap tests and the offset buffer. The offset buffer contains the index 
/// into the work queue where the data (triangle-tile pairs) for each triangle 
/// begins.
///
/// The construction of the work queue proceeds almost identically to how the 
/// tile overlaps are calculated. The same overlap tests are performed, but 
/// this time, instead of counting the number of overlaps each triangle has, 
/// each overlapping tile is encoded into the work queue along with the 
/// triangle id.
///////////////////////////////////////////////////////////////////////////////
__global__ void constructWorkQueue
    ( 
    float const * vertices,    ///< [in] Vertices of the model.
    uint const * indices,      ///< [in] Indices of the model.
    uint * workQueueTriangles, ///< [out] The triangles of the work queue.
    uint * workQueueTiles,     ///< [out] The tiles of the work queue.
    uint const * offsetBuffer, /**< [in] The offsets to where each triangle's 
                                         data begins in the work queue. */
    uint nrOfTriangles,        ///< [in] Number of triangles in the model.
    double3	minVertex,         /**< [in] Minimum corner of the subspace's 
                                         bounding box. */
    int firstIndex,            ///< [in] The first triangle that has overlaps.
    double voxelLength,        ///< [in] Distance between voxel centers.
    Bounds<uint3> resolution,  /**< [in] Bounding box of the voxelization 
                                         space. */
    Bounds<uint2> subSpace     ///< [in] Bounding box of the subspace.
    )
{
    // Loop until there are no more triangles to process.
    for ( uint triangle = blockIdx.x * blockDim.x + threadIdx.x; 
          triangle < nrOfTriangles; 
          triangle += gridDim.x * blockDim.x )
    {
        if (triangle < firstIndex)
            continue;

        uint offset = offsetBuffer[triangle];

        // Don't process triangles that don't have any previously detected 
        // overlap.
        if ( (triangle != firstIndex) && (offset == 0) )
            continue;

        // Vertices of the triangle.
        double3 verts[3];
        fetchTriangleVertices( vertices, indices, verts, triangle );

        // Bounding box of the triangle.
        Bounds<double2> triBB;
        getTriangleBounds( verts, triBB );

        // Rename the distance between voxel centers for convenience.
        double d = voxelLength;

        // Transform bounding box to voxel coordinates.
        Bounds<double2> voxBB;
        getVoxelBoundsDouble( triBB, minVertex, voxBB, d );

        // Cast voxel coordinates to integers.
        
        Bounds<int2> voxels = { 
            make_int2( ceil( voxBB.min.x ), ceil( voxBB.min.y ) ), 
            make_int2( floor( voxBB.max.x ) + 1.0, floor( voxBB.max.y ) + 1.0 ) 
        };
        
        // Shift the subSpaces to begin at the origin of resolution.
        Bounds<uint2> space = {
            make_uint2( subSpace.min.x - resolution.min.y,
                        subSpace.min.y - resolution.min.z ),
            make_uint2( subSpace.max.x - resolution.min.y, 
                        subSpace.max.y - resolution.min.z )
        };

        // Take the intersection between the triangle's bounding box and the 
        // bounding box of the subspace, if it is defined. Else, take the 
        // smaller bounding box.
        Bounds<uint2> commonSpace = { 
            make_uint2( max( make_int2(space.min), voxels.min ) ), 
            min( space.max, make_uint2(voxels.max) ) 
        };

        // x-component of the triangle's normal.
        double tN = ( verts[0].y * verts[1].z + verts[2].y * 
                      verts[0].z + verts[2].z * verts[1].y ) - 
                    ( verts[2].y * verts[1].z + verts[0].z * 
                      verts[1].y + verts[2].z * verts[0].y );

        // Edgenormals & distances.
        double2 edgeNormals[3];
        double distances[3];

        if (tN > 0.0)
            for (int i = 0; i < 3; i++)
                edgeNormals[i] = make_double2( verts[i].z - verts[(i+1)%3].z, 
                                               verts[(i+1)%3].y - verts[i].y );
        else
            for (int i = 0; i < 3; i++)
                edgeNormals[i] = make_double2( verts[(i+1)%3].z - verts[i].z, 
                                               verts[i].y - verts[(i+1)%3].y );

        for (int i = 0; i < 3; i++)
            distances[i] = -( edgeNormals[i].x * verts[i].y + 
                              edgeNormals[i].y * verts[i].z );

        // Minimum and maximum tile coordinates by 
        // dividing voxel coordinates by 4.
        Bounds<uint2> tiles = { 
            make_uint2( commonSpace.min.x / 4, commonSpace.min.y / 4 ),
            make_uint2( ( commonSpace.max.x - 1 ) / 4, 
                        ( commonSpace.max.y - 1 ) / 4 ) 
        };

        // Go through the relevant tiles and relevant voxels within each tile, 
        // and record what tiles the intersections happen in. Once an 
        // intersection is found, skip to the next tile.
        traverseTilesForWorkQueue( edgeNormals, 
                                   distances, 
                                   workQueueTriangles, 
                                   workQueueTiles, 
                                   offset, 
                                   triangle, 
                                   tiles, 
                                   commonSpace, 
                                   resolution.max - resolution.min,
                                   minVertex, 
                                   d );
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Traverses the given tiles, performs overlap tests and writes to the work 
/// queue.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
void traverseTilesForWorkQueue
    ( 
    double2 const * edgeNormals,  ///< [in] The edge normals.
    double const * distances,     ///< [in] Distances to the edges.
    uint * workQueueTriangles,    ///< [out] Triangles of the work queue.
    uint * workQueueTiles,        ///< [out] Tiles of the work queue.
    uint offset,                  /**< [in] Offset into the work queue 
                                            where the current triangle's 
                                            data begins. */
    uint triangle,                ///< [in] Current triangle's index.
    Bounds<uint2> const & tiles,  ///< [in] Tile space to traverse.
    Bounds<uint2> const & voxels, /**< [in] Limitations on the voxel 
                                            coordinates. */
    uint3 const & resolution,     ///< [in] Bounds of the voxel space.
    double3 const & minVertex,    /**< [in] Minimum corner of the voxel 
                                            space's bounding box. */
    double d                      ///< [in] Distance between voxels.
    )
{
    bool skipTile = false;
    double2 p = make_double2(0.0, 0.0);
    bool testResult = true;

    uint currentIntersection = 0;
    for ( uint t = tiles.min.x; t <= tiles.max.x; t++ )
    {
        for ( uint u = tiles.min.y; u <= tiles.max.y; u++ )
        {
            skipTile = false;

            for ( uint i = t*4; 
                  i < t*4 + 4 && i < voxels.max.x && !skipTile; 
                  i++ )
            {
                for ( uint j = u*4; 
                      j < u*4 + 4 && j < voxels.max.y && !skipTile; 
                      j++ )
                {
                    if ( i >= voxels.min.x && j >= voxels.min.y )
                    {
                        p.x = double(i) * d + minVertex.y;
                        p.y = double(j) * d + minVertex.z;

                        testResult = true;
                        for (int k = 0; k < 3; k++)
                            testResult &= ( edgeNormals[k].x * p.x + 
                                            edgeNormals[k].y * p.y + 
                                            distances[k] >= 0.0 );

                        if (testResult)
                        {
                            workQueueTriangles[offset + currentIntersection] = 
                                triangle;
                            workQueueTiles[offset + currentIntersection] = 
                                u * ((resolution.y + 3) / 4) + t;
                            currentIntersection++;
                            skipTile = true;
                        }
                    }
                }
            }
        }
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Generates the plain voxelization into an array of unsigned integers. 
/// The algorithm more or less performs the same steps as in the tile overlap 
/// and work queue calculations. A triangle is fetched, its edge normals and 
/// related information to the testing is calculated, and the overlap tests are 
/// performed. Only now the different threads are picking a tile to process 
/// from the \p tileList, check its offset into the work queue from the \p 
/// tileOffsets, then start processing the triangles that are paired with the 
/// particular tile that is being processed.
///
/// Unlike the tile overlap and work queue calculations, where each thread was 
/// processing one triangle, now each thread is processing one voxel center of 
/// a tile. Several triangles can be processed simultaneously, as well. The 
/// overlap tests happen between the triangle and the voxel centers, and in the 
/// case of an overlap, a ray is shot from the voxel center on the yz-plane and 
/// the exact intersection with the triangle's plane is determined. This 
/// intersection location is then translated to a voxel index along the x-axis 
/// and the corresponding voxel is considered as intersecting with the 
/// triangle.
///
/// Intersections are kept track of in two ways:
///   -# An entire 32-bit <tt>unisgned int</tt> can be kept in memory and 
///      written to when possible in order to reduce the amount of accesses to 
///      <em>global memory</em>. This is called an <em>active segment</em>.
///   -# A single 32-bit <tt>unsigned int</tt> per voxel center is updated 
///      throughout the processing of a single tile, where each bit represents 
///      a full integer from <em>global memory</em>. This is called the \a 
///      bitmask.
/// Whenever an intersection occurs, all voxels along the x-axis after and 
/// including the intersecting voxel need to be \a flipped, i.e. solid voxels 
/// become non-solid and vice versa. After all intersections along the same 
/// (y,z)-coordinate have been processed in this way, only the voxels inside 
/// the volume enclosed by the model will be solid.
///
/// \p XOR-operations can be used to quickly flip bits in integers and the 
/// bitmask can keep track of whole integers whose bits all need to be flipped 
/// due to an intersection that occurred in an earlier integer. After the 
/// whole tile is processed, the bitmask and any open active segments can be 
/// processed and the global memory updated.
///////////////////////////////////////////////////////////////////////////////
__global__ void generateVoxelization
    ( 
    float const * vertices,          ///< [in] The vertices of the model.
    uint const * indices,            ///< [in] The indices of the model.
    uint const * workQueueTriangles, ///< [in] Triangles of the work queue.
    uint const * workQueueTiles,     ///< [in] Tiles of the work queue.
    uint const * tileList,           /**< [in] Ordered unique tiles of the 
                                               work queue. */
    uint const * tileOffsets,        /**< [in] Offset into the work queue 
                                               where a particular tile from 
                                               the tileList begins. */
    VoxInt * voxels,                 /**< [out] The voxels as an array of 
                                                integers. */
    uint nrOfTriangles,              /**< [in] The number of triangles in the 
                                               model. */
    uint workQueueSize,              /**< [in] Number of elements in the work 
                                               queue. */
    double3 minVertex,               /**< [in] Minimum corner of the 
                                          voxelization space's bounding 
                                          box. */
    double voxelLength,              ///< [in] Distance between voxel centers.
    bool left,                       /**< [in] Is another device voxelizing 
                                          on the left? */
    bool right,                      /**< [in] Is another device voxelizing 
                                          on the right? */
    bool up,                         /**< [in] Is another device voxelizing 
                                               above? */
    bool down,                       /**< [in] Is another device voxelizing 
                                               below? */
    Bounds<uint3> resolution,        /**< [in] Bounds of the voxelization 
                                               space. */
    Bounds<uint3> subSpace           ///< [in] Bounds of the subspace.
    )
{
    uint tileId = blockIdx.x;
    uint threadId = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ bool finalTriangleReached[4];
    __shared__ uint currentTriangle;

    __shared__ double3 triangles[12];       // Current triangle indices.
    __shared__ double2 edgeNormals[12];     // Edge normals.
    __shared__ double distances[12];        // Distances to edges.
    __shared__ double3 triangleNormals[4];  // Normals of the triangles.
    __shared__ VoxInt activeSegments[64];   // Local copy of some voxel data.
    __shared__ VoxInt bitmasks[64];         /* Integers that need to be 
                                               flipped. */
    __shared__ uint currentSegments[64];    // Addresses of active segments.

    // Initialization.
    if (threadIdx.x == 0)
        finalTriangleReached[threadIdx.y] = false;
    if (threadIdx.x == 1)
        currentTriangle = 0;
    activeSegments[threadId] = 0;
    bitmasks[threadId] = 0;
    currentSegments[threadId] = UINT_MAX;

    /* Shared triangle setup. Four triangles can be processed at once, 
       provided the current tile has that many overlapping triangles. 
       Processing ends when all triangle paths have finished. */
    while ( (!finalTriangleReached[0]) || (!finalTriangleReached[1]) || 
            (!finalTriangleReached[2]) || (!finalTriangleReached[3]) )
    {
        __syncthreads();

        if (threadIdx.x < 3) //3
        {
            // Make sure that the triangles exist for this tile.
            uint workQueueIndex = tileOffsets[tileId] + 
                                  blockDim.y * currentTriangle + threadIdx.y;
            if (workQueueIndex < workQueueSize)
            {
                uint triangleIndex = workQueueTriangles[workQueueIndex];
                if (workQueueTiles[workQueueIndex] == tileList[tileId])
                {
                    uint t = 3*threadIdx.y;

                    triangles[t + threadIdx.x] = make_double3(
                        vertices[3*indices[3*triangleIndex + threadIdx.x]],
                        vertices[3*indices[3*triangleIndex + threadIdx.x] + 1],
                        vertices[3*indices[3*triangleIndex + threadIdx.x] + 2]
                    );
                }
                else
                {
                    finalTriangleReached[threadIdx.y] = true;
                }
            }
            else
            {
                finalTriangleReached[threadIdx.y] = true;
            }
        }
        
        __syncthreads();

        // Triangle's normal.
        if ((threadIdx.x == 0) && (!finalTriangleReached[threadIdx.y]))
        {
            uint t = 3*threadIdx.y;

            double3 U = triangles[t] - triangles[t + 2];
            double3 V = triangles[t + 1] - triangles[t];

            triangleNormals[threadIdx.y] = cross(U, V);
        }

        __syncthreads();

        // Edge normals & distances.
        if ((threadIdx.x < 3) && (!finalTriangleReached[threadIdx.y]))
        {
            uint i = threadIdx.x;
            uint t = 3*threadIdx.y;
            
            if (triangleNormals[threadIdx.y].x > 0.0)
                edgeNormals[t + i] = make_double2( 
                    triangles[t + i].z - triangles[t + ((i + 1) % 3)].z, 
                    triangles[t + ((i + 1) % 3)].y - triangles[t + i].y
                );
            else
                edgeNormals[t + i] = make_double2( 
                    triangles[t + ((i + 1) % 3)].z - triangles[t + i].z, 
                    triangles[t + i].y - triangles[t + ((i + 1) % 3)].y
                );

            distances[t + i] = - (edgeNormals[t + i].x * triangles[t + i].y + 
                                  edgeNormals[t + i].y * triangles[t + i].z );
        }

        __syncthreads();

        // Intersection test & memory writes.
        if (!finalTriangleReached[threadIdx.y])
        {
            uint3 res = resolution.max - resolution.min;
            uint tileNr = tileList[tileId];
            uint tileZ = tileNr / ((res.y + 3) / 4); // Decode tile id.
            uint tileY = tileNr % ((res.y + 3) / 4); // Decode tile id.
            uint voxelZ = (threadIdx.x / 4) + (tileZ * 4);
            uint voxelY = (threadIdx.x % 4) + (tileY * 4);

            // Move minimum corner of the subspace to the origin.
            Bounds<uint2> space = {
                make_uint2( subSpace.min.y - resolution.min.y,
                            subSpace.min.z - resolution.min.z ),
                make_uint2( subSpace.max.y - resolution.min.y,
                            subSpace.max.z - resolution.min.z )
            };

            // Don't process voxels outside the subspace.
            if (voxelY >= space.min.x && voxelY < space.max.x && 
                voxelZ >= space.min.y && voxelZ < space.max.y )
            {
                double d = voxelLength;
                double2 p = make_double2( double(voxelY) * d + minVertex.y, 
                    double(voxelZ) * d + minVertex.z );

                uint testSuccesses = 0;
                double testResult;

                for (int i = 0; i < 3; i++)
                {
                    uint j = 3*threadIdx.y + i;

                    testResult = edgeNormals[j].x * p.x + 
                        edgeNormals[j].y * p.y + distances[j];
                    if ( edgeNormals[j].x > 0.0 || edgeNormals[j].x == 0.0 && 
                         edgeNormals[j].y < 0.0 )
                        testResult += DBL_MIN;

                    if (testResult > 0.0)
                        testSuccesses++;
                }

                if (testSuccesses == 3)
                {
                    // Calculate intersection with the triangle's plane.

                    double3 v = triangles[3*threadIdx.y];
                    double3 n = triangleNormals[threadIdx.y];

                    double3 A = v - make_double3( minVertex.x, p.x, p.y );
                    double B = A.x * n.x + A.y * n.y + A.z * n.z;
                    double px = B / n.x;

                    px = px < 0.0 ? 0.0 : px;

                    uint voxelX = uint(ceil(px / d));

                    /* Account for the zero-padding at the beginning of the 
                       voxelization, if padding is enabled. If there are no 
                       additional devices then each coordinate needs to be 
                       increased by one, otherwise the voxel data would be 
                       written to the voxels designated as padding and would 
                       later be zeroed out.

                       If there are other devices involved, then at least one 
                       of the boolean parameters is true. Then, there is a 
                       point to voxelizing onto the area designated as padding 
                       since that data will be used for neighborhood analysis 
                       during the boundary id calculations -- there needs to 
                       be overlap between two separate voxelization spaces, 
                       otherwise the boundary ids will be wrong at the border.

                       In this case, the coordinates are not increased and 
                       the unnecessary voxels will be zeroed out as padding 
                       after the boundary ids have been calculated. 
                       
                       The increase to the resolutions is simply there to make 
                       the resolution equal to the allocated size. The amount 
                       to add along each axis depends on the boolean values. */

                    voxelX += 1;
                    if (!left) {
                        voxelY += 1;
                        res.y += 1;
                    }

                    if (!right) 
                        res.y += 1;

                    if (!up) 
                        res.z += 1;

                    if (!down) {
                        voxelZ += 1;
                        res.z += 1;
                    }

                    /* Find out in what segment the intersection is. When 
                       the voxelization is divided along the x-axis the same 
                       intersections need to be handled multiple times -- 
                       once for each split along the x-axis. If an intersection 
                       happens in a split after the current one, then it can 
                       be completely ignored. If it happens in a previous 
                       split, then the entire bitmask needs to be inverted. */
                    uint ix = voxelX >> VOX_DIV;
                    uint min_ix = ( subSpace.min.x + 2 ) >> VOX_DIV;

                    if (ix >= ( ( subSpace.max.x + 2 ) >> VOX_DIV))
                    {
                        // Ignore the intersection.
                    }
                    else if (ix < min_ix)
                    {
                        // Ignore the intersection, but flip all bits of the 
                        // bitmasks.
                        uint inti = 
                            res.y * (res.x >> VOX_DIV) * voxelZ + 
                            (res.x >> VOX_DIV) * voxelY + min_ix;

                        /* If the wanted integer is not in memory as an 
                           activeSegment. */
                        if (currentSegments[threadId] != inti)
                        {
                            /* XOR the current active segment into global 
                               memory. */
                            if (activeSegments[threadId] != 0)
                                atomicXor( &voxels[currentSegments[threadId]], 
                                activeSegments[threadId] );

                            // Init new active segment.
                            activeSegments[threadId] = 0;
                            currentSegments[threadId] = inti;
                        }

                        bitmasks[threadId] ^= VOX_MAX;
                    }
                    else
                    {
                        // Proceed normally.
                        uint inti = 
                            res.y * (res.x >> VOX_DIV) * voxelZ + 
                            (res.x >> VOX_DIV) * voxelY + ix;

                        /* If the wanted integer is not in memory as an 
                           activeSegment. */
                        if (currentSegments[threadId] != inti)
                        {
                            /* XOR the current active segment into global 
                               memory. */
                            if (activeSegments[threadId] != 0)
                                atomicXor( &voxels[currentSegments[threadId]], 
                                activeSegments[threadId] );

                            // Init new active segment.
                            activeSegments[threadId] = 0;
                            currentSegments[threadId] = inti;
                        }
                        else
                        {
                            /* Already have the segment in memory. No need to 
                               load anything. */
                        }

                        // XOR or create the segment accordingly.

                        uint intersectionLocation = voxelX % VOX_BPI;
                        VoxInt segmentBitmask = 
                            VOX_MAX >> intersectionLocation;
                        activeSegments[threadId] ^= segmentBitmask;

                        // XOR or create the bitmasks accordingly.

                        uint lastIX = (subSpace.max.x+1) >> VOX_DIV;
                        VoxInt bitmask = ix < lastIX ? VOX_MAX >> 
                            ( ix % ( ( subSpace.max.x - subSpace.min.x ) >> 
                            VOX_DIV ) + 1 ) : 0;
                        bitmasks[threadId] ^= bitmask;

                    }
                }
            }
        }

        if (threadId == 0)
        {
            currentTriangle++;
        }
    }

    __syncthreads();

    // Flush the segments and bitmaps to global memory.

    if (currentSegments[threadId] == UINT_MAX)
        return;

    if (activeSegments[threadId] != 0)
        atomicXor( &voxels[currentSegments[threadId]], 
                   activeSegments[threadId] );
    
    uint intsPerDim = ( subSpace.max.x - subSpace.min.x ) >> VOX_DIV;
    uint startIndex = currentSegments[threadId] - 
                      ( currentSegments[threadId] % intsPerDim );
    for (int i = 0; i < intsPerDim; i++)
    {
        VoxInt value = ( bitmasks[threadId] & 
            ( 1u << ( VOX_BPI - 1 - i ) ) ) > 0 ? VOX_MAX : 0;
        if (value != 0)
            atomicXor( &voxels[startIndex + i], value );
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Traverses through the array of integers and reads each integer bit for bit 
/// and translates each bit to a \p Node in the node array. Supports the 
/// processing of voxels in smaller spaces than the complete array of voxels by 
/// dividing the space into \a subspaces. The conversion of \p Nodes is a very 
/// light operation, so it shouldn't really be necessary to split the 
/// processing like that, in most cases anyway.
///////////////////////////////////////////////////////////////////////////////
template <class Node>
__global__ void constructNodeList2
    ( 
    VoxInt const * voxels,    ///< [in] Integer array of voxels.
    Node * nodes,             ///< [out] Empty node array.
    Bounds<uint3> resolution, /**< [in] Bounding box of the total node space 
                                        for this device. */
    Bounds<uint2> yzSubSpace  ///< [in] Bounding box of the current subspace.
    )
{
    uint3 res = resolution.max - resolution.min;
    uint bit = threadIdx.y;
    uint ipx = res.x >> VOX_DIV;
    uint ipy = res.y * ipx;
    uint nrOfInts = res.z * ipy;

    Bounds<uint2> space = {
        yzSubSpace.min - make_uint2( resolution.min.y, resolution.min.z ),
        yzSubSpace.max - make_uint2( resolution.min.y, resolution.min.z )
    };

    uint startInt = ipy * space.min.y + ipx * space.min.x;
    uint endInt = ipy * space.max.y;

    uint nodeIdx;
    int x, y, z;
    Node node;
    VoxInt currentInt;

    for (uint i = startInt + blockIdx.x * blockDim.x + threadIdx.x; 
         i < nrOfInts && i < endInt; 
         i += gridDim.x * blockDim.x )
    {
        currentInt = voxels[i];

        node = Node();

        y = (i % ipy) / ipx;

        if ( y < space.min.x || y >= space.max.x )
            continue;

        x = i % ipx;
        z = i / ipy;

        if ( z < space.min.y || z >= space.max.y )
            continue;

        node.bid(uchar((currentInt >> (VOX_BPI - 1 - bit)) & 1));
        node.r(float(node.bid() > 0));

        nodeIdx = res.x * res.y * z + res.x * y + VOX_BPI * x + bit;

        if ( Node::usesTwoArrays() )
        {
            if ( node.bid() != 0 ) nodes[nodeIdx] = node;
        }
        else
            nodes[nodeIdx] = node;
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Traverses through the array of integers and reads each integer bit for bit 
/// and translates the voxels to an array of \p Nodes. In order to construct a 
/// full fcc grid, four slightly shifted, separate voxelizations are required.
/// Which voxelization is being used is given by the \p gridType parameter.
/// 
/// Given a voxelization with a point spacing (distance between sample points 
/// along some axis) \a a, the four different voxelizations are defined as 
/// follows:
/// 1. Base voxelization shifted by \f$ (0, 0, 0) \f$.
/// 2. Base voxelization shifted by \f$ (\frac{a}{2}, \frac{a}{2}, 0) \f$.
/// 3. Base voxelization shifted by \f$ (\frac{a}{2}, 0, \frac{a}{2}) \f$.
/// 4. Base voxelization shifted by \f$ (0, \frac{a}{2}, \frac{a}{2}) \f$.
///
/// The resulting FCC grid will fulfill the following indexing scheme:
///
/// \f[ G_{n} = \left\{ \mathbf{\vec{x}_{m,a}} = \left[ \mathbf{\vec{m}} + 
/// m_{y}\mathbf{\hat{e}_{y}} + \left( \left( m_{x} + m_{z} \pmod 2 \right)
/// \mathbf{\hat{e}_{y}} \right) \right] \cdot \frac{a}{2} \right\}, \f]
///
/// where \f$ \mathbf{\vec{x}_{m,a}} \f$ is the position in space of a point on 
/// the fcc lattice, \f$ \mathbf{\vec{m}} = m_{x}\mathbf{\hat{e}_{x}} + 
/// m_{y}\mathbf{\hat{e}_{y}} + m_{z}\mathbf{\hat{e}_{z}} \f$ are the lattice 
/// coordinates and \f$ a \f$ is the side length of the cubic cell, as well as 
/// the distance between two sample points in the individual voxelizations, as 
/// mentioned earlier. The shortest distance between points in the lattice is 
/// \f$ \frac{a}{\sqrt{2}} \f$.
///////////////////////////////////////////////////////////////////////////////
template <class Node>
__global__ void convertToFCCGrid
    ( 
    VoxInt const * voxels,    ///< [in] Integer array of voxels.
    Node * nodes,             ///< [out] Empty node array.
    int gridType,             /**< [in] Which of the 4 grids we are processing.
                                        Accepted values are 1, 2, 3 or 4. */
    Bounds<uint3> resolution, /**< [in] Bounding box of the total voxel space 
                                        for this device. */
    Bounds<uint2> yzSubSpace  ///< [in] Bounding box of the current subspace.
    )
{
    uint3 res = resolution.max - resolution.min;
    uint bit = threadIdx.y;
    uint ipx = res.x >> VOX_DIV;
    uint ipy = res.y * ipx;
    uint nrOfInts = res.z * ipy;

    Bounds<uint2> space = {
        yzSubSpace.min - make_uint2( resolution.min.y, resolution.min.z ),
        yzSubSpace.max - make_uint2( resolution.min.y, resolution.min.z )
    };

    uint startInt = ipy * space.min.y + ipx * space.min.x;
    uint endInt = ipy * space.max.y;

    for (uint i = startInt + blockIdx.x * blockDim.x + threadIdx.x; 
         i < nrOfInts && i < endInt; 
         i += gridDim.x * blockDim.x )
    {
        VoxInt currentInt = voxels[i];

        int y = (i % ipy) / ipx; // Y coordinate of the voxel and node.

        if ( y < space.min.x || y >= space.max.x )
            continue;

        int x = (i % ipx) * VOX_BPI + bit; // X coordinate of the voxel only.
        int z = i / ipy; // Z coordinate of the voxel and node.

        if ( z < space.min.y || z >= space.max.y )
            continue;

        // Put the generated node in a different location depending on which 
        // of the four voxel grids is currently being processed.
        uint nodeIdx = 0;
        switch ( gridType )
        {
        case 1:
            nodeIdx = 2*res.x * res.y * 2*z + 2*res.x * y + 2*x;
            break;
        case 2:
            nodeIdx = 2*res.x * res.y * 2*z + 2*res.x * y + 2*x + 1;
            break;
        case 3:
            nodeIdx = 2*res.x * res.y * (2*z + 1) + 2*res.x * y + 2*x;
            break;
        case 4:
            nodeIdx = 2*res.x * res.y * (2*z + 1) + 2*res.x * y + 2*x + 1;
            break;
        default:
            break;
        };

        nodes[ nodeIdx ] = Node( (currentInt >> (VOX_BPI - 1 - bit)) & 1u );
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Traverses each \p Node in the array and calculates a <em>boundary id</em> 
/// that depends on the boundary ids (specifically if they are non-zero or not) 
/// of its neighboring \p Nodes. The nodes are also validated in the sense that 
/// if a configuration of \p Node + neighbors wouldn't create a well-defined 
/// volume, then the node's <em>boundary id</em> will be set to zero. This 
/// requires the re-running of the kernel to take into account the new non-
/// solid node. This is achieved by setting an <tt>error bool</tt> to \p true 
/// whenever a \p Node is set to zero. The calling thread on the host side will 
/// then know to re-run this kernel. As with most kernels, this one also 
/// supports the division of space into multiple \a subspaces to lessen the 
/// computational burden caused by one invocation of the kernel. 
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
__global__ void fillNodeList2
    ( 
    Node * nodes,             ///< [in,out] Array of Nodes.
    Bounds<uint3> resolution, ///< [in] Total bounds of this device's space.
    Bounds<uint2> yzSubSpace, ///< [in] Bounds of the subspace.
    bool * error,             /**< [out] Boolean that signals the need to run 
                                   the kernel again. */
    HashMap hashMap,
    SNode * surfNodes
    )
{
    uint3 res = resolution.max - resolution.min;
    uint npx = res.x;
    uint npy = npx * res.y;

    Bounds<uint2> space = {
        yzSubSpace.min - make_uint2( resolution.min.y, resolution.min.z ),
        yzSubSpace.max - make_uint2( resolution.min.y, resolution.min.z )
    };

    uint startNode = npy * space.min.y + npx * space.min.x;
    uint endNode = npy * space.max.y;

    // Lookup table that provides a boundary id for each permutation.
    __shared__ uchar orientationLookup[64];

    if (threadIdx.x < 64)
        orientationLookup[threadIdx.x] = getOrientation(threadIdx.x);

    __syncthreads();

    uint nrOfNodes = npy * res.z;
    for ( uint n = startNode + blockIdx.x * blockDim.x + threadIdx.x; 
          n < nrOfNodes && n < endNode; 
          n += gridDim.x * blockDim.x )
    {
        Node node = nodes[n];

        // Don't process nodes that are non-solid.
        if (node.bid() == 0)
            continue;

        uint y = (n % npy) / npx;

        if ( y < space.min.x || y >= space.max.x )
            continue;

        uint x = n % npx;
        uint z = n / npy;

        if ( z < space.min.y || z >= space.max.y )
            continue;

        uchar permutation = 0;

        if (x > 0)
        {
            // Neighor closer to origin along x is non-solid.
            if (nodes[n - 1].bid() > 0)
            {
                permutation += 1;
            }
        }

        if (x < res.x - 1)
        {
            // Neighor further from the origin along x is non-solid.
            if (nodes[n + 1].bid() > 0)
            {
                permutation += 2;
            }
        }

        if (y > 0)
        {
            // Neighor closer to the origin along y is non-solid.
            if (nodes[n - npx].bid() > 0)
            {
                permutation += 4;
            }
        }

        if (y < res.y - 1)
        {
            // Neighor further from the origin along y is non-solid.
            if (nodes[n + npx].bid() > 0)
            {
                permutation += 8;
            }
        }

        if (z > 0)
        {
            // Neighor closer to the origin along z is non-solid.
            if (nodes[n - npy].bid() > 0)
            {
                permutation += 16;
            }
        }

        if (z < res.z - 1)
        {
            // Neighor further from the origin along z is non-solid.
            if (nodes[n + npy].bid() > 0)
            {
                permutation += 32;
            }
        }

        uchar newBid = orientationLookup[permutation];

        node.bid(newBid);
        
        if (newBid == 0)
        {
            *error = true;
        }

        if ( Node::usesTwoArrays() )
        {
            uint surfNodeIdx = hashMap.get( n );
            if ( surfNodeIdx != UINT32_MAX )
            {
                SNode surfNode = surfNodes[surfNodeIdx];
                surfNode.orientation = newBid;
                surfNodes[surfNodeIdx] = surfNode;
            }
        }

        nodes[n] = node;
    }
}
///////////////////////////////////////////////////////////////////////////////
/// ???
///
/// Traverses each \p Node in the array and calculates a <em>neighbor id</em> 
/// that depends on the boundary ids (specifically if they are non-zero or not) 
/// of its neighboring \p Nodes. The nodes are also validated in the sense that 
/// if a configuration of \p Node + neighbors wouldn't create a well-defined 
/// volume, then the node's <em>boundary id</em> will be set to zero. This 
/// requires the re-running of the kernel to take into account the new non-
/// solid node. This is achieved by setting an <tt>error bool</tt> to \p true 
/// whenever a \p Node is set to zero. The calling thread on the host side will 
/// then know to re-run this kernel. As with most kernels, this one also 
/// supports the division of space into multiple \a subspaces to lessen the 
/// computational burden caused by one invocation of the kernel. 
///////////////////////////////////////////////////////////////////////////////
template <class Node>
__global__ void calculateFCCBoundaries
    ( 
    Node * nodes,             ///< [in,out] Array of Nodes.
    Bounds<uint3> resolution, ///< [in] Total bounds of this device's space.
    Bounds<uint2> yzSubSpace  ///< [in] Bounds of the subspace.
    )
{
    const uint3 res = resolution.max - resolution.min;
    const uint npx = res.x;
    const uint npy = npx * res.y;

    const Bounds<uint2> space = {
        yzSubSpace.min - make_uint2( resolution.min.y, resolution.min.z ),
        yzSubSpace.max - make_uint2( resolution.min.y, resolution.min.z )
    };

    const uint startNode = npy * space.min.y + npx * space.min.x;
    const uint endNode = npy * space.max.y;

    const uint nrOfNodes = npy * res.z;
    for ( uint n = startNode + blockIdx.x * blockDim.x + threadIdx.x; 
          n < nrOfNodes && n < endNode; 
          n += gridDim.x * blockDim.x )
    {
        Node node = nodes[n];

        // Don't process nodes that are non-solid.
        if (node.bid() == 0)
            continue;

        const uint y = (n % npy) / npx;

        if ( y < space.min.x || y >= space.max.x )
            continue;

        const uint x = n % npx;
        const uint z = n / npy;

        if ( z < space.min.y || z >= space.max.y )
            continue;

        ushort newBid = 0;

        const bool left = x > 0;
        const bool right = x < res.x - 1;
        const bool in = y > 0;
        const bool out = y < res.y - 1;
        const bool down = z > 0;
        const bool up = z < res.z - 1;
        const bool xEven = (x + z) % 2 == 0;

        if ( down && ( xEven && in || !xEven ) ) // Neighbor 00
        {
            if ( nodes[ n - npy - xEven * npx ].bid() > 0 )
                newBid += 1;
        }
        if ( down && left ) // Neighbor 01
        {
            if ( nodes[ n - npy - 1 ].bid() > 0 )
                newBid += 2;
        }
        if ( down && right ) // Neighbor 02
        {
            if ( nodes[ n - npy + 1 ].bid() > 0 )
                newBid += 4;
        }
        if ( down && ( !xEven && out || xEven) ) // Neighbor 03
        {
            if ( nodes[ n - npy + !xEven * npx ].bid() > 0 )
                newBid += 8;
        }

        if ( left && ( xEven && in || !xEven ) ) // Neighbor 04
        {
            if ( nodes[ n - 1 - xEven * npx ].bid() > 0 )
                newBid += 16;
        }
        if ( right && ( xEven && in || !xEven ) ) // Neighbor 05
        {
            if ( nodes[ n + 1 - xEven * npx ].bid() > 0 )
                newBid += 32;
        }
        if ( left && ( !xEven && out || xEven ) ) // Neighbor 06
        {
            if ( nodes[ n - 1 + !xEven * npx ].bid() > 0 )
                newBid += 64;
        }
        if ( right && ( !xEven && out || xEven ) ) // Neighbor 07
        {
            if ( nodes[ n + 1 + !xEven * npx ].bid() > 0 )
                newBid += 128;
        }

        if ( up && ( xEven && in || !xEven ) ) // Neighbor 08
        {
            if ( nodes[ n + npy - xEven * npx ].bid() > 0 )
                newBid += 256;
        }
        if ( up && left ) // Neighbor 09
        {
            if ( nodes[ n + npy - 1 ].bid() > 0 )
                newBid += 512;
        }
        if ( up && right ) // Neighbor 10
        {
            if ( nodes[ n + npy + 1 ].bid() > 0 )
                newBid += 1024;
        }
        if ( up && ( !xEven && out || xEven) ) // Neighbor 11
        {
            if ( nodes[ n + npy + !xEven * npx ].bid() > 0 )
                newBid += 2048;
        }

        node.bid(newBid);
        
        nodes[n] = node;
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Traverses each \p Node in the input array, calculates the new coordinates, 
/// and then copies the \p Node into the new array at the new coordinates.
/// 
/// The transformation is as follows: \f[ \left[ \begin{array}{ccc} x' \\ y' \\ 
/// z' \end{array} \right] = \left[ \begin{array}{ccc} y \\ dim_{x} - 1 - x \\ 
/// z \end{array} \right], \f] where \f$dim_{x}\f$ is the size of the array 
/// along the x-direction. The transformation simply rotates the array 
/// clockwise around the z-axis. When calculating the new index, the dimensions 
/// of the y- and x-direction change places.
///////////////////////////////////////////////////////////////////////////////
template <class Node>
__global__ void unRotateNodes
    ( Node * inputNodes        ///< [in] Filled node array to be un-rotated.
    , Node * outputNodes       ///< [out] Empty node array to be filled.
    , Bounds<uint3> resolution ///< [in] \brief Bounds of the space allocated 
                               ///<             on device. 
                               ///<
    , Bounds<uint2> yzSubSpace ///< [in] \brief Bounds of the subspace to be 
                               ///<             processed.
                               ///<
    )
{
    uint3 res = resolution.max - resolution.min;

    // Space in which the processing should take place.
    Bounds<uint2> space = {
        yzSubSpace.min - make_uint2( resolution.min.y, resolution.min.z ),
        yzSubSpace.max - make_uint2( resolution.min.y, resolution.min.z )
    };

    uint npx = res.x;
    uint npy = npx * res.y;

    uint startNode = npy * space.min.y + npx * space.min.x;
    uint endNode = npy * space.max.y;

    uint nrOfNodes = npy * res.z;
    for ( uint n = startNode + blockIdx.x * blockDim.x + threadIdx.x; 
          n < nrOfNodes && n < endNode; 
          n += gridDim.x * blockDim.x )
    {
        Node node = inputNodes[n];

        uint y = (n % npy) / npx;

        if ( y < space.min.x || y >= space.max.x )
            continue;

        uint x = n % npx;
        uint z = n / npy;

        if ( z < space.min.y || z >= space.max.y )
            continue;

        // New coordinates.
        uint newNodeIdx = 0;

        if ( Node::isFCCNode() )
        {
            const uint newX = 2*y + (x + z + 1) % 2;
            const uint newY = uint(res.x / 2) - 1 - uint(x / 2);

            newNodeIdx = res.x * res.y * z + 2 * res.y * newY + newX;
        }
        else
        {
            const uint newX = y;
            const uint newY = res.x - 1 - x;

            newNodeIdx = res.x * res.y * z + res.y * newY + newX;
        }

        outputNodes[newNodeIdx] = node;
    }
}
///////////////////////////////////////////////////////////////////////////////
/// The <em>boundary id</em>s have been predefined based on the configuration 
/// of neighbors the permutation specifies. The function is just a lookup 
/// table. The permutation is calculated by adding specific values depending on 
/// which neighbors of a \p Node are \a solid. <em>Non-solid</em> neighbors are 
/// ignored. The values that are added are as follows:
///   - Left (-x): 1
///   - Right (+x): 2
///   - In (-y): 4
///   - Out (+y): 8
///   - Down (-z): 16
///   - Up (+z): 32
/// Every different configuration of neighboring nodes produces a unique value.
/// \param[in] permutation The configuration of neighbors encoded into an 
///                        integer.
/// \return Boundary id of a \p Node that matches the given \p permutation.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ uchar getOrientation( uchar permutation )
{
    switch (permutation)
    {
        /* Three neighboring nodes are solid. */
    case 21:
        /* Down, Left, In */
        return 1;
    case 22:
        /* Down, Right, In */
        return 2;
    case 25:
        /* Down, Left, Out */
        return 3;
    case 26:
        /* Down, Right, Out */
        return 4;
    case 37: 
        /* Up, Left, In */
        return 5;
    case 38:
        /* Up, Right, In */
        return 6;
    case 41:
        /* Up, Left, Out */
        return 7;
    case 42:
        /* Up, Right, Out */
        return 8;
        /* Four neighboring nodes are solid. */
    case 23:
        /* Down, Left, Right, In */
        return 9;
    case 27:
        /* Down, Left, Right, Out */
        return 10;
    case 29:
        /* Down, Left, In, Out */
        return 11;
    case 30:
        /* Down, Right, In, Out */
        return 12;
    case 39:
        /* Up, Left, Right, In */
        return 13;
    case 43:
        /* Up, Left, Right, Out */
        return 14;
    case 45:
        /* Up, Left, In, Out */
        return 15;
    case 46:
        /* Up, Right, In, Out */
        return 16;
    case 53:
        /* Up, Down, Left, In */
        return 17;
    case 54:
        /* Up, Down, Right, In */
        return 18;
    case 57:
        /* Up, Down, Left, Out */
        return 19;
    case 58:
        /* Up, Down, Right, Out */
        return 20;
        /* Five neighboring nodes are solid. */
    case 31:
        /* Left, Right, In, Out, Down */
        return 21;
    case 59:
        /* Left, Right, Out, Down, Up */
        return 22;
    case 55:
        /* Left, Right, In, Down, Up */
        return 23;
    case 62:
        /* Right, In, Out, Down, Up */
        return 24;
    case 61:
        /* Left, In, Out, Down, Up */
        return 25;
    case 47:
        /* Left, Right, In, Out, Up */
        return 26;
        /* Six neighboring nodes are solid. */
    case 63:
        /* Left, Right, In, Out, Down, Up */
        return 27;
    default:
        break;
    };
    
    return 0;
}
///////////////////////////////////////////////////////////////////////////////
/// Convenience function that loads \a vertices into an array based on a given  
/// <em>triangle id</em>. Requires that the format in which the vertices are 
/// stored is an array of \p double3.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ void fetchTriangleVertices
    ( float const * vertices ///< [in] Vertices of the model.
    , uint const * indices   ///< [in] Indices of the mdoel.
    , double3 * triangle     ///< [out] Array of 3 double3s.
    , uint triIdx            ///< [in] Triangle index.
    )
{
    for (int t = 0; t < 3; ++t)
        triangle[t] = make_double3( vertices[3*indices[3*triIdx + t]]
                                  , vertices[3*indices[3*triIdx + t] + 1]
                                  , vertices[3*indices[3*triIdx + t] + 2] );
}
///////////////////////////////////////////////////////////////////////////////
/// A simple version of the surface voxelizer. Its overlap test performs the 
/// full test that checks if a triangle intersects with the volume of a voxel, 
/// instead of just a two-dimensional test against a projected voxel center.
/// As a result, this algorithm is much more demanding than the solid 
/// voxelizer. This version is a naive implementation that isn't very 
/// optimized, so it is not recommended to be used for anything. It does, 
/// however present the solution rather cleanly, making it perhaps worth 
/// keeping around.
///
/// The algorithm works in the following way:
///   - Set up all the overlap tests.
///   - Determine if the triangle's place overlaps the voxel.
///   - Determine if the triangle's and voxel's yz-projections overlap.
///   - Determine if the triangle's and voxel's xz-projections overlap.
///   - Determine if the triangle's and voxel's xy-projections overlap.
/// If any of the overlap tests fail, the testing for the triangle can be 
/// aborted. If all of them succeed, then the triangle is confirmed to overlap 
/// the voxel.
///////////////////////////////////////////////////////////////////////////////
template <class Node>
__global__ void SimpleSurfaceVoxelizer
    ( float const * vertices  ///< [in] Vertices of the model.
    , uint const * indices    ///< [in] Indices of the model.
    , Node * nodes            ///< [out] Node array. 
    , uchar const * materials ///< [in] Materials of the triangles.
    , uint nrOfTriangles      ///< [in] Number of triangles in the model.
    , float3 modelBBMin       ///< [in] Minimum bounds of the device's space.
    , float voxelLength       ///< [in] Distance between voxel centers.
    , uint3 resolution        ///< [in] Bounds of the device's space.
    )
{
    float3 triangle[3], triNormal, p;
    float2 ds;
    Bounds<float3> triBB;
    Bounds<uint3> voxBB;
    OverlapData data[3];

    HashMap hm; // Hashmap needs to be passed to processVoxel even though it 
                // is not used there.

    // Loop until there are no more triangles to process.
    for( uint triangleIdx = blockDim.x * blockIdx.x + threadIdx.x
       ; triangleIdx < nrOfTriangles
       ; triangleIdx += gridDim.x * blockDim.x )
    {
        // Load vertices from global memory.
        for ( int i = 0; i < 3; i++ )
            triangle[i] = make_float3( 
                vertices[3*indices[3*triangleIdx + i]], 
                vertices[3*indices[3*triangleIdx + i] + 1], 
                vertices[3*indices[3*triangleIdx + i] + 2] 
            );

        // Calculate normal.
        triNormal = cross( triangle[0] - triangle[2]
                         , triangle[1] - triangle[0] );

        // Ignore degenerate triangles.
        if ( triNormal.x != 0.0f ? true : 
             triNormal.y != 0.0f ? true : 
             triNormal.z != 0.0f ? true : false )
        {
            // Prepare data for overlap testing.
            ds = setupPlaneOverlapTest( triangle, triNormal, voxelLength );
            data[0] = setupYZOverlapTest( triangle, triNormal, voxelLength );
            data[1] = setupZXOverlapTest( triangle, triNormal, voxelLength );
            data[2] = setupXYOverlapTest( triangle, triNormal, voxelLength );

            // Calculate bounding box.
            getTriangleBounds( triangle, triBB );

            // Translate bounding box to voxel coordinates.
            getVoxelBounds( triBB, modelBBMin, voxBB, voxelLength );

            // Perform overlap test on the voxels inside the bounding box.
            for ( uint i = voxBB.min.x; i <= voxBB.max.x; i++ )
            {
                for ( uint j = voxBB.min.y; j <= voxBB.max.y; j++ )
                {
                    for ( uint k = voxBB.min.z; k <= voxBB.max.z; k++ )
                    {
                        p = getSingleVoxelBounds( i
                                                , j
                                                , k
                                                , modelBBMin
                                                , voxelLength );

                        if ( voxelOverlapsTriangle( ds, data, triNormal, p ) )
                            processVoxel( nodes
                                        , materials
                                        , triangleIdx
                                        , triangle
                                        , triNormal
                                        , modelBBMin
                                        , voxelLength
                                        , make_int3( i, j, k )
                                        , make_int3( 0, 0, 0 )
                                        , 1
                                        , resolution
                                        , false
                                        , hm
                                        , (SurfaceNode*)0 );
                    }
                }
            }
        }
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Traverses each vertex of the triangle and finds the minimum and maximum 
/// coordinate components and uses them to construct the minimum and maximum 
/// corners of the bounding box.
/// This version uses \p float3.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ void getTriangleBounds
    ( float3 const * vertices ///< [in] Vertices of the triangle.
    , Bounds<float3> & bounds ///< [out] Bounding box of the triangle.
    )
{
    bounds.min = vertices[0];
    bounds.max = vertices[0];
    
    // Traverse each vertex and find the smallest / largest coordinates.
    for (int i = 1; i < 3; i++)
    {
        if (vertices[i].x < bounds.min.x)
            bounds.min.x = vertices[i].x;
        if (vertices[i].y < bounds.min.y)
            bounds.min.y = vertices[i].y;
        if (vertices[i].z < bounds.min.z)
            bounds.min.z = vertices[i].z;

        if (vertices[i].x > bounds.max.x)
            bounds.max.x = vertices[i].x;
        if (vertices[i].y > bounds.max.y)
            bounds.max.y = vertices[i].y;
        if (vertices[i].z > bounds.max.z)
            bounds.max.z = vertices[i].z;
    }

    return;
}
///////////////////////////////////////////////////////////////////////////////
/// Traverses each vertex of the triangle and finds the minimum and maximum 
/// coordinate components and uses them to construct the minimum and maximum 
/// corners of the bounding box.
/// This version uses \p double3.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ void getTriangleBounds
    ( double3 const * vertices ///< [in] Vertices of the triangle.
    , Bounds<double2> & bounds ///< [out] Bounding box of the triangle.
    )
{
    bounds.min = make_double2(vertices[0].y, vertices[0].z);
    bounds.max = bounds.min;

    // Traverse each vertex and find the smallest / largest coordinates.
    for (int i = 1; i < 3; i++)
    {
        bounds.min.x = 
            vertices[i].y < bounds.min.x ? vertices[i].y : bounds.min.x;
        bounds.min.y = 
            vertices[i].z < bounds.min.y ? vertices[i].z : bounds.min.y;

        bounds.max.x = 
            vertices[i].y > bounds.max.x ? vertices[i].y : bounds.max.x;
        bounds.max.y = 
            vertices[i].z > bounds.max.y ? vertices[i].z : bounds.max.y;
    }

    return;
}
///////////////////////////////////////////////////////////////////////////////
/// The minimum corner is floored and the maximum corner is ceiled.
/// Expects the triangle's bounding box to be made of \p float3 and returns a 
/// bounding box made of \p uint3.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ void getVoxelBounds
    ( Bounds<float3> const & triBB   ///< [in] Triangle's bounding box.
    , float3 const & modelBBMin      /**< [in] Minimum corner of the device's 
                                               voxelization space. */
    , Bounds<uint3> & voxBB          /**< [out] Triangle's bounding 
                                                box in voxel coordinates. */
    , float d                        ///< [in] Distance between voxel centers.
    )
{
    /* Convert to fractional voxel coordinates, then take their floor for the 
       minimum and ceiling for the maximum coodinates. */
    voxBB.min = make_uint3( uint( floorf( (triBB.min.x - modelBBMin.x) / d) )
                          , uint( floorf( (triBB.min.y - modelBBMin.y) / d) )
                          , uint( floorf( (triBB.min.z - modelBBMin.z) / d) ));
    voxBB.max = make_uint3( uint( ceilf( (triBB.max.x - modelBBMin.x) / d) )
                          , uint( ceilf( (triBB.max.y - modelBBMin.y) / d) )
                          , uint( ceilf( (triBB.max.z - modelBBMin.z) / d) ) );
    return;
}
///////////////////////////////////////////////////////////////////////////////
/// Expects the triangle's bounding box to be made of \p double2 and returns a 
/// bounding box made of \p double2.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ void getVoxelBoundsDouble
    ( Bounds<double2> const & triBB ///< [in] Triangle's bounding box.
    , double3 const & modelBBMin    /**< [in] Minimum corner of the device's 
                                              voxelization space. */
    , Bounds<double2> & voxBB       /**< [out] Triangle's bounding 
                                               box in voxel coordinates. */
    , double d                      ///< [in] Distance between voxel centers.
    )
{
    // Convert to fractional voxel coordinates.
    voxBB.min = make_double2( (triBB.min.x - modelBBMin.y) / d, 
                              (triBB.min.y - modelBBMin.z) / d );
    voxBB.max = make_double2( (triBB.max.x - modelBBMin.y) / d, 
                              (triBB.max.y - modelBBMin.z) / d );
}
///////////////////////////////////////////////////////////////////////////////
/// Expects the triangle's bounding box to be made of \p float3 and returns a 
/// bounding box made of \p int3. 
/// \return \p Bounds of \p int3 that describes a triangle's bounding box in 
///         voxel coordinates.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ Bounds<int3> getVoxelBoundsHalf
    ( Bounds<float3> const & triBB ///< [in] Triangle's bounding box.
    , float3 const & modelBBMin    /**< [in] Minimum corner of the device's 
                                        voxelization space. */
    , float d                      ///< [in] Distance between voxel centers.
    )
{
    /* Takes into consideration the fact that voxels have a length of 1.0 after
       the conversion. Adding or subtracting 0.5 and then taking the ceiling 
       or floor gives a more accurate conversion to voxel coordinates than 
       simply casting to int. */
    Bounds<int3> voxBB = {
        make_int3( int( ceilf( (triBB.min.x - modelBBMin.x) / d - 0.5f ) )
                 , int( ceilf( (triBB.min.y - modelBBMin.y) / d - 0.5f ) )
                 , int( ceilf( (triBB.min.z - modelBBMin.z) / d - 0.5f ) ) ),
        make_int3( int( floorf( (triBB.max.x - modelBBMin.x) / d + 0.5f ) )
                 , int( floorf( (triBB.max.y - modelBBMin.y) / d + 0.5f ) )
                 , int( floorf( (triBB.max.z - modelBBMin.z) / d + 0.5f ) ) )
    };

    return voxBB;
}
///////////////////////////////////////////////////////////////////////////////
/// Does what \p getVoxelBoundsHalf() does, but for a single component and only 
/// for the minimum vertex of the bounding box.
/// \return Component of the minimum corner of the triangle's bounding box in 
///         voxel coordinates.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ int getVoxCBoundMin
    ( float v /**< [in] Component of the minimum corner of a triangle's 
                        bounding box. */
    , float m /**< [in] Component of the minimum corner of the device's 
                        voxelization space. */
    , float d ///< [in] Distance between voxel centers.
    ) 
{ 
    return int( ceilf( (v - m) / d - 0.5f ) ); 
}
///////////////////////////////////////////////////////////////////////////////
/// Does what getVoxelBoundsHalf() does, but for a single component and only 
/// for the maximum vertex of the bounding box.
/// \return Component of the maximum corner of the triangle's bounding box in 
///         voxel coordinates.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ int getVoxCBoundMax
    ( float v /**< [in] Component of the maximum corner of a triangle's 
                        bounding box. */
    , float m /**< [in] Component of the minimum corner of the device's 
                        voxelization space. */
    , float d ///< [in] Distance between voxel centers.
    ) 
{ 
    return int( floorf( (v - m) / d + 0.5f ) ); 
}
///////////////////////////////////////////////////////////////////////////////
/// The dominant axis is the component of the normal that has the largest 
/// magnitude.
/// \param[in] triNormal Triangle normal.
/// \return The dominant axis of the triangle's normal.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    MainAxis determineDominantAxis( float3 triNormal )
{
    MainAxis dominantAxis;

    if ( (abs( triNormal.x ) > abs( triNormal.y )) && 
         (abs( triNormal.x ) > abs( triNormal.z )) )
    {
        dominantAxis = xAxis;
    }
    else if ( (abs( triNormal.y ) > abs( triNormal.x )) && 
              (abs( triNormal.y ) > abs( triNormal.z )) )
    {
        dominantAxis = yAxis;
    }
    else
        dominantAxis = zAxis;

    return dominantAxis;
}
///////////////////////////////////////////////////////////////////////////////
/// Used in surface voxelization where the dimensions of a voxel are important.
/// \return Coordinates of the minimum corner of the voxel.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ const float3 getSingleVoxelBounds
    ( int x             ///< [in] X-coordinate of the voxel.
    , int y             ///< [in] Y-coordinate of the voxel.
    , int z             ///< [in] Z-coordinate of the voxel.
    , float3 modelBBMin /**< [in] Minimum corner of the bounding box of the 
                             device's voxelization space. */
    , float d           ///< [in] Distance between voxel centers.
    )
{
    float3 p = {
        modelBBMin.x + (float(x) - 0.5f) * d, 
        modelBBMin.y + (float(y) - 0.5f) * d,
        modelBBMin.z + (float(z) - 0.5f) * d
    };

    return p;
}
///////////////////////////////////////////////////////////////////////////////
/// Calculates a single component of the minimum corner in world coordinates 
/// of a particular voxel. Used in surface voxelization where the dimensions of 
/// a voxel are important.
/// \return Component of the coordinates of the minimum corner of a voxel.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float getSingleVoxelBoundsComponent
    ( int coord             ///< [in] X, Y or Z coordinate of the voxel.
    , float modelBBMinComp  /**< [in] Matching component of the minimum 
                                 corner of the device's voxelization space. */
    , float d               ///< [in] Distance between voxel centers.
    )
{
    float r;

    r = modelBBMinComp + (float(coord) - 0.5f) * d;

    return r;
}
///////////////////////////////////////////////////////////////////////////////
/// Complete overlap test battery used in the simple surface voxelizer. Tests
/// for plane overlap and overlap with each of the three major planes. If all 
/// tests pass, overlap between voxel and triangle is confirmed and \p true is 
/// returned. If a test fails at any point, the whole chain is aborted and 
/// \p false is returned.
/// \return \p true if an overlap was found, \p false otherwise.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ bool voxelOverlapsTriangle
    ( float2 ds                ///< [in] \brief Distances used in the plane 
                               ///<             overlap test.
                               ///<
    , OverlapData const * data ///< [in] \brief Data for the various other 
                               ///<             overlap tests.
                               ///<
    , float3 triNormal         ///< [in] Triangle normal.
    , float3 p                 ///< [in] Voxel coordinates in world space.
    )
{
    // Plane overlap test.
    if ( !planeOverlapTest( ds, triNormal, p ) )
        return false;

    // X overlap test.
    if ( !overlapTestYZ( data[0], p ) )
        return false;

    // Y overlap test.
    if ( !overlapTestZX( data[1], p ) )
        return false;

    // Z overlap test.
    if ( !overlapTestXY( data[2], p ) )
        return false;

    // If all pass, return true.
    return true;
}
///////////////////////////////////////////////////////////////////////////////
/// Precalculates the data needed for the plane overlap test.
/// \return The distances used in the plane overlap test.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float2 setupPlaneOverlapTest
    ( float3 const * tri ///< [in] Triangle vertices.
    , float3 n           ///< [in] Triangle normal.
    , float d            ///< [in] Distance to voxel center.
    )
{
    // Critical point.
    float3 c = {
        n.x > 0.0f ? d : 0.0f,
        n.y > 0.0f ? d : 0.0f,
        n.z > 0.0f ? d : 0.0f
    };

    // d1 and d2.
    float2 r = {
        dot( n, c - tri[0] ),
        dot( n, (make_float3( d, d, d ) - c) - tri[0] )
    };

    return r;
}
///////////////////////////////////////////////////////////////////////////////
/// Convenience function that sets up one of the overlap tests for the surface 
/// voxelization.
/// \return The precalculated data that is used in the actual test.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ OverlapData setupOverlapTest
    ( float3 const * tri ///< [in] Triangle vertices.
    , float3 n           ///< [in] Triangle normal.
    , float d            ///< [in] Distance between voxel centers.
    , MainAxis axis      /**< [in] Which test. The axis along which things 
                                   are projected. */
    )
{
    if (axis == xAxis)
        return setupYZOverlapTest(tri, n, d);
    else if (axis == yAxis)
        return setupZXOverlapTest(tri, n, d);
    else
        return setupXYOverlapTest(tri, n, d);
}
///////////////////////////////////////////////////////////////////////////////
/// \return Edge normals and distances wrapped in a \p OverlapData struct.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ OverlapData setupYZOverlapTest
    ( float3 const * tri ///< [in] Triangle vertices.
    , float3 n           ///< [in] Triangle normal.
    , float d            ///< [in] Distance between voxel centers.
    )
{
    OverlapData data;

    for ( int i = 0; i < 3; i++ )
    {
        // Edge normals.
        if ( n.x >= 0.0f )
            data.ne[i] = make_float2( tri[i].z - tri[(i + 1)%3].z
                                    , tri[(i + 1)%3].y - tri[i].y );
        else
            data.ne[i] = make_float2( tri[(i + 1)%3].z - tri[i].z
                                    , tri[i].y - tri[(i + 1)%3].y );
        // Distances to edges.
        data.de[i] = -(data.ne[i].x * tri[i].y + data.ne[i].y * tri[i].z) + 
                     fmaxf( 0.0f, d * data.ne[i].x ) + 
                     fmaxf( 0.0f, d * data.ne[i].y );
    }

    return data;
}
///////////////////////////////////////////////////////////////////////////////
/// \return Edge normals and distances wrapped in a \p OverlapData struct.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ OverlapData setupZXOverlapTest
    ( float3 const * tri ///< [in] Triangle vertices.
    , float3 n           ///< [in] Triangle normal.
    , float d            ///< [in] Distance between voxel centers.
    )
{
    OverlapData data;

    for ( int i = 0; i < 3; i++ )
    {
        // Edge normals.
        if ( n.y >= 0.0f )
            data.ne[i] = make_float2( tri[i].x - tri[(i + 1)%3].x
                                    , tri[(i + 1)%3].z - tri[i].z );
        else
            data.ne[i] = make_float2( tri[(i + 1)%3].x - tri[i].x
                                    , tri[i].z - tri[(i + 1)%3].z );
        // Distances to edges.
        data.de[i] = -(data.ne[i].x * tri[i].z + data.ne[i].y * tri[i].x) + 
                     fmaxf( 0.0f, d * data.ne[i].x ) + 
                     fmaxf( 0.0f, d * data.ne[i].y );
    }

    return data;
}
///////////////////////////////////////////////////////////////////////////////
/// \return Edge normals and distances wrapped in a \p OverlapData struct.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ OverlapData setupXYOverlapTest
    ( float3 const * tri ///< [in] Triangle vertices.
    , float3 n           ///< [in] Triangle normal.
    , float d            ///< [in] Distance between voxel centers.
    )
{
    OverlapData data;

    for ( int i = 0; i < 3; i++ )
    {
        // Edge normals.
        if ( n.z >= 0.0f )
            data.ne[i] = make_float2( tri[i].y - tri[(i + 1)%3].y
                                    , tri[(i + 1)%3].x - tri[i].x );
        else
            data.ne[i] = make_float2( tri[(i + 1)%3].y - tri[i].y
                                    , tri[i].x - tri[(i + 1)%3].x );
        // Distances.
        data.de[i] = -(data.ne[i].x * tri[i].x + data.ne[i].y * tri[i].y) + 
                     fmaxf( 0.0f, d * data.ne[i].x ) + 
                     fmaxf( 0.0f, d * data.ne[i].y );
    }

    return data;
}
///////////////////////////////////////////////////////////////////////////////
/// This test forgoes additional modifications to the distances, making it 
/// usable only with voxel centers, and not entire voxels.
/// \return Edge normals and distances wrapped in a \p OverlapData struct.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ OverlapData setupSimpleYZOverlapTest
    ( float3 const * tri ///< [in] Triangle vertices.
    , float3 n           ///< [in] Triangle normal.
    , float d            ///< [in] Distance between voxel centers.
    )
{
    OverlapData data;

    for ( int i = 0; i < 3; i++ )
    {
        // Edge normals.
        if ( n.x >= 0.0f )
            data.ne[i] = make_float2( tri[i].z - tri[(i + 1)%3].z
                                    , tri[(i + 1)%3].y - tri[i].y );
        else
            data.ne[i] = make_float2( tri[(i + 1)%3].z - tri[i].z
                                    , tri[i].y - tri[(i + 1)%3].y );
        // Distances.
        data.de[i] = -(data.ne[i].x * tri[i].y + data.ne[i].y * tri[i].z);
    }

    return data;
}
///////////////////////////////////////////////////////////////////////////////
/// This test forgoes additional modifications to the distances, making it 
/// usable only with voxel centers, and not entire voxels.
/// \return Edge normals and distances wrapped in a \p OverlapData struct.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ OverlapData setupSimpleZXOverlapTest
    ( float3 const * tri ///< [in] Triangle vertices.
    , float3 n           ///< [in] Triangle normal.
    , float d            ///< [in] Distance between voxel centers.
    )
{
    OverlapData data;

    for ( int i = 0; i < 3; i++ )
    {
        // Edge normals.
        if ( n.y >= 0.0f )
            data.ne[i] = make_float2( tri[i].x - tri[(i + 1)%3].x
                                    , tri[(i + 1)%3].z - tri[i].z );
        else
            data.ne[i] = make_float2( tri[(i + 1)%3].x - tri[i].x
                                    , tri[i].z - tri[(i + 1)%3].z );
        // Distances.
        data.de[i] = -(data.ne[i].x * tri[i].z + data.ne[i].y * tri[i].x);
    }

    return data;
}
///////////////////////////////////////////////////////////////////////////////
/// This test forgoes additional modifications to the distances, making it 
/// usable only with voxel centers, and not entire voxels.
/// \return Edge normals and distances wrapped in a \p OverlapData struct.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ OverlapData setupSimpleXYOverlapTest
    ( float3 const * tri ///< [in] Triangle vertices.
    , float3 n           ///< [in] Triangle normal.
    , float d            ///< [in] Distance between voxel centers.
    )
{
    OverlapData data;

    for ( int i = 0; i < 3; i++ )
    {
        if ( n.z >= 0.0f )
            data.ne[i] = make_float2( tri[i].y - tri[(i + 1)%3].y
                                    , tri[(i + 1)%3].x - tri[i].x );
        else
            data.ne[i] = make_float2( tri[(i + 1)%3].y - tri[i].y
                                    , tri[i].x - tri[(i + 1)%3].x );

        data.de[i] = -(data.ne[i].x * tri[i].x + data.ne[i].y * tri[i].y);
    }

    return data;
}
///////////////////////////////////////////////////////////////////////////////
/// \return \p true if the plane of the triangle overlaps the voxel, \p false 
///         otherwise.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ bool planeOverlapTest
    ( float2 ds        ///< [in] Distances.
    , float3 triNormal ///< [in] Triangle normal.
    , float3 p         ///< [in] Voxel's minimum corner in world coordinates.
    )
{
    return ((dot( triNormal, p ) + ds.x) * 
            (dot( triNormal, p ) + ds.y)) <= 0.0f;
}
///////////////////////////////////////////////////////////////////////////////
/// The axis along which things are projected can be specified and this 
/// determines which test is performed.
/// \return \p true if the overlap test succeeded, \p false otherwise.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ bool overlapTest
    ( OverlapData data ///< [in] Data used during the overlap test.
    , float3 p         ///< [in] Voxel's minimum corner in world coordinates.
    , MainAxis axis    ///< [in] Axis along which the projections occur.
    )
{
    bool testResults = true;
    float2 p2;

    if ( axis == xAxis ) // YZ test.
    {
        p2 = make_float2( p.y, p.z );
    }
    else if ( axis == yAxis ) // ZX test.
    {
        p2 = make_float2( p.z, p.x );
    }
    else // XY test.
    {
        p2 = make_float2( p.x, p.y );
    }

    for (int i = 0; i < 3; i++)
        testResults &= dot( data.ne[i], p2 ) + data.de[i] >= 0.0f;

    return testResults;
}
///////////////////////////////////////////////////////////////////////////////
/// \param[in] data Precalculated data used in the overlap test.
/// \param[in] p Voxel's minimum corner in world coordinates.
/// \return \p true if the voxel overlaps the triangle in the given plane, \p 
///         false otherwise.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    bool overlapTestYZ( OverlapData data, float3 p )
{
    bool testResults = true;
    float2 p2;

    p2 = make_float2( p.y, p.z );

    for (int i = 0; i < 3; i++)
        testResults &= dot( data.ne[i], p2 ) + data.de[i] >= 0.0f;

    return testResults;
}
///////////////////////////////////////////////////////////////////////////////
/// \param[in] data Precalculated data used in the overlap test.
/// \param[in] p Voxel's minimum corner in world coordinates.
/// \return \p true if the voxel overlaps the triangle in the given plane, \p 
///         false otherwise.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    bool overlapTestZX( OverlapData data, float3 p )
{
    bool testResults = true;
    float2 p2;

    p2 = make_float2( p.z, p.x );

    for (int i = 0; i < 3; i++)
        testResults &= dot( data.ne[i], p2 ) + data.de[i] >= 0.0f;

    return testResults;
}
///////////////////////////////////////////////////////////////////////////////
/// \param[in] data Precalculated data used in the overlap test.
/// \param[in] p Voxel's minimum corner in world coordinates.
/// \return \p true if the voxel overlaps the triangle in the given plane, \p 
///         false otherwise.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    bool overlapTestXY( OverlapData data, float3 p )
{
    bool testResults = true;
    float2 p2;

    p2 = make_float2( p.x, p.y );

    for (int i = 0; i < 3; i++)
        testResults &= dot( data.ne[i], p2 ) + data.de[i] >= 0.0f;

    return testResults;
}
///////////////////////////////////////////////////////////////////////////////
/// ???
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
inline __device__ void processVoxel
    ( Node * nodes            ///< [out] Node array.
    , uchar const * materials ///< [in] Material to triangle mapping.
    , uint triangleIdx        ///< [in] Current triangle index.
    , float3 * triangle       ///< [in] Triangle vertices.
    , float3 triNormal        ///< [in] Triangle normal.
    , float3 modelBBMin       /**< [in] Minimum corner of the device's voxel 
                                        space. */
    , float voxelLength       ///< [in] Distance between voxel centers.
    , int3 coords
    , int3 adjs
    , int gridType            ///< [in] How the grid is shifted - 1 for normal.
    , uint3 resolution        ///< [in] Dimensions of the device's voxel space.
    , bool countVoxels
    , HashMap & hashMap
    , SNode * surfNodes
    )
{
    uint nodeIdx = 0;

    const uint x = coords.x + adjs.x;
    const uint y = coords.y + adjs.y;
    const uint z = coords.z + adjs.z;

    if ( Node::isFCCNode() )
    {   // The node is to be written into a FCC grid.
        const uint l = 2 * resolution.x;
        const uint a = l * resolution.y;
        
        if ( gridType == 1 )
            nodeIdx = a * 2*z + l * y + 2*x;
        else if ( gridType == 2 )
            nodeIdx = a * 2*z + l * y + 2*x + 1;
        else if ( gridType == 3 )
            nodeIdx = a * (2*z + 1) + l * y + 2*x;
        else if ( gridType == 4 )
            nodeIdx = a * (2*z + 1) + l * y + 2*x + 1;
    }
    else
    {   // The node is to be written into a normal grid.

        const uint l = resolution.x;
        const uint a = l * resolution.y;

        nodeIdx = a*z + l*y + x;
    }

    if ( countVoxels )
    {
        nodes[nodeIdx] = Node( 1 );
        return;
    }

    if ( Node::hasRatio() )
    {
        if ( Node::usesTwoArrays() )
        {
            SNode n;
            float a0, a1, a2, a3, a4, a5, a6;
            float volume = 
                calculateCutVolumeAndAreas( triangle
                                          , make_int3( coords.x
                                                     , coords.y
                                                     , coords.z )
                                          , triNormal
                                          , modelBBMin
                                          , voxelLength
                                          , a0
                                          , a1
                                          , a2
                                          , a3
                                          , a4
                                          , a5
                                          , a6 );

            n.volume = volume;

            n.cutNormal = triNormal;
            n.cutArea = abs(a0);

            n.xPosArea = abs(a1);
            n.xNegArea = abs(a6);

            if ( a6 < 0.0f )
            {
                n.xPosArea = abs(a6);
                n.xNegArea = abs(a1);
            }

            bool x = triNormal.x >= 0.0f;
            bool y = triNormal.y >= 0.0f;
            bool z = triNormal.z >= 0.0f;

            if ( (x + y + z) % 2 == 1 )
            {   // Don't swap y and z.
                n.yPosArea = abs(a2);
                n.yNegArea = abs(a5);

                if ( a5 < 0.0f )
                {
                    n.yPosArea = abs(a5);
                    n.yNegArea = abs(a2);
                }

                n.zPosArea = abs(a3);
                n.zNegArea = abs(a4);

                if ( a4 <= 0.0f )
                {
                    n.zPosArea = abs(a4);
                    n.zNegArea = abs(a3);
                }
            }
            else
            {   // Swap y and z.
                n.yPosArea = abs(a3);
                n.yNegArea = abs(a4);

                if ( a4 < 0.0f )
                {
                    n.yPosArea = abs(a4);
                    n.yNegArea = abs(a3);
                }

                n.zPosArea = abs(a2);
                n.zNegArea = abs(a5);

                if ( a5 < 0.0f )
                {
                    n.zPosArea = abs(a5);
                    n.zNegArea = abs(a2);
                }
            }

            n.material = materials[triangleIdx];

            uint surfNodeIdx = hashMap.get( nodeIdx );

            if ( surfNodeIdx != UINT32_MAX )
            {
                surfNodes[surfNodeIdx] = n;
                //nodes[nodeIdx] = Node( 1 );
            }
        }
        else
        {
            float volume, volRatio;

            // Calculate the fractional volume of the cut voxel.
            volume = calculateVoxelPlaneIntersectionVolume( 
                triangle, 
                make_int3( coords.x, coords.y, coords.z ), 
                triNormal, 
                modelBBMin, 
                voxelLength );

            // The volume ratio we need is the other side of the cut voxel.
            volRatio = 
                1.0f - volume / (voxelLength * voxelLength * voxelLength);

            // Put the node into memory.
            nodes[nodeIdx] = 
                Node( 1, materials[triangleIdx], abs( volRatio ) );
        }
        return;
    }

    // Make the node solid and set its material.
    nodes[nodeIdx] = Node( 1, materials[triangleIdx] );
}

///////////////////////////////////////////////////////////////////////////////
/// Kernel that processes each triangle of the model and produces a 
/// classification for each one. The classification is then meant to be sorted 
/// so that similar triangles would be processed close to each other. Most of 
/// the actual work is done in the analyzeBoundingBox() function.
///////////////////////////////////////////////////////////////////////////////
__global__ void classifyTriangles
    ( float const * vertices ///< [in] Vertices of the model.
    , uint const * indices   ///< [in] Indices of the model.
    , uint * triTypeBuffer   ///< [out] Array of triangle types to write to.
    , uint nrOfTriangles     ///< [in] Number of triangles in the model.
    , float3 modelBBMin      /**< [in] Minimum corner of the device's 
                                  voxelization space. */
    , float	voxelLength      ///< [in] Distance between voxel centers.
    )
{
    float3 triangle[3], triNormal;
    Bounds<float3> triBB;
    TriData data;
    uint triType;

    // Loop until there are no more triangles to process.
    for( uint triangleIdx = blockDim.x * blockIdx.x + threadIdx.x
       ; triangleIdx < nrOfTriangles
       ; triangleIdx += gridDim.x * blockDim.x )
    {
        // Load vertices from global memory.
        for ( int i = 0; i < 3; i++ )
            triangle[i] = 
                make_float3( vertices[3*indices[3*triangleIdx + i]]
                           , vertices[3*indices[3*triangleIdx + i] + 1]
                           , vertices[3*indices[3*triangleIdx + i] + 2] );

        // Calculate normal.
        triNormal = cross( triangle[0] - triangle[2]
                         , triangle[1] - triangle[0] );

        // Ignore degenerate triangles.
        if ( triNormal.x != 0.0f ? true : 
             triNormal.y != 0.0f ? true : 
             triNormal.z != 0.0f ? true : false )
        {
            // Calculate bounding box.
            getTriangleBounds( triangle, triBB );

            // Analyze bounding box.
            data = analyzeBoundingBox( triBB
                                     , triNormal
                                     , modelBBMin
                                     , voxelLength );

            // Encode the data.
            triType = encodeTriangleType( data );

            // Write to memory.
            triTypeBuffer[triangleIdx] = triType;
        }
        else
        {
            // Set up degenerate triangle type.
            data.bbType = BBType_Degen;
            data.domAxis = xAxis;
            data.nrOfVoxCols = 0;

            // Encode the data.
            triType = encodeTriangleType( data );

            // Write to memory.
            triTypeBuffer[triangleIdx] = triType;
        }
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Calculates the dimensions of the triangle's bounding box in voxel 
/// coordinates and then analyzes it to determine what kind of bounding box the 
/// triangle has. A one-dimensional bounding box is one voxel thick along at 
/// least two directions. A two-dimensional bounding box is one voxel thick 
/// along one direction. A three dimensional bounding box doesn't fit any of 
/// the two previous criteria.
/// \return The triangle's bounding box type, its dominant axis and how many 
///         voxel columns the triangle spans at most along the dominant axis, 
///         wrapped into a \p TriData struct.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ const TriData analyzeBoundingBox
    ( Bounds<float3> triBB ///< [in] Triangle's bounding box.
    , float3 triNormal     ///< [in] Triangle normal.
    , float3 modelBBMin    /**< [in] Minimum corner of the device's voxel 
                                     space. */
    , float d              ///< [in] Distance between voxel centers.
    ) 
{
    Bounds<int3> voxBB;
    int3 voxelDiff;
    TriData data;

    // Turn the triangle bounding box into voxel coordinates.
    voxBB = getVoxelBoundsHalf( triBB, modelBBMin, d );

    // Get the dimension of the box of voxels.
    voxelDiff = voxBB.max - voxBB.min;

    if (voxelDiff.x == 0) // At least the X-direction is one voxel thick.
    {
        if (voxelDiff.y == 0) // X- and Y-directions are one voxel thick.
        {
            if (voxelDiff.z == 0) // X-, Y- and Z-directions are one voxel thick.
            {
                data.bbType = BBType_1D;
                data.domAxis = xAxis;
                data.nrOfVoxCols = 1;
            }
            else // Only the X- and Y-directions are one voxel thick.
            {
                data.bbType = BBType_1D;
                data.domAxis = zAxis;
                data.nrOfVoxCols = voxelDiff.z;
            }
        }
        else if (voxelDiff.z == 0) // X- and Z-directions are one voxel thick.
        {
            data.bbType = BBType_1D;
            data.domAxis = yAxis;
            data.nrOfVoxCols = voxelDiff.y;
        }
        else // Only the X-direction is one voxel thick.
        {
            data.bbType = BBType_2D;
            data.domAxis = xAxis;
            data.nrOfVoxCols = voxelDiff.y * voxelDiff.z;
        }
    }
    else if (voxelDiff.y == 0) // The Y-direction is one voxel thick, but not the X-direction.
    {
        if (voxelDiff.z == 0) // Y- and Z-directions are on voxel thick.
        {
            data.bbType = BBType_1D;
            data.domAxis = xAxis;
            data.nrOfVoxCols = voxelDiff.x;
        }
        else // Only the Y-direction is one voxel thick.
        {
            data.bbType = BBType_2D;
            data.domAxis = yAxis;
            data.nrOfVoxCols = voxelDiff.x * voxelDiff.z;
        }
    }
    else if (voxelDiff.z == 0) // Only the Z-direction is one voxel thick.
    {
        data.bbType = BBType_2D;
        data.domAxis = zAxis;
        data.nrOfVoxCols = voxelDiff.x * voxelDiff.y;
    }
    else // No directions are just one voxel thick.
    {
        data.bbType = BBType_3D;
        data.domAxis = determineDominantAxis(triNormal);
        if (data.domAxis == xAxis)
        {
            data.nrOfVoxCols = voxelDiff.y * voxelDiff.z;
        }
        else if (data.domAxis == yAxis)
        {
            data.nrOfVoxCols = voxelDiff.x * voxelDiff.z;
        }
        else 
        {
            data.nrOfVoxCols = voxelDiff.x * voxelDiff.y;
        }
    }

    return data;
}
///////////////////////////////////////////////////////////////////////////////
/// Encodes a triangle's bounding box type, dominant axis and number of voxel 
/// columns into a single unsigned integer. The most weight is given to the 
/// bounding box type, followed by the dominant axis and then the number of 
/// voxel columns.
/// \param[in] data The triangle data wrapped in a \p TriData struct.
/// \return An unsigned integer that encodes the information within the data.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ uint encodeTriangleType( TriData data )
{
    int bits = VOX_BPI - 4;
    uint bb_type = uint(data.bbType) << (bits + 2);
    uint dom_axis = uint(data.domAxis) << bits;

    return uint(data.nrOfVoxCols) | dom_axis | bb_type;
}
///////////////////////////////////////////////////////////////////////////////
/// \param[in] type Encoded \p TriData.
/// \return Decoded \p TriData.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    const TriData decodeTriangleType( uint type )
{
    TriData data;

    data.bbType = readBBType(type);
    data.domAxis = readDomAxis(type);
    data.nrOfVoxCols = readNrOfVoxCols(type);

    return data;
}
///////////////////////////////////////////////////////////////////////////////
/// \param[in] type Encoded \p TriData.
/// \return Bounding box type.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ BBType readBBType( uint type )
{
    return static_cast<BBType>(type >> (VOX_BPI - 2));
}
///////////////////////////////////////////////////////////////////////////////
/// \param[in] type Encoded \p TriData.
/// \return Dominant axis.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ MainAxis readDomAxis( uint type )
{
    return static_cast<MainAxis>((type >> (VOX_BPI - 4)) & 3u);
}
///////////////////////////////////////////////////////////////////////////////
/// \param[in] type Encoded \p TriData.
/// \return Number of voxel columns.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ uint readNrOfVoxCols( uint type )
{
    return type & ((1 << (VOX_BPI - 4)) - 1);
}
///////////////////////////////////////////////////////////////////////////////
/// Determines largest surface of a face of a box-shaped volume.
/// \param[in] res The dimensions of the volume in voxels.
/// \return The largest surface area of a single face.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    uint calculateLargestSideOfVoxels( uint3 res )
{
    uint result;

    if (res.x > res.y)
    {
        if (res.z > res.y)
        {
            // x * z
            result = res.x * res.z;
        }
        else
        {
            // x * y
            result = res.x * res.y;
        }
    }
    else
    {
        if (res.z > res.x)
        {
            // y * z
            result = res.y * res.z;
        }
        else
        {
            // x * y
            result = res.x * res.y;
        }
    }

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Kernel that processes and voxelizes one dimensional triangles. One 
/// dimensional triangles are triangles that fit into a one voxel thick column
/// of voxel(s). Because of this, the triangle has to intersect with all of the 
/// voxels in its bounding box. No overlap tests need to be performed.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
__global__ void process1DTriangles
    ( float const * vertices  ///< [in] Vertices of the model.
    , uint const * indices    ///< [in] Indices of the model.
    , uint const * triTypes   ///< [in] Triangle classifications.
    , uint const * triangles  ///< [in] Classification to triangle mapping.
    , uchar const * materials ///< [in] Triangle to material mapping.
    , Node * nodes            ///< [out] Array of nodes.
    , uint triStartIndex      ///< [in] Starting index for this triangle type.
    , uint triEndIndex        ///< [in] End index for this triangle type.
    , bool left               ///< [in] Other device on the left?
    , bool right              ///< [in] Other device on the right?
    , bool up                 ///< [in] Other device above?
    , bool down               ///< [in] Other device below?
    , float3 modelBBMin       ///< [in] Minimum corner of the device's 
                              ///<      voxelization space.
                              ///<
    , float	voxelLength       ///< [in] Distance between voxel centers.
    , Bounds<uint3> totalResolution ///< [in] Bounding box of the device's 
                                    ///<      voxelization space.
                                    ///<
    , Bounds<uint3> subSpace  ///< [in] Bounding box of the subspace.
    , int gridType            ///< [in] Which of the four grids to use.
    , bool countVoxels        ///< [in] \p true to forego the normal 
                              ///<      voxelization and instead calculate the 
                              ///<      number of overlaps per voxel.
    , HashMap hashMap
    , SNode * surfNodes
    )
{
    float3 triangle[3];
    Bounds<float3> triBB;
    Bounds<int3> voxBB;
    MainAxis domAxis;
    uint triType;

    // Loop until there are no more triangles to process.
    for( uint t = triStartIndex + blockDim.x * blockIdx.x + threadIdx.x; 
         t < triEndIndex; 
         t += gridDim.x * blockDim.x )
    {
        // Get the triangle index and load the vertices from global memory.
        uint const triIdx = triangles[t];
        for (int i = 0; i < 3; i++)
            triangle[i] = make_float3( vertices[3*indices[3*triIdx + i]], 
                                       vertices[3*indices[3*triIdx + i] + 1], 
                                       vertices[3*indices[3*triIdx + i] + 2] );

        // Calculate bounding box.
        getTriangleBounds( triangle, triBB );

        // Calculate voxel range.
        voxBB = getVoxelBoundsHalf( triBB, modelBBMin, voxelLength );

        // Make subSpace relative to the new origin of modelBBMin.
        Bounds<uint3> space = {
            subSpace.min - totalResolution.min,
            subSpace.max - totalResolution.min
        };

        // Bind the voxel range to the subspace.
        adjustVoxelRange( voxBB, space );

        // Go to next triangle if the processable voxels aren't part of the 
        // subspace.
        if (voxBB.min.x == -1)
            continue;

        // Retrieve triangle data.
        triType = triTypes[t];
        domAxis = readDomAxis(triType);
        
        // Adjust coordinates to account for the padding.
        int3 const adjustment = { 1,
                                  left ? 0 : 1,
                                  down ? 0 : 1 };
        //voxBB.min += adjustment;
        //voxBB.max += adjustment;

        uint3 res = totalResolution.max - totalResolution.min;

        res.y += adjustment.y + (right ? 0 : 1);
        res.z += adjustment.z + (up ? 0 : 1);

        float3 triNormal = 
            Node::hasRatio() ? cross( triangle[0] - triangle[2]
                                    , triangle[1] - triangle[0] )
                             : make_float3( 0.0f );

        // Process the voxels.
        if (domAxis == xAxis)
        {
            for ( int i = voxBB.min.x; i <= voxBB.max.x; i++ )
            {
                processVoxel( nodes
                            , materials
                            , triIdx
                            , triangle
                            , triNormal
                            , modelBBMin
                            , voxelLength
                            , make_int3( i, voxBB.min.y, voxBB.min.z )
                            , adjustment
                            , gridType
                            , res
                            , countVoxels
                            , hashMap
                            , surfNodes );
            }
        }
        else if (domAxis == yAxis)
        {
            for ( int j = voxBB.min.y; j <= voxBB.max.y; j++ )
            {
                processVoxel( nodes
                            , materials
                            , triIdx
                            , triangle
                            , make_float3(0.0f)
                            , modelBBMin
                            , voxelLength
                            , make_int3( voxBB.min.x, j, voxBB.min.z )
                            , adjustment
                            , gridType
                            , res
                            , countVoxels
                            , hashMap
                            , surfNodes );
            }
        }
        else
        {
            for ( int k = voxBB.min.z; k <= voxBB.max.z; k++ )
            {
                processVoxel( nodes
                            , materials
                            , triIdx
                            , triangle
                            , make_float3(0.0f)
                            , modelBBMin
                            , voxelLength
                            , make_int3( voxBB.min.x, voxBB.min.y, k )
                            , adjustment
                            , gridType
                            , res
                            , countVoxels
                            , hashMap
                            , surfNodes );
            }
        }
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Kernel that processes and voxelizes two-dimensional triangles. 
/// Two-dimensional triangles are triangles that have a bounding box whose one 
/// side is at most one voxel thick. This means that the triangle only needs to 
/// be tested for overlap along its dominant axis.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
__global__ void process2DTriangles
    ( float const * vertices        ///< [in] Vertices of the model.
    , uint const * indices          ///< [in] Indices of the model.
    , uint const * triTypes         ///< [in] Triangle classifications.
    , uint const * triangles        /**< [in] Classification to triangle 
                                              mapping. */
    , uchar const * materials       ///< [in] Triangle to material mapping.
    , Node * nodes                  ///< [out] Array of nodes.
    , uint triStartIndex            /**< [in] Starting index for this triangle 
                                              type. */
    , uint triEndIndex              ///< [in] End index for this triangle type.
    , bool left                     ///< [in] Other device on the left?
    , bool right                    ///< [in] Other device on the right?
    , bool up                       ///< [in] Other device above?
    , bool down                     ///< [in] Other device below?
    , float3 modelBBMin             /**< [in] Minimum corner of the device's 
                                              voxelization space. */
    , float	voxelLength             ///< [in] Distance between voxel centers.
    , Bounds<uint3> totalResolution /**< [in] Bounding box of the device's 
                                              voxelization space. */
    , Bounds<uint3> subSpace        ///< [in] Bounding box of the subspace.
    , int gridType                  ///< [in] Which of the four grids to use.
    , bool countVoxels
    , HashMap hashMap
    , SNode * surfNodes
    )
{
    float3 triangle[3], triNormal, p;
    Bounds<float3> triBB;
    Bounds<int3> voxBB;
    MainAxis domAxis;
    OverlapData data;
    uint triType, triIdx;

    // Loop until there are no more triangles to process.
    for( uint t = triStartIndex + blockDim.x * blockIdx.x + threadIdx.x; 
         t < triEndIndex; 
         t += gridDim.x * blockDim.x )
    {
        // Get the triangle index and load the vertices from global memory.
        triIdx = triangles[t];
        for ( int i = 0; i < 3; i++ )
            triangle[i] = make_float3( vertices[3*indices[3*triIdx + i]], 
                                       vertices[3*indices[3*triIdx + i] + 1], 
                                       vertices[3*indices[3*triIdx + i] + 2] );

        // Calculate normal.
        triNormal = 
            cross( triangle[0] - triangle[2], triangle[1] - triangle[0] );

        // Calculate bounding box.
        getTriangleBounds( triangle, triBB );

        // Calculate voxel range.
        voxBB = getVoxelBoundsHalf( triBB, modelBBMin, voxelLength );

        // Make subSpace relative to the new origin of modelBBMin.
        Bounds<uint3> space = {
            subSpace.min - totalResolution.min,
            subSpace.max - totalResolution.min
        };

        // Bind the voxel range to the subspace.
        adjustVoxelRange( voxBB, space );

        // Go to next triangle if the procesable voxels aren't part of the 
        // subspace.
        if (voxBB.min.x == -1)
            continue;

        // Determine axis of projection.
        triType = triTypes[t];
        domAxis = readDomAxis(triType);

        // Adjust coordinates to account for the padding.
        int3 const adjs = make_int3( 1, (left ? 0 : 1), (down ? 0 : 1) );

        uint3 res = totalResolution.max - totalResolution.min;

        res.y += adjs.y + (right ? 0 : 1);
        res.z += adjs.z + (up ? 0 : 1);

        // Go through the voxels.
        if (domAxis == xAxis)
        {
            data = setupYZOverlapTest( triangle, triNormal, voxelLength );

            for ( int j = voxBB.min.y; j <= voxBB.max.y; j++ )
            {
                for ( int k = voxBB.min.z; k <= voxBB.max.z; k++ )
                {
                    p = make_float3(
                            0.0f,
                            getSingleVoxelBoundsComponent( j, 
                                                           modelBBMin.y, 
                                                           voxelLength ),
                            getSingleVoxelBoundsComponent( k, 
                                                           modelBBMin.z, 
                                                           voxelLength ) );
                    if ( overlapTestYZ( data, p ) )
                    {
                        processVoxel( nodes
                                    , materials
                                    , triIdx
                                    , triangle
                                    , triNormal
                                    , modelBBMin
                                    , voxelLength
                                    , make_int3( voxBB.min.x, j, k )
                                    , adjs
                                    , gridType
                                    , res
                                    , countVoxels
                                    , hashMap
                                    , surfNodes );
                    }
                }
            }
        }
        else if (domAxis == yAxis)
        {
            data = setupZXOverlapTest( triangle, triNormal, voxelLength );

            for ( int i = voxBB.min.x; i <= voxBB.max.x; i++ )
            {
                for ( int k = voxBB.min.z; k <= voxBB.max.z; k++ )
                {
                    p = make_float3(
                        getSingleVoxelBoundsComponent( i, 
                                                       modelBBMin.x, 
                                                       voxelLength ),
                        0.0f,
                        getSingleVoxelBoundsComponent( k, 
                                                       modelBBMin.z, 
                                                       voxelLength ) );
                    if ( overlapTestZX( data, p ) )
                    {
                        processVoxel( nodes
                                    , materials
                                    , triIdx
                                    , triangle
                                    , triNormal
                                    , modelBBMin
                                    , voxelLength
                                    , make_int3( i, voxBB.min.y, k )
                                    , adjs
                                    , gridType
                                    , res
                                    , countVoxels
                                    , hashMap
                                    , surfNodes );
                    }
                }
            }
        }
        else if (domAxis == zAxis)
        {
            data = setupXYOverlapTest( triangle, triNormal, voxelLength );

            for ( int i = voxBB.min.x; i <= voxBB.max.x; i++ )
            {
                for ( int j = voxBB.min.y; j <= voxBB.max.y; j++ )
                {
                    p = make_float3(
                        getSingleVoxelBoundsComponent( i, 
                                                       modelBBMin.x, 
                                                       voxelLength ),
                        getSingleVoxelBoundsComponent( j, 
                                                       modelBBMin.y, 
                                                       voxelLength ),
                        0.0f );
                    if ( overlapTestXY( data, p ) )
                    {
                        processVoxel( nodes
                                    , materials
                                    , triIdx
                                    , triangle
                                    , triNormal
                                    , modelBBMin
                                    , voxelLength
                                    , make_int3( i, j, voxBB.min.z )
                                    , adjs
                                    , gridType
                                    , res
                                    , countVoxels
                                    , hashMap
                                    , surfNodes );
                    }
                }
            }
        }
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Kernel that processes and voxelizes three-dimensional triangles. These are 
/// triangles that have a bounding box that is more than one voxel thick in 
/// all directions. In order to determine overlap, three overlap tests need to 
/// be performed -- one along each main axis.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
__global__ void process3DTriangles
    ( float const * vertices        ///< [in] Vertices of the model.
    , uint const * indices          ///< [in] Indices of the model.
    , uint const * triTypes         ///< [in] Triangle classifications.
    , uint const * triangles        /**< [in] Classification to triangle 
                                              mapping. */
    , uchar const * materials       ///< [in] Triangle to material mapping.
    , Node * nodes                  ///< [out] Array of nodes.
    , uint triStartIndex            /**< [in] Starting index for this triangle 
                                              type. */
    , uint triEndIndex              ///< [in] End index for this triangle type.
    , bool left                     ///< [in] Other device on the left?
    , bool right                    ///< [in] Other device on the right?
    , bool up                       ///< [in] Other device above?
    , bool down                     ///< [in] Other device below?
    , float3 modelBBMin             /**< [in] Minimum corner of the device's 
                                              voxelization space. */
    , float	voxelLength             ///< [in] Distance between voxel centers.
    , Bounds<uint3> totalResolution /**< [in] Bounding box of the device's 
                                              voxelization space. */
    , Bounds<uint3> subSpace        ///< [in] Bounding box of the subspace.
    , int gridType                  ///< [in] Which of the four grids to use.
    , bool countVoxels
    , HashMap hashMap
    , SNode * surfNodes
    )
{
    float3 triangle[3], triNormal, p;
    Bounds<float3> triBB;
    Bounds<int3> voxBB;
    MainAxis domAxis;
    OverlapData data[3];
    int2 voxRange;
    uint triType, triIdx;

    // Loop until there are no more triangles to process.
    for( uint t = triStartIndex + blockDim.x * blockIdx.x + threadIdx.x; 
         t < triEndIndex; 
         t += gridDim.x * blockDim.x)
    {
        // Get the triangle index and load the vertices from global memory.
        triIdx = triangles[t];
        for (int i = 0; i < 3; i++)
            triangle[i] = make_float3( vertices[3*indices[3*triIdx + i]], 
                                       vertices[3*indices[3*triIdx + i] + 1], 
                                       vertices[3*indices[3*triIdx + i] + 2] );

        // Calculate normal.
        triNormal = 
            cross( triangle[0] - triangle[2], triangle[1] - triangle[0] );

        // Calculate bounding box.
        getTriangleBounds( triangle, triBB );

        // Calculate voxel range.
        voxBB = getVoxelBoundsHalf( triBB, modelBBMin, voxelLength );

        // Make subSpace relative to the new origin of modelBBMin.
        Bounds<uint3> space = {
            subSpace.min - totalResolution.min,
            subSpace.max - totalResolution.min
        };

        // Bind the voxel range to the subspace.
        adjustVoxelRange( voxBB, space );

        // Go to next triangle if the processable voxels aren't part of the 
        // subspace.
        if (voxBB.min.x == -1)
            continue;

        // Determine axis of projection.
        triType = triTypes[t];
        domAxis = readDomAxis(triType);

        // Setup overlap tests.
        data[0] = setupYZOverlapTest( triangle, triNormal, voxelLength );
        data[1] = setupZXOverlapTest( triangle, triNormal, voxelLength );
        data[2] = setupXYOverlapTest( triangle, triNormal, voxelLength );

        // Adjust coordinates to account for the padding.
        int3 const adjs = { 1, (left ? 0 : 1), (down ? 0 : 1) };

        uint3 res = totalResolution.max - totalResolution.min;
        res.y += adjs.y + (right ? 0 : 1);
        res.z += adjs.z + (up ? 0 : 1);

        // Optimize per dominant axis.
        if (domAxis == xAxis)
        {
            // Loop through the voxel columns of the yz-plane.
            for ( int j = voxBB.min.y; j <= voxBB.max.y; j++ )
            {
                for ( int k = voxBB.min.z; k <= voxBB.max.z; k++ )
                {
                    p = make_float3(
                        0.0f, 
                        getSingleVoxelBoundsComponent( j, 
                                                       modelBBMin.y, 
                                                       voxelLength ),
                        getSingleVoxelBoundsComponent( k, 
                                                       modelBBMin.z, 
                                                       voxelLength ) );

                    if ( overlapTestYZ( data[0], p ) )
                    {
                        // Determine voxel depth.
                        voxRange = determineDepthRangeX( triangle, 
                                                         p, 
                                                         triNormal, 
                                                         modelBBMin, 
                                                         voxelLength );

                        // Ignore coordinates that are outside the subspace.
                        if ( voxRange.x > voxBB.max.x || 
                             voxRange.y < voxBB.min.x )
                            continue;

                        // Adjust range if parts of it are outside the 
                        // subspace.
                        voxRange.x = 
                            voxRange.x < voxBB.min.x ? voxBB.min.x 
                                                     : voxRange.x;
                        voxRange.y = 
                            voxRange.y > voxBB.max.x ? voxBB.max.x 
                                                     : voxRange.y;

                        // No need to perform overlap tests if the range is 
                        // only one voxel thick.
                        if ( voxRange.x == voxRange.y )
                        {
                            processVoxel( nodes
                                        , materials
                                        , triIdx
                                        , triangle
                                        , triNormal
                                        , modelBBMin
                                        , voxelLength
                                        , make_int3( voxRange.x, j, k )
                                        , adjs
                                        , gridType
                                        , res
                                        , countVoxels
                                        , hashMap
                                        , surfNodes );
                        }
                        else
                        {
                            // Perform remaining overlap tests on max 3 voxels.
                            for ( int i = voxRange.x; i <= voxRange.y; i++ )
                            {
                                p.x = getSingleVoxelBoundsComponent(
                                    i, 
                                    modelBBMin.x, 
                                    voxelLength );

                                if ( overlapTestZX( data[1], p ) )
                                {
                                    if ( overlapTestXY( data[2], p ) )
                                    {
                                        processVoxel( nodes
                                                    , materials
                                                    , triIdx
                                                    , triangle
                                                    , triNormal
                                                    , modelBBMin
                                                    , voxelLength
                                                    , make_int3( i, j, k )
                                                    , adjs
                                                    , gridType
                                                    , res
                                                    , countVoxels
                                                    , hashMap
                                                    , surfNodes );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (domAxis == yAxis)
        {
            // Loop through the voxel columns of the zx-plane.
            for ( int k = voxBB.min.z; k <= voxBB.max.z; k++ )
            {
                for ( int i = voxBB.min.x; i <= voxBB.max.x; i++ )
                {
                    p = make_float3(
                        getSingleVoxelBoundsComponent( i, 
                                                       modelBBMin.x, 
                                                       voxelLength ), 
                        0.0f,
                        getSingleVoxelBoundsComponent( k, 
                                                       modelBBMin.z, 
                                                       voxelLength ) );

                    if (overlapTestZX(data[1], p))
                    {
                        // Determine voxel depth.
                        voxRange = determineDepthRangeY( triangle, p, 
                                                         triNormal, modelBBMin, 
                                                         voxelLength );

                        // Ignore coordinates that are outside the subspace.
                        if ( voxRange.x > voxBB.max.y || 
                             voxRange.y < voxBB.min.y )
                            continue;

                        // Adjust range if parts of it are outside the 
                        // subspace.
                        voxRange.x = 
                            voxRange.x < voxBB.min.y ? voxBB.min.y 
                                                     : voxRange.x;
                        voxRange.y = 
                            voxRange.y > voxBB.max.y ? voxBB.max.y 
                                                     : voxRange.y;

                        // No need to perform overlap tests if the range is 
                        // only one voxel thick.
                        if (voxRange.x == voxRange.y)
                        {
                            processVoxel( nodes
                                        , materials
                                        , triIdx
                                        , triangle
                                        , triNormal
                                        , modelBBMin
                                        , voxelLength
                                        , make_int3( i, voxRange.x, k )
                                        , adjs
                                        , gridType
                                        , res
                                        , countVoxels
                                        , hashMap
                                        , surfNodes );
                        }
                        else
                        {
                            // Perform remaining overlap tests on max 3 voxels.
                            for (int j = voxRange.x; j <= voxRange.y; j++)
                            {
                                p.y = getSingleVoxelBoundsComponent( 
                                    j, 
                                    modelBBMin.y, 
                                    voxelLength );

                                if ( overlapTestXY( data[2], p ) )
                                {
                                    if ( overlapTestYZ( data[0], p ) )
                                    {
                                        processVoxel( nodes
                                                    , materials
                                                    , triIdx
                                                    , triangle
                                                    , triNormal
                                                    , modelBBMin
                                                    , voxelLength
                                                    , make_int3( i, j, k )
                                                    , adjs
                                                    , gridType
                                                    , res
                                                    , countVoxels
                                                    , hashMap
                                                    , surfNodes );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (domAxis == zAxis)
        {
            // Loop through the voxel columns of the xy-plane.
            for ( int i = voxBB.min.x; i <= voxBB.max.x; i++ )
            {
                for ( int j = voxBB.min.y; j <= voxBB.max.y; j++ )
                {
                    p = make_float3(
                        getSingleVoxelBoundsComponent( i, 
                                                       modelBBMin.x, 
                                                       voxelLength ), 
                        getSingleVoxelBoundsComponent( j, 
                                                       modelBBMin.y, 
                                                       voxelLength ),
                        0.0f);

                    if ( overlapTestXY( data[2], p ) )
                    {
                        // Determine voxel depth.
                        voxRange = determineDepthRangeZ( triangle, 
                                                         p, 
                                                         triNormal, 
                                                         modelBBMin, 
                                                         voxelLength );

                        // Ignore coordinates that are outside the subspace.
                        if ( voxRange.x > voxBB.max.z || 
                             voxRange.y < voxBB.min.z )
                            continue;

                        // Adjust range if parts of it are outside the 
                        // subspace.
                        voxRange.x = 
                            voxRange.x < voxBB.min.z ? voxBB.min.z 
                                                     : voxRange.x;
                        voxRange.y = 
                            voxRange.y > voxBB.max.z ? voxBB.max.z 
                                                     : voxRange.y;

                        /* No need to perform overlap tests if the range is 
                           only one voxel thick. */
                        if (voxRange.x == voxRange.y)
                        {
                            processVoxel( nodes
                                        , materials
                                        , triIdx
                                        , triangle
                                        , triNormal
                                        , modelBBMin
                                        , voxelLength
                                        , make_int3( i, j, voxRange.x )
                                        , adjs
                                        , gridType
                                        , res
                                        , countVoxels
                                        , hashMap
                                        , surfNodes );
                        }
                        else
                        {
                            // Perform remaining overlap tests on max 3 voxels.
                            for ( int k = voxRange.x; k <= voxRange.y; k++ )
                            {
                                p.z = getSingleVoxelBoundsComponent(
                                        k, 
                                        modelBBMin.z, 
                                        voxelLength );

                                if ( overlapTestYZ( data[0], p ) )
                                {
                                    if ( overlapTestZX( data[1], p ) )
                                    {
                                        processVoxel( nodes
                                                    , materials
                                                    , triIdx
                                                    , triangle
                                                    , triNormal
                                                    , modelBBMin
                                                    , voxelLength
                                                    , make_int3( i, j, k )
                                                    , adjs
                                                    , gridType
                                                    , res
                                                    , countVoxels
                                                    , hashMap
                                                    , surfNodes );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Attempts to find the intersection between the \a subspace (the space this 
/// execution is allowed to operate in) and the space of interesting voxels 
/// (basically the triangle's bounding box converted into voxel coordinates).
/// If such a common space cannot be found, a negative minimum bound is 
/// returned.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ void adjustVoxelRange
    ( Bounds<int3> & voxBB           /**< [in,out] Triangle's bounding box in 
                                                   voxels. */
    , Bounds<uint3> const & subSpace ///< [in] Bounding box of subspace.
    )
{
    Bounds<int3> space = {
        make_int3(subSpace.min),
        make_int3(subSpace.max)
    };

    // Make upper bound inclusive.
    space.max -= 1;

    if ( space.min.x > voxBB.max.x || space.min.y > voxBB.max.y || 
         space.min.z > voxBB.max.z ) {
        voxBB.min = make_int3(-1, -1, -1);
        return;
    }

    if ( voxBB.min.x > space.max.x || voxBB.min.y > space.max.y || 
         voxBB.min.z > space.max.z ) {
        voxBB.min = make_int3(-1, -1, -1);
        return;
    }

    voxBB.min = max( voxBB.min, space.min );
    voxBB.max = min( voxBB.max, space.max );

    return;
}
///////////////////////////////////////////////////////////////////////////////
/// The results are used to limit the amount of voxels that has to be traversed 
/// along the x-direction. It should have been established previously that the 
/// dominant axis of the triangle is also the x-axis.
///
/// The minimum and maximum corners of the voxel are determined and projected 
/// onto the triangle's plane. Their difference is then used to determine the 
/// how thick the triangle is, in terms of how many voxels the plane overlaps.
/// \return Minimum and maximum voxel coordinates of the plane's thickness.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ int2 determineDepthRangeX
    ( float3 * triangle ///< [in] Vertices of the triangle.
    , float3 p          ///< [in] Minimum corner of voxel.
    , float3 triNormal  ///< [in] Triangle normal.
    , float3 modelBBMin /**< [in] Minimum corner of device's voxelization 
                                  space. */
    , float d           ///< [in] Distance between voxel centers.
    )
{
    float3 p_min, p_max;
    float a, b;
    bool ny_pos, nz_pos;

    // false = 0, true = 1 trickery.
    ny_pos = triNormal.y > 0.0f;
    nz_pos = triNormal.z > 0.0f;
    
    // Determine minimum and maximum corners.
    p_min = make_float3( 0.0f, p.y + ny_pos * d, p.z + nz_pos * d );
    p_max = make_float3( 0.0f, p.y + !ny_pos * d, p.z + !nz_pos * d );

    // Determine intersection with triangle's plane.
    a = intersectWithPlaneX( p_min, triangle[0], triNormal );
    b = intersectWithPlaneX( p_max, triangle[0], triNormal );

    // Validate bounds.
    boundsCheck( a, b );

    // Turn into voxel coordinates and return.
    return make_int2( getVoxCBoundMin( a, modelBBMin.x, d )
                    , getVoxCBoundMax( b, modelBBMin.x, d ) );
}
///////////////////////////////////////////////////////////////////////////////
/// The results are used to limit the amount of voxels that has to be traversed 
/// along the y-direction. It should have been established previously that the 
/// dominant axis of the triangle is also the y-axis.
///
/// The minimum and maximum corners of the voxel are determined and projected 
/// onto the triangle's plane. Their difference is then used to determine the 
/// how thick the triangle is, in terms of how many voxels the plane overlaps.
/// \return Minimum and maximum voxel coordinates of the plane's thickness.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ int2 determineDepthRangeY
    ( float3 * triangle ///< [in] Vertices of the triangle.
    , float3 p          ///< [in] Minimum corner of voxel.
    , float3 triNormal  ///< [in] Triangle normal.
    , float3 modelBBMin /**< [in] Minimum corner of device's voxelization 
                                  space. */
    , float d           ///< [in] Distance between voxel centers.
    )
{
    float3 p_min, p_max;
    float a, b;
    bool nx_pos, nz_pos;

    // false = 0, true = 1 trickery.
    nx_pos = triNormal.x > 0.0f;
    nz_pos = triNormal.z > 0.0f;
    
    // Determine minimum and maximum corners.
    p_min = make_float3(p.x + nx_pos * d, 0.0f, p.z + nz_pos * d);
    p_max = make_float3(p.x + !nx_pos * d, 0.0f, p.z + !nz_pos * d);

    // Determine intersection with triangle's plane.
    a = intersectWithPlaneY(p_min, triangle[0], triNormal);
    b = intersectWithPlaneY(p_max, triangle[0], triNormal);

    // Validate bounds.
    boundsCheck(a, b);

    // Turn into voxel coordinates and return.
    return make_int2( getVoxCBoundMin( a, modelBBMin.y, d )
                    , getVoxCBoundMax( b, modelBBMin.y, d ) );
}
///////////////////////////////////////////////////////////////////////////////
/// The results are used to limit the amount of voxels that has to be traversed 
/// along the z-direction. It should have been established previously that the 
/// dominant axis of the triangle is also the z-axis.
///
/// The minimum and maximum corners of the voxel are determined and projected 
/// onto the triangle's plane. Their difference is then used to determine the 
/// how thick the triangle is, in terms of how many voxels the plane overlaps.
/// \return Minimum and maximum voxel coordinates of the plane's thickness.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ int2 determineDepthRangeZ
    ( float3 * triangle ///< [in] Vertices of the triangle.
    , float3 p          ///< [in] Minimum corner of voxel.
    , float3 triNormal  ///< [in] Triangle normal.
    , float3 modelBBMin /**< [in] Minimum corner of device's voxelization 
                                  space. */
    , float d           ///< [in] Distance between voxel centers.
    )
{
    float3 p_min, p_max;
    float a, b;
    bool nx_pos, ny_pos;

    // false = 0, true = 1 trickery.
    nx_pos = triNormal.x > 0.0f;
    ny_pos = triNormal.y > 0.0f;
    
    // Determine minimum and maximum corners.
    p_min = make_float3(p.x + nx_pos * d, p.y + ny_pos * d, 0.0f);
    p_max = make_float3(p.x + !nx_pos * d, p.y + !ny_pos * d, 0.0f);

    // Determine intersection with triangle's plane.
    a = intersectWithPlaneZ(p_min, triangle[0], triNormal);
    b = intersectWithPlaneZ(p_max, triangle[0], triNormal);

    // Validate bounds.
    boundsCheck(a, b);

    // Turn into voxel coordinates and return.
    return make_int2( getVoxCBoundMin( a, modelBBMin.z, d )
                    , getVoxCBoundMax( b, modelBBMin.z, d ) );
}
///////////////////////////////////////////////////////////////////////////////
/// Projects a ray from \p p along direction \p d onto a plane with normal \p n 
/// and a vertex \p v, which is part of the plane. Used in the voxelizer to 
/// project a voxel center from the yz-plane onto the triangle's plane after 
/// intersection with the triangle has been inferred. The result can be 
/// directly converted to a particular voxel that intersects the triangle.
/// \return The distance from \p p to the \a plane along \p d.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float calculateIntersectionWithPlane
    ( float3 p ///< [in] Starting point of the ray.
    , float3 v ///< [in] Point on the triangle / plane.
    , float3 n ///< [in] Normal of the triangle / plane.
    , float3 d ///< [in] Direction of the ray.
    ) 
{ 
    return dot( v - p, n ) / dot( n, d ); 
}
///////////////////////////////////////////////////////////////////////////////
/// Projects a ray, that is parallel to the x-axis, from \p p to the plane that 
/// is defined by its normal \p n and a point on the plane, \p v. This version 
/// is a little faster to calculate than the more general version. Used in the 
/// voxelizer to project a voxel center from the yz-plane onto the triangle's 
/// plane after intersection with the triangle has been inferred. The result 
/// can be directly converted to a particular voxel that intersects the 
/// triangle.
/// \return The distance from \p p to the \a plane along the x-axis.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float intersectWithPlaneX
    ( float3 p ///< [in] Starting point of the ray.
    , float3 v ///< [in] Point on the triangle / plane.
    , float3 n ///< [in] Normal of the triangle / plane.
    ) 
{ 
    return dot( v - p, n ) / n.x; 
}
///////////////////////////////////////////////////////////////////////////////
/// Projects a ray, that is parallel to the y-axis, from \p p to the plane that 
/// is defined by its normal \p n and a point on the plane, \p v. This version 
/// is a little faster to calculate than the more general version.
/// \return The distance from \p p to the \a plane along the y-axis.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float intersectWithPlaneY
    ( float3 p ///< [in] Starting point of the ray.
    , float3 v ///< [in] Point on the triangle / plane.
    , float3 n ///< [in] Normal of the triangle / plane.
    ) 
{ 
    return dot( v - p, n ) / n.y; 
}
///////////////////////////////////////////////////////////////////////////////
/// Projects a ray, that is parallel to the z-axis, from \p p to the plane that 
/// is defined by its normal \p n and a point on the plane, \p v. This version 
/// is a little faster to calculate than the more general version.
/// \return The distance from \p p to the \a plane along the z-axis.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float intersectWithPlaneZ
    ( float3 p ///< [in] Starting point of the ray.
    , float3 v ///< [in] Point on the triangle / plane.
    , float3 n ///< [in] Normal of the triangle / plane.
    ) 
{ 
    return dot(v - p, n) / n.z; 
}
///////////////////////////////////////////////////////////////////////////////
/// Makes sure that the minimum of a \a bound is less than, or equal to, the 
/// maximum. If this is not the case, the values are swapped.
/// \tparam T Type of the minimum and maximum values. \p T needs to have the 
///           comparison operators defined.
///////////////////////////////////////////////////////////////////////////////
template <class T>
inline __host__ __device__ void boundsCheck
    ( T &min ///< [in,out] Minimum value of the bound.
    , T &max ///< [in,out] Maximum value of the bound.
    )
{
    if (min > max)
    {
        T temp = min;
        min = max;
        max = temp;
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Calculates the \a volume of a \a voxel that has been cut by a \a plane. The 
/// volume returned is the part of the voxel that is \a solid. The algorithm 
/// involves calculating the points where the plane intersects with the voxel 
/// and then determining the polygons that make up the faces of the volume we 
/// are interested in. Then, one point is chosen arbitrarily to represent the 
/// highest point of the volume. Each face of the volume combined with the 
/// highest point form a pyramid, and by calculating the volume of these 
/// pyramids and adding them up, the total volume of the cut voxel is 
/// calculated.
/// 
/// The actual details of the algorithm rely on ordering the vertices of the 
/// voxel in a particular order, depending on which vertex is closest to the 
/// cutting plane. The method is based on a publication by Salama and Kolb 
/// \cite salama-kolb-2005 and an extension of the method by Francis Xavier 
/// Timmes \cite timmes-2013.
/// \return The \a volume of the cut voxel.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float calculateVoxelPlaneIntersectionVolume
    ( float3 * triangle ///< [in] Vertices of the triangle.
    , int3 voxel        ///< [in] Voxel coordinates.
    , float3 triNormal  ///< [in] Triangle normal.
    , float3 modelBBMin ///< [in] Minimum corner of the device's voxel space.
    , float d           ///< [in] Distance between voxel centers.
    )
{
    bool x, y, z;               // Sign of triangle normal components.
    float dh;                   // Half of distance between voxel centers.
    float3 
        vc,                     // Voxel center in world coordinates.
        dx, dy, dz,             // Component vectors of a right-handed system.
        voxVerts[8],            /* Corners of the voxel, defined in a specific
                                   order */
        intersectionPoints[6],  /* Intersection points between triangle and 
                                   voxel. */
        polyFace1[6],           // Face 1 of the cut voxel.
        polyFace2[6],           // Face 2 of the cut voxel.
        polyFace3[6];           // Face 3 of the cut voxel.
    int 
        nrOfIntersectionPoints, /* Number of intersection points between the 
                                   voxel and the triangle. */
        nrOfFace1Points,        // Number of vertices in face 1.
        nrOfFace2Points,        // Number of vertices in face 2. 
        nrOfFace3Points;        // Number of vertices in face 3.

    // Voxel center.
    vc = modelBBMin + d * make_float3( float( voxel.x )
                                     , float( voxel.y )
                                     , float( voxel.z ) );

    // Special cases where the volume is easy to calculate.
    if (abs(triNormal.x) == 1.0f)
        return boxParallelCutVolume(vc.x, triangle[0].x, d);
    if (abs(triNormal.y) == 1.0f)
        return boxParallelCutVolume(vc.y, triangle[0].y, d);
    if (abs(triNormal.z) == 1.0f)
        return boxParallelCutVolume(vc.z, triangle[0].z, d);

    /* Signs of the normal's component vectors. Used to determine the vertex 
       closest to the triangle / plane and also to ensure that the volume 
       is the solid volume and not the non-solid one. A triangle's normals 
       point away from the inside of the object. */
    x = triNormal.x > 0.0f;
    y = triNormal.y > 0.0f;
    z = triNormal.z > 0.0f;

    // Component vectors that form a right-handed system.
    if ((x + y + z) % 2 == 0)
    {
        dx = make_float3(!x * d + x * -d, 0.0f, 0.0f);
        dy = make_float3(0.0f, !y * d + y * -d, 0.0f);
        dz = make_float3(0.0f, 0.0f, !z * d + z * -d);
    }
    else
    {
        dx = make_float3(!x * d + x * -d, 0.0f, 0.0f);
        dz = make_float3(0.0f, !y * d + y * -d, 0.0f);
        dy = make_float3(0.0f, 0.0f, !z * d + z * -d);
    }

    dh = d / 2.0f;

    /* Corner points of the voxel. Index 0 ends up being the closest to the 
       plane, and index 7 the furthest. */
    voxVerts[0] = vc + make_float3( dh * x - dh * !x
                                  , dh * y - dh * !y
                                  , dh * z - dh * !z );
    voxVerts[1] = voxVerts[0] + dx;
    voxVerts[2] = voxVerts[0] + dy;
    voxVerts[3] = voxVerts[0] + dz;
    voxVerts[4] = voxVerts[1] + dz;
    voxVerts[5] = voxVerts[2] + dx;
    voxVerts[6] = voxVerts[3] + dy;
    voxVerts[7] = voxVerts[4] + dy;

    /* Calculates the intersection points between the triangle and the voxel 
       and constructs the faces that make up the irregular polyhedron that 
       represents the volume behind the plane. */
    constructPolyhedronFaces( intersectionPoints
                            , polyFace1
                            , polyFace2
                            , polyFace3
                            , nrOfIntersectionPoints
                            , nrOfFace1Points
                            , nrOfFace2Points
                            , nrOfFace3Points
                            , voxVerts
                            , triangle
                            , triNormal );

    // Calculate the volume of the irregular polyhedron.
    return volumeOfPolyhedron( voxVerts
                             , intersectionPoints
                             , polyFace1
                             , polyFace2
                             , polyFace3
                             , nrOfIntersectionPoints
                             , nrOfFace1Points
                             , nrOfFace2Points
                             , nrOfFace3Points
                             , triNormal );
}
///////////////////////////////////////////////////////////////////////////////
/// Calculates the volume after a voxel has been cut off with a plane that is 
/// parallel to one of the main axes.
/// \return Volume of cut voxel.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float boxParallelCutVolume
    ( float voxCenterComponent /**< [in] X-, Y- or Z-component of the voxel 
                                         center. */
    , float triVertexComponent /**< [in] X-, Y- or Z-component of a vertex of 
                                         the triangle. */
    , float d                  ///< [in] Distance between voxel centers.
    )
{
    float base, height;

    base = d * d;
    height = (d / 2.0f) + triVertexComponent - voxCenterComponent;

    return base * height;
}
///////////////////////////////////////////////////////////////////////////////
/// Constructs the \a faces of the <em>irregular polyhedron</em> that remains 
/// after cutting the voxel. The faces are constructed as the algorithm 
/// traverses the edges of the voxel in a particular order and tests for 
/// overlap between the edges and the plane.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ void constructPolyhedronFaces
    ( float3 * iPts     ///< [out] The intersection points.
    , float3 * face1    ///< [out] Vertices for face 1.
    , float3 * face2    ///< [out] Vertices for face 2.
    , float3 * face3    ///< [out] Vertices for face 3.
    , int &nrOfIPts     ///< [out] Number of intersection points.
    , int &nrF1         ///< [out] Number of vertices in face 1.
    , int &nrF2         ///< [out] Number of vertices in face 2.
    , int &nrF3         ///< [out] Number of vertices in face 3.
    , float3 * voxVerts ///< [in] Vertices of the voxel.
    , float3 * triangle ///< [in] Vertices of the triangle.
    , float3 triNormal  ///< [in] Triangle normal.
    )
{
    float t;

    nrOfIPts = 0;
    nrF1 = 0;
    nrF2 = 0;
    nrF3 = 0;

    // Path 1: v0 -> v1
    t = dot( triNormal
           , triangle[0] - voxVerts[0] ) / dot( triNormal
                                              , voxVerts[1] - voxVerts[0] );
    if (t < 0.0f || t > 1.0f)
    {
        // Add v1 to the vertex list of face1.
        face1[nrF1] = voxVerts[1];
        nrF1++;

        // Path 1: v1 -> v4
        t = dot(triNormal, triangle[0] - voxVerts[1]) / dot(triNormal, voxVerts[4] - voxVerts[1]);
        if (t < 0.0f || t > 1.0f)
        {
            // Path 1: v4 -> v7
            t = dot(triNormal, triangle[0] - voxVerts[4]) / dot(triNormal, voxVerts[7] - voxVerts[4]);
            if (t >= 0.0f && t <= 1.0f)
            {
                iPts[nrOfIPts] = voxVerts[4] + t * (voxVerts[7] - voxVerts[4]);
                // Add v4 and the intersection point to the vertex list of face1.
                face1[nrF1] = voxVerts[4];
                nrF1++;
                face1[nrF1] = iPts[nrOfIPts];
                nrF1++;
                // Add v4 and the intersection point in reverse order to the vertex list of face3.
                face3[nrF3] = iPts[nrOfIPts];
                nrF3++;
                face3[nrF3] = voxVerts[4];
                nrF3++;
                nrOfIPts++;
            }
            else
            {
                // Add v4 to the vertex lists of face1 and face3.
                face1[nrF1] = voxVerts[4];
                nrF1++;
                face3[nrF3] = voxVerts[4];
                nrF3++;
            }
        }
        else
        {
            iPts[nrOfIPts] = voxVerts[1] + t * (voxVerts[4] - voxVerts[1]);
            face1[nrF1] = iPts[nrOfIPts];
            nrOfIPts++;
            nrF1++;
        }
    }
    else
    {
        iPts[nrOfIPts] = voxVerts[0] + t * (voxVerts[1] - voxVerts[0]);
        nrOfIPts++;
    }

    // Path 1: v1 -> v5
    t = dot(triNormal, triangle[0] - voxVerts[1]) / dot(triNormal, voxVerts[5] - voxVerts[1]);
    if (t >= 0.0f && t <= 1.0f)
    {
        iPts[nrOfIPts] = voxVerts[1] + t * (voxVerts[5] - voxVerts[1]);
        face1[nrF1] = iPts[nrOfIPts];
        nrF1++;
        nrOfIPts++;
    }

    // Path 2: v0 -> v2
    t = dot(triNormal, triangle[0] - voxVerts[0]) / dot(triNormal, voxVerts[2] - voxVerts[0]);
    if (t < 0.0f || t > 1.0f)
    {
        // Add v2 to the vertex list of face2.
        face2[nrF2] = voxVerts[2];
        nrF2++;

        // Path 2: v2 -> v5
        t = dot(triNormal, triangle[0] - voxVerts[2]) / dot(triNormal, voxVerts[5] - voxVerts[2]);
        if (t < 0.0f || t > 1.0f)
        {
            // Path 2: v5 -> v7
            t = dot(triNormal, triangle[0] - voxVerts[5]) / dot(triNormal, voxVerts[7] - voxVerts[5]);
            if (t >= 0.0f && t <= 1.0f)
            {
                iPts[nrOfIPts] = voxVerts[5] + t * (voxVerts[7] - voxVerts[5]);
                // Add v5 and the intersection point to the vertex list of face2.
                face2[nrF2] = voxVerts[5];
                nrF2++;
                face2[nrF2] = iPts[nrOfIPts];
                nrF2++;
                // Add v5 and the intersection point in reverse order to the vertex list of face1.
                face1[nrF1] = iPts[nrOfIPts];
                nrF1++;
                face1[nrF1] = voxVerts[5];
                nrF1++;
                nrOfIPts++;
            }
            else
            {
                // Add v5 to the vertex lists of face2 and face1.
                face2[nrF2] = voxVerts[5];
                nrF2++;
                face1[nrF1] = voxVerts[5];
                nrF1++;
            }
        }
        else
        {
            iPts[nrOfIPts] = voxVerts[2] + t * (voxVerts[5] - voxVerts[2]);
            face2[nrF2] = iPts[nrOfIPts];
            nrOfIPts++;
            nrF2++;
        }
    }
    else
    {
        iPts[nrOfIPts] = voxVerts[0] + t * (voxVerts[2] - voxVerts[0]);
        nrOfIPts++;
    }

    // Path 2: v2 -> v6
    t = dot(triNormal, triangle[0] - voxVerts[2]) / dot(triNormal, voxVerts[6] - voxVerts[2]);
    if (t >= 0.0f && t <= 1.0f)
    {
        iPts[nrOfIPts] = voxVerts[2] + t * (voxVerts[6] - voxVerts[2]);
        face2[nrF2] = iPts[nrOfIPts];
        nrF2++;
        nrOfIPts++;
    }

    // Path 3: v0 -> v3
    t = dot(triNormal, triangle[0] - voxVerts[0]) / dot(triNormal, voxVerts[3] - voxVerts[0]);
    if (t < 0.0f || t > 1.0f)
    {
        // Add v3 to the vertex list of face3.
        face3[nrF3] = voxVerts[3];
        nrF3++;

        // Path 3: v3 -> v6
        t = dot(triNormal, triangle[0] - voxVerts[3]) / dot(triNormal, voxVerts[6] - voxVerts[3]);
        if (t < 0.0f || t > 1.0f)
        {
            // Path 3: v6 -> v7
            t = dot(triNormal, triangle[0] - voxVerts[6]) / dot(triNormal, voxVerts[7] - voxVerts[6]);
            if (t >= 0.0f && t <= 1.0f)
            {
                iPts[nrOfIPts] = voxVerts[6] + t * (voxVerts[7] - voxVerts[6]);
                // Add v6 and the intersection point to the vertex list of face3.
                face3[nrF3] = voxVerts[6];
                nrF3++;
                face3[nrF3] = iPts[nrOfIPts];
                nrF3++;
                // Add v6 and the intersection point in reverse order to the vertex list of face2.
                face2[nrF2] = iPts[nrOfIPts];
                nrF2++;
                face2[nrF2] = voxVerts[6];
                nrF2++;
                nrOfIPts++;
            }
            else
            {
                // Add v6 to the vertex lists of face3 and face2.
                face3[nrF3] = voxVerts[6];
                nrF3++;
                face2[nrF2] = voxVerts[6];
                nrF2++;
            }
        }
        else
        {
            iPts[nrOfIPts] = voxVerts[3] + t * (voxVerts[6] - voxVerts[3]);
            face3[nrF3] = iPts[nrOfIPts];
            nrOfIPts++;
            nrF3++;
        }
    }
    else
    {
        iPts[nrOfIPts] = voxVerts[0] + t * (voxVerts[3] - voxVerts[0]);
        nrOfIPts++;
    }

    // Path 3: v3 -> v4
    t = dot(triNormal, triangle[0] - voxVerts[3]) / dot(triNormal, voxVerts[4] - voxVerts[3]);
    if (t >= 0.0f && t <= 1.0f)
    {
        iPts[nrOfIPts] = voxVerts[3] + t * (voxVerts[4] - voxVerts[3]);
        face3[nrF3] = iPts[nrOfIPts];
        nrF3++;
        nrOfIPts++;
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Defines vertex 0 of the voxel as the tip of the pyramid. Then, the vertices 
/// of each of the involved faces (up to 4) and the tip of the pyramid are 
/// \a rotated so that the face is parallel to the xy-plane. After that, the 
/// volume of the resulting pyramids are calculated and summed up. The sum is 
/// the volume of the polyhedron. Only four faces of the polyhedron need to be 
/// considered as the volume of the others will end up being zero due to the 
/// choice of tip for the pyramids.
/// \return Volume of the polyhedron.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float volumeOfPolyhedron
    ( float3 * voxVerts ///< [in] Vertices of the voxel.
    , float3 * iPts     /**< [in] Set of intersection points between the plane 
                                  and voxel. */
    , float3 * face1    /**< [in] Set of vertices of the first face of the 
                             polyhedron. */ 
    , float3 * face2    /**< [in] Set of vertices of the second face of the 
                             polyhedron. */ 
    , float3 * face3    /**< [in] Set of vertices of the third face of the 
                             polyhedron. */ 
    , int nrOfIPts      /**< [in] Number of vertices in the face made up 
                                  entirely of the intersection points. */
    , int nrF1          ///< [in] Number of vertices in face 1.
    , int nrF2          ///< [in] Number of vertices in face 2.
    , int nrF3          ///< [in] Number of vertices in face 3.
    , float3 triNormal  ///< [in] Triangle normal.
    )
{
    float pl, sin_t, cos_t, volume;
    float3 rotN, height;

    volume = 0.0f;

    // Rotate the intersection polygon.
    rotN = triNormal;
    height = voxVerts[0];
    
    if (rotN.y != 0.0f)
    {
        pl = length(make_float2(rotN.y, rotN.z));
        sin_t = rotN.y / pl;
        cos_t = rotN.z / pl;

        height = rotX(height, sin_t, cos_t);

        for (int i = 0; i < nrOfIPts; i++)
            iPts[i] = rotX(iPts[i], sin_t, cos_t);

        rotN = rotX(rotN, sin_t, cos_t);
    }

    if (rotN.x != 0.0f)
    {
        pl = length(make_float2(rotN.z, rotN.x));
        sin_t = -rotN.x / pl;
        cos_t = rotN.z / pl;

        height = rotY(height, sin_t, cos_t);

        for (int i = 0; i < nrOfIPts; i++)
            iPts[i] = rotY(iPts[i], sin_t, cos_t);
    }

    volume += volumeOfPyramid(iPts, nrOfIPts, height);

    // Rotate the face1 polygon.
    if (nrF1 > 2)
    {
        rotN = normalize(cross(face1[0] - face1[2], face1[1] - face1[0]));
        height = voxVerts[0];

        if (rotN.y != 0.0f)
        {
            pl = length(make_float2(rotN.y, rotN.z));
            sin_t = rotN.y / pl;
            cos_t = rotN.z / pl;

            height = rotX(voxVerts[0], sin_t, cos_t);

            for (int i = 0; i < nrF1; i++)
                face1[i] = rotX(face1[i], sin_t, cos_t);

            rotN = rotX(rotN, sin_t, cos_t);
        }

        if (rotN.x != 0.0f)
        {
            pl = length(make_float2(rotN.z, rotN.x));
            sin_t = -rotN.x / pl;
            cos_t = rotN.z / pl;

            height = rotY(height, sin_t, cos_t);

            for (int i = 0; i < nrF1; i++)
                face1[i] = rotY(face1[i], sin_t, cos_t);
        }

        volume += volumeOfPyramid(face1, nrF1, height);
    }

    // Rotate the face2 polygon.
    if (nrF2 > 2)
    {
        rotN = normalize(cross(face2[0] - face2[2], face2[1] - face2[0]));
        height = voxVerts[0];

        if (rotN.y != 0.0f)
        {
            pl = length(make_float2(rotN.y, rotN.z));
            sin_t = rotN.y / pl;
            cos_t = rotN.z / pl;

            height = rotX(voxVerts[0], sin_t, cos_t);

            for (int i = 0; i < nrF2; i++)
                face2[i] = rotX(face2[i], sin_t, cos_t);

            rotN = rotX(rotN, sin_t, cos_t);
        }

        if (rotN.x != 0.0f)
        {
            pl = length(make_float2(rotN.z, rotN.x));
            sin_t = -rotN.x / pl;
            cos_t = rotN.z / pl;
    
            height = rotY(height, sin_t, cos_t);

            for (int i = 0; i < nrF2; i++)
                face2[i] = rotY(face2[i], sin_t, cos_t);
        }

        volume += volumeOfPyramid(face2, nrF2, height);
    }

    // Rotate the face3 polygon.
    if (nrF3 > 2)
    {
        rotN = normalize(cross(face3[0] - face3[2], face3[1] - face3[0]));
        height = voxVerts[0];

        if (rotN.y != 0.0f)
        {
            pl = length(make_float2(rotN.y, rotN.z));
            sin_t = rotN.y / pl;
            cos_t = rotN.z / pl;

            height = rotX(voxVerts[0], sin_t, cos_t);

            for (int i = 0; i < nrF3; i++)
                face3[i] = rotX(face3[i], sin_t, cos_t);

            rotN = rotX(rotN, sin_t, cos_t);
        }

        if (rotN.x != 0.0f)
        {
            pl = length(make_float2(rotN.z, rotN.x));
            sin_t = -rotN.x / pl;
            cos_t = rotN.z / pl;

            height = rotY(height, sin_t, cos_t);

            for (int i = 0; i < nrF3; i++)
                face3[i] = rotY(face3[i], sin_t, cos_t);
        }

        volume += volumeOfPyramid(face3, nrF3, height);
    }

    return volume;
}
///////////////////////////////////////////////////////////////////////////////
/// The rotation is an application of the following rotation \a matrix: 
/// \f[ \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos \theta & -\sin \theta \\ 0 & \sin 
/// \theta & \cos \theta \end{bmatrix} \cdot \begin{bmatrix} x \\ y \\ z 
/// \end{bmatrix} = \begin{bmatrix} x \\ y \cdot \cos \theta - z \cdot \sin 
/// \theta \\ y \cdot \sin \theta + z \cdot \cos \theta \end{bmatrix} . \f]
/// The rotation is performed counter-clockwise when looking at the negative 
/// direction of the x-axis. The angle is not supplied as as an angle, but 
/// rather as the \a sine and \a cosine of the angle.
/// \return The rotated point.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float3 rotX
    ( float3 v     ///< [in] Point to be rotated.
    , float sine   ///< [in] Sine of the rotation angle.
    , float cosine ///< [in] Cosine of the rotation angle.
    )
{
    float3 result;

    result = make_float3( v.x
                        , v.y * cosine - v.z * sine
                        , v.y * sine + v.z * cosine );

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// The rotation is an application of the following rotation \a matrix: 
/// \f[ \begin{bmatrix} \cos \theta & 0 & \sin \theta \\ 0 & 1 & 0 \\ -\sin 
/// \theta & 0 & \cos \theta \end{bmatrix} \cdot \begin{bmatrix} x \\ y \\ z 
/// \end{bmatrix} = \begin{bmatrix} x \cdot \cos \theta + z \cdot \sin \theta 
/// \\ y \\ z \cdot \cos \theta - x \cdot \sin \theta \end{bmatrix} . \f]
/// The rotation is performed counter-clockwise when looking at the negative 
/// direction of the y-axis. The angle is not supplied as as an angle, but 
/// rather as the \a sine and \a cosine of the angle.
/// \return The rotated point.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float3 rotY
    ( float3 v     ///< [in] Point to be rotated.
    , float sine   ///< [in] Sine of the rotation angle.
    , float cosine ///< [in] Cosine of the rotation angle.
    )
{
    float3 result;

    result = make_float3( v.x * cosine + v.z * sine
                        , v.y
                        , v.z * cosine - v.x * sine );

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Uses the <em>surveyor's algorithm</em> to calculate the volume of the 
/// \a pyramid. The base of the pyramid can have any number of vertices.
/// \return Volume of the pyramid.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float volumeOfPyramid
    ( float3* base  ///< [in] Vertices of the base polygon.
    , int nrOfVerts ///< [in] Number of vertices in the base.
    , float3 height ///< [in] Highest point of the pyramid.
    )
{
    float baseArea, volume;
    float3 v0, v1;

    // Surveyor's algorithm for the area of the base polygon.
    baseArea = 0.0f;
    for (int i = 0; i < nrOfVerts; i++)
    {
        v0 = base[i];
        v1 = base[(i+1) % nrOfVerts];

        baseArea += 0.5f * (v0.x * v1.y - v1.x * v0.y);
    }

    volume = (1.0f / 3.0f) * baseArea * (height.z - base[0].z);
    return abs( volume );
}

/// \tparam Node Type of node used in the array.
template <class Node>
__global__ void zeroPadding
    ( Node * nodes           ///< [out] \p Node array to apply padding to.
    , const uint3 dimensions ///< [in] Dimensions of the \p Node array.
    )
{
    const uint maxNode = dimensions.x * dimensions.y * dimensions.z;
    for ( uint n = blockIdx.x * blockDim.x + threadIdx.x; 
          n < maxNode; 
          n += gridDim.x * blockDim.x )
    {
        const uint x = n % dimensions.x;
        const uint y = (n % (dimensions.x * dimensions.y)) / dimensions.x;
        const uint z = n / (dimensions.x * dimensions.y);

        if ( Node::isFCCNode() )
        {
            if ( x < 2 || x > dimensions.x - 3 ||
                 y == 0 || y == dimensions.y - 1 || 
                 z < 2 || z > dimensions.z - 3 )
            {
                nodes[n] = Node();
            }
        }
        else
        {
            if ( x == 0 || x == dimensions.x - 1 || 
                 y == 0 || y == dimensions.y - 1 || 
                 z == 0 || z == dimensions.z - 1 ) 
            {
                nodes[n] = Node();
            }
        }
    }
}

__device__ float calculateCutVolumeAndAreas
    ( float3 * triangle
    , int3 voxel
    , float3 triNormal
    , float3 modelBBMin
    , float d
    , float & ipArea
    , float & f1Area
    , float & f2Area
    , float & f3Area
    , float & f4Area
    , float & f5Area
    , float & f6Area
    )
{
    // Voxel center.
    float3 vc = modelBBMin + d * make_float3( float( voxel.x )
                                            , float( voxel.y )
                                            , float( voxel.z ) );

    // Signs of the normal's component vectors. Used to determine the vertex 
    // closest to the triangle / plane and also to ensure that the volume 
    // is the solid volume and not the non-solid one. A triangle's normals 
    // point away from the inside of the object.

    bool x, y, z;               // Sign of triangle normal components.
    x = -triNormal.x > 0.0f;
    y = -triNormal.y > 0.0f;
    z = -triNormal.z > 0.0f;

    float3 dx, dy, dz;

    // Component vectors that form a right-handed system.
    if ((x + y + z) % 2 == 0)
    {
        dx = make_float3(!x * d + x * -d, 0.0f, 0.0f);
        dy = make_float3(0.0f, !y * d + y * -d, 0.0f);
        dz = make_float3(0.0f, 0.0f, !z * d + z * -d);
    }
    else
    {
        dx = make_float3(!x * d + x * -d, 0.0f, 0.0f);
        dz = make_float3(0.0f, !y * d + y * -d, 0.0f);
        dy = make_float3(0.0f, 0.0f, !z * d + z * -d);
    }

    float dh = d / 2.0f; // Half of distance between voxel centers.

    float3 vertices[14];
    char indices[48] = {0};

    // Corner points of the voxel. Index 0 ends up being the closest to the 
    // plane, and index 7 the furthest.
    vertices[0] = vc + make_float3( dh * x - dh * !x
                                  , dh * y - dh * !y
                                  , dh * z - dh * !z );
    vertices[1] = vertices[0] + dx;
    vertices[2] = vertices[0] + dy;
    vertices[3] = vertices[0] + dz;
    vertices[4] = vertices[1] + dz;
    vertices[5] = vertices[2] + dx;
    vertices[6] = vertices[3] + dy;
    vertices[7] = vertices[4] + dy;

    // Calculates the intersection points between the triangle and the voxel 
    // and constructs the faces that make up the irregular polyhedron that 
    // represents the volume behind the plane.
    char nrOfIntersectionPoints = 0,
                nrOfFace1Points = 0,
                nrOfFace2Points = 0,
                nrOfFace3Points = 0,
                nrOfFace4Points = 0,
                nrOfFace5Points = 0,
                nrOfFace6Points = 0;
    constructAllPolyhedronFaces( vertices
                               , indices
                               , nrOfIntersectionPoints
                               , nrOfFace1Points
                               , nrOfFace2Points
                               , nrOfFace3Points
                               , nrOfFace4Points
                               , nrOfFace5Points
                               , nrOfFace6Points
                               , triangle
                               , -triNormal );

    if ( nrOfIntersectionPoints == 0 )
    {
        float dist = dot( triNormal, vertices[0] - triangle[0] );

        bool v = dist <= 0.0f;

        float maxArea = d * d;

        ipArea = 0.0f;
        f1Area = v ? maxArea : 0.0f;
        f2Area = v ? maxArea : 0.0f;
        f3Area = v ? maxArea : 0.0f;
        f4Area = v ? maxArea : 0.0f;
        f5Area = v ? maxArea : 0.0f;
        f6Area = v ? maxArea : 0.0f;

        return v ? maxArea * d : 0.0f;
    }

    // Calculate area of face 4.
    float3 base[8];
    float3 height;
    if (nrOfFace4Points > 2)
    {
        for ( int i = 0; i < nrOfFace4Points; ++i )
        {
            float3 v = vertices[indices[8*3+i]];

            base[i].x = dz.x != 0.0f ? -v.z : v.x;
            base[i].y = dz.y != 0.0f ? -v.z : v.y;

            base[i].z = 0.0f;
            base[i].z += dz.x != 0.0f ? v.x : 0.0f;
            base[i].z += dz.y != 0.0f ? v.y : 0.0f;
            base[i].z += dz.z != 0.0f ? v.z : 0.0f;
        }

        height.x = dz.x != 0.0f ? -vertices[0].z : vertices[0].x;
        height.y = dz.y != 0.0f ? -vertices[0].z : vertices[0].y;

        height.z = 0.0f;
        height.z += dz.x != 0.0f ? vertices[0].x : 0.0f;
        height.z += dz.y != 0.0f ? vertices[0].y : 0.0f;
        height.z += dz.z != 0.0f ? vertices[0].z : 0.0f;

        f4Area = polygonArea( base, nrOfFace4Points );
    }

    // Calculate area of face 5.
    if (nrOfFace5Points > 2)
    {
        for ( int i = 0; i < nrOfFace5Points; ++i )
        {
            float3 v = vertices[indices[8*4+i]];

            base[i].x = dy.x != 0.0f ? -v.z : v.x;
            base[i].y = dy.y != 0.0f ? -v.z : v.y;

            base[i].z = 0.0f;
            base[i].z += dy.x != 0.0f ? v.x : 0.0f;
            base[i].z += dy.y != 0.0f ? v.y : 0.0f;
            base[i].z += dy.z != 0.0f ? v.z : 0.0f;
        }

        height.x = dy.x != 0.0f ? -vertices[0].z : vertices[0].x;
        height.y = dy.y != 0.0f ? -vertices[0].z : vertices[0].y;

        height.z = 0.0f;
        height.z += dy.x != 0.0f ? vertices[0].x : 0.0f;
        height.z += dy.y != 0.0f ? vertices[0].y : 0.0f;
        height.z += dy.z != 0.0f ? vertices[0].z : 0.0f;

        f5Area = polygonArea( base, nrOfFace5Points );
    }

    // Calculate area of face 6.
    if (nrOfFace6Points > 2)
    {
        for ( int i = 0; i < nrOfFace6Points; ++i )
        {
            float3 v = vertices[indices[8*5+i]];

            base[i].x = dx.x != 0.0f ? -v.z : v.x;
            base[i].y = dx.y != 0.0f ? -v.z : v.y;

            base[i].z = 0.0f;
            base[i].z += dx.x != 0.0f ? v.x : 0.0f;
            base[i].z += dx.y != 0.0f ? v.y : 0.0f;
            base[i].z += dx.z != 0.0f ? v.z : 0.0f;
        }

        height.x = dx.x != 0.0f ? -vertices[0].z : vertices[0].x;
        height.y = dx.y != 0.0f ? -vertices[0].z : vertices[0].y;

        height.z = 0.0f;
        height.z += dx.x != 0.0f ? vertices[0].x : 0.0f;
        height.z += dx.y != 0.0f ? vertices[0].y : 0.0f;
        height.z += dx.z != 0.0f ? vertices[0].z : 0.0f;

        f6Area = polygonArea( base, nrOfFace6Points );
    }

    
    // Calculate the volume of the irregular polyhedron.
    return polyhedronVolume( vertices
                             , indices
                             , nrOfIntersectionPoints
                             , nrOfFace1Points
                             , nrOfFace2Points
                             , nrOfFace3Points
                             , -triNormal
                             , dx
                             , dy
                             , dz
                             , ipArea
                             , f1Area
                             , f2Area
                             , f3Area );
}

__device__ void constructAllPolyhedronFaces
    ( float3 * vertices
    , char      * indices
    , char      & nrOfIPts  ///< [out] Number of intersection points.
    , char      & nrF1      ///< [out] Number of vertices in face 1.
    , char      & nrF2      ///< [out] Number of vertices in face 2.
    , char      & nrF3      ///< [out] Number of vertices in face 3.
    , char      & nrF4      ///< [out] Number of vertices in face 4.
    , char      & nrF5      ///< [out] Number of vertices in face 5.
    , char      & nrF6      ///< [out] Number of vertices in face 6.
    , float3 * triangle  ///< [in] Vertices of the triangle.
    , float3   triNormal ///< [in] Triangle normal.
    )
{
    float t; // Intersection point along an edge.

    nrOfIPts = 0;
    nrF1 = 0;
    nrF2 = 0;
    nrF3 = 0;
    nrF4 = 0;
    nrF5 = 0;
    nrF6 = 0;

    indices[ 8*3 + nrF4 ] = 0;
    nrF4++;
    indices[ 8*5 + nrF6 ] = 0;
    nrF6++;

    // Path 1: v0 -> v1.
    t = dot( triNormal, triangle[0] - vertices[0] ) / 
        dot( triNormal, vertices[1] - vertices[0] );
    t = isfinite(t) ? t : -1.0f;
    if (t < 0.0f || t > 1.0f) // No intersection.
    {
        indices[ nrF1 ] = 1;
        nrF1++;
        indices[ 8*3 + nrF4 ] = 1;
        nrF4++;

        // Path 1: v1 -> v4
        t = dot( triNormal, triangle[0] - vertices[1] ) / 
            dot( triNormal, vertices[4] - vertices[1] );
        t = isfinite(t) ? t : -1.0f;
        if (t < 0.0f || t > 1.0f) // No intersection.
        {
            indices[ nrF1 ] = 4;
            nrF1++;
            indices[ 8*4 + nrF5 ] = 4;
            nrF5++;
            indices[ 8*4 + nrF5 ] = 1;
            nrF5++;

            // Path 1: v4 -> v7
            t = dot( triNormal, triangle[0] - vertices[4] ) / 
                dot( triNormal, vertices[7] - vertices[4] );
            t = isfinite(t) ? t : -1.0f;
            if (t < 0.0f || t > 1.0f) // No intersection.
            {
                /*
                indices[ nrF1 ] = 7;
                nrF1++;
                indices[ 8*1 + nrF2 ] = 7;
                nrF2++;
                indices[ 8*2 + nrF3 ] = 7;
                nrF3++;
                indices[ 8*2 + nrF3 ] = 4;
                nrF3++;
                */

                nrOfIPts = 0;
                return;
            }
            else // Intersection in path 1, v4 -> v7.
            {
                vertices[ 8 + nrOfIPts ] = vertices[4] + t * (vertices[7] - vertices[4]);

                indices[ nrF1 ] = 8 + nrOfIPts;
                nrF1++;
                indices[ 8*2 + nrF3 ] = 8 + nrOfIPts;
                nrF3++;
                indices[ 8*2 + nrF3 ] = 4;
                nrF3++;

                nrOfIPts++;
                
            }
        }
        else // Intersection in path 1, v1 -> v4.
        {
            vertices[ 8 + nrOfIPts ] = vertices[1] + t * (vertices[4] - vertices[1]);

            indices[ nrF1 ] = 8 + nrOfIPts;
            nrF1++;
            indices[ 8*4 + nrF5 ] = 8 + nrOfIPts;
            nrF5++;
            indices[ 8*4 + nrF5 ] = 1;
            nrF5++;

            nrOfIPts++;
        }
    }
    else // Intersection in path 1, v0 -> v1.
    {
        vertices[ 8 + nrOfIPts ] = vertices[0] + t * (vertices[1] - vertices[0]);

        indices[ 8*3 + nrF4 ] = 8 + nrOfIPts;
        nrF4++;
        indices[ 8*4 + nrF5 ] = 8 + nrOfIPts;
        nrF5++;

        nrOfIPts++;
    }

    // Path 1: v1 -> v5 (Extra)
    t = dot( triNormal, triangle[0] - vertices[1] ) / 
        dot( triNormal, vertices[5] - vertices[1] );
    t = isfinite(t) ? t : -1.0f;
    if (t >= 0.0f && t <= 1.0f) // Intersection
    {
        vertices[ 8 + nrOfIPts ] = vertices[1] + t * (vertices[5] - vertices[1]);

        indices[ nrF1 ] = 8 + nrOfIPts;
        nrF1++;
        indices[ 8*3 + nrF4 ] = 8 + nrOfIPts;
        nrF4++;

        nrOfIPts++;
    }

    indices[ 8*4 + nrF5 ] = 0;
    nrF5++;

    // Path 2: v0 -> v2
    t = dot( triNormal, triangle[0] - vertices[0] ) / 
        dot( triNormal, vertices[2] - vertices[0] );
    t = isfinite(t) ? t : -1.0f;
    if (t < 0.0f || t > 1.0f) // No intersection
    {
        indices[ 8*1 + nrF2 ] = 2;
        nrF2++;
        indices[ 8*5 + nrF6 ] = 2;
        nrF6++;

        // Path 2: v2 -> v5
        t = dot( triNormal, triangle[0] - vertices[2] ) / 
            dot( triNormal, vertices[5] - vertices[2] );
        t = isfinite(t) ? t : -1.0f;
        if (t < 0.0f || t > 1.0f) // No intersection
        {
            indices[ 8*1 + nrF2 ] = 5;
            nrF2++;
            indices[ 8*3 + nrF4 ] = 5;
            nrF4++;
            indices[ 8*3 + nrF4 ] = 2;
            nrF4++;

            // Path 2: v5 -> v7
            t = dot( triNormal, triangle[0] - vertices[5] ) / 
                dot( triNormal, vertices[7] - vertices[5] );
            t = isfinite(t) ? t : -1.0f;
            if (t < 0.0f || t > 1.0f) // No intersection
            {
                indices[ nrF1 ] = 7;
                nrF1++;
                indices[ nrF1 ] = 5;
                nrF1++;
                indices[ 8*1 + nrF2 ] = 7;
                nrF2++;
                indices[ 8*2 + nrF3 ] = 7;
                nrF3++;
            }
            else // Intersection
            {
                vertices[ 8 + nrOfIPts ] = vertices[5] + t * (vertices[7] - vertices[5]);

                indices[ nrF1 ] = 8 + nrOfIPts;
                nrF1++;
                indices[ nrF1 ] = 5;
                nrF1++;
                indices[ 8*1 + nrF2 ] = 8 + nrOfIPts;
                nrF2++;

                nrOfIPts++;
            }
        }
        else // Intersection in path 2, v2 -> v5
        {
            vertices[ 8 + nrOfIPts ] = vertices[2] + t * (vertices[5] - vertices[2]);

            indices[ 8*1 + nrF2 ] = 8 + nrOfIPts;
            nrF2++;
            indices[ 8*3 + nrF4 ] = 8 + nrOfIPts;
            nrF4++;
            indices[ 8*3 + nrF4 ] = 2;
            nrF4++;

            nrOfIPts++;
        }
    }
    else // Intersection in path 2, v0 -> v2
    {
        vertices[ 8 + nrOfIPts ] = vertices[0] + t * (vertices[2] - vertices[0]);

        indices[ 8*3 + nrF4 ] = 8 + nrOfIPts;
        nrF4++;
        indices[ 8*5 + nrF6 ] = 8 + nrOfIPts;
        nrF6++;

        nrOfIPts++;
    }

    // Path 2: v2 -> v6 (Extra)
    t = dot( triNormal, triangle[0] - vertices[2] ) / 
        dot( triNormal, vertices[6] - vertices[2] );
    t = isfinite(t) ? t : -1.0f;
    if (t >= 0.0f && t <= 1.0f)
    {
        vertices[ 8 + nrOfIPts ] = vertices[2] + t * (vertices[6] - vertices[2]);

        indices[ 8*1 + nrF2 ] = 8 + nrOfIPts;
        nrF2++;
        indices[ 8*5 + nrF6 ] = 8 + nrOfIPts;
        nrF6++;

        nrOfIPts++;
    }

    // Path 3: v0 -> v3
    t = dot( triNormal, triangle[0] - vertices[0] ) / 
        dot( triNormal, vertices[3] - vertices[0] );
    t = isfinite(t) ? t : -1.0f;
    if (t < 0.0f || t > 1.0f) // No intersection
    {
        indices[ 8*2 + nrF3 ] = 3;
        nrF3++;
        indices[ 8*4 + nrF5 ] = 3;
        nrF5++;

        // Path 3: v3 -> v6
        t = dot( triNormal, triangle[0] - vertices[3] ) / 
            dot( triNormal, vertices[6] - vertices[3] );
        t = isfinite(t) ? t : -1.0f;
        if (t < 0.0f || t > 1.0f) // No intersection
        {
            indices[ 8*2 + nrF3 ] = 6;
            nrF3++;
            indices[ 8*5 + nrF6 ] = 6;
            nrF6++;
            indices[ 8*5 + nrF6 ] = 3;
            nrF6++;

            // Path 3: v6 -> v7
            t = dot( triNormal, triangle[0] - vertices[6] ) / 
                dot(triNormal, vertices[7] - vertices[6] );
            t = isfinite(t) ? t : -1.0f;
            if (t < 0.0f || t > 1.0f) // No intersection
            {
                indices[ nrF1 ] = 7;
                nrF1++;
                indices[ 8*1 + nrF2 ] = 7;
                nrF2++;
                indices[ 8*1 + nrF2 ] = 6;
                nrF2++;
                indices[ 8*2 + nrF3 ] = 7;
                nrF3++;
            }
            else // Intersection
            {
                vertices[ 8 + nrOfIPts ] = vertices[6] + t * (vertices[7] - vertices[6]);

                indices[ 8*1 + nrF2 ] = 8 + nrOfIPts;
                nrF2++;
                indices[ 8*1 + nrF2 ] = 6;
                nrF2++;
                indices[ 8*2 + nrF3 ] = 8 + nrOfIPts;
                nrF3++;

                nrOfIPts++;
            }
        }
        else // Intersection in path 3, v3 -> v6
        {
            vertices[ 8 + nrOfIPts ] = vertices[3] + t * (vertices[6] - vertices[3]);

            indices[ 8*2 + nrF3 ] = 8 + nrOfIPts;
            nrF3++;
            indices[ 8*5 + nrF6 ] = 8 + nrOfIPts;
            nrF6++;
            indices[ 8*5 + nrF6 ] = 3;
            nrF6++;

            nrOfIPts++;
        }
    }
    else // Intersection in path 3, v0 -> v3
    {
        vertices[ 8 + nrOfIPts ] = vertices[0] + t * (vertices[3] - vertices[0]);

        indices[ 8*4 + nrF5 ] = 8 + nrOfIPts;
        nrF5++;
        indices[ 8*5 + nrF6 ] = 8 + nrOfIPts;
        nrF6++;

        nrOfIPts++;
    }

    // Path 3: v3 -> v4 (Extra)
    t = dot( triNormal, triangle[0] - vertices[3] ) / 
        dot( triNormal, vertices[4] - vertices[3] );
    t = isfinite(t) ? t : -1.0f;
    if (t >= 0.0f && t <= 1.0f) // Intersection
    {
        vertices[ 8 + nrOfIPts ] = vertices[3] + t * (vertices[4] - vertices[3]);

        indices[ 8*2 + nrF3 ] = 8 + nrOfIPts;
        nrF3++;
        indices[ 8*4 + nrF5 ] = 8 + nrOfIPts;
        nrF5++;

        nrOfIPts++;
    }
}

__device__ float polyhedronVolume
    ( float3 * vertices
    , char   * indices
    , char     nrOfIPts  ///< [in] Number of vertices in the face made up 
                         ///<      entirely of the intersection points.
                         ///<
    , char     nrF1      ///< [in] Number of vertices in face 1.
    , char     nrF2      ///< [in] Number of vertices in face 2.
    , char     nrF3      ///< [in] Number of vertices in face 3.
    , float3   triNormal ///< [in] Triangle normal.
    , float3 & dx
    , float3 & dy
    , float3 & dz
    , float  & ipArea
    , float  & f1Area
    , float  & f2Area
    , float  & f3Area
    )
{
    float pl, sin_t, cos_t, volume;
    float3 rotN, height;

    volume = 0.0f;

    // Rotate the intersection polygon.
    rotN = triNormal;
    height = vertices[0];
    
    float3 base[6];

    ipArea = 0.0f;
    f1Area = 0.0f;
    f2Area = 0.0f;
    f3Area = 0.0f;

    if ( nrOfIPts < 3 )
        return 0.0f;

    for ( int i = 0; i < nrOfIPts; ++i )
        base[i] = vertices[8 + i];

    if (rotN.y != 0.0f)
    {
        pl = length(make_float2(rotN.y, rotN.z));
        sin_t = rotN.y / pl;
        cos_t = rotN.z / pl;

        height = rotX(height, sin_t, cos_t);

        for (int i = 0; i < nrOfIPts; i++)
            base[i] = rotX(base[i], sin_t, cos_t);

        rotN = rotX(rotN, sin_t, cos_t);
    }

    if (rotN.x != 0.0f)
    {
        pl = length(make_float2(rotN.z, rotN.x));
        sin_t = -rotN.x / pl;
        cos_t = rotN.z / pl;

        height = rotY(height, sin_t, cos_t);

        for (int i = 0; i < nrOfIPts; i++)
            base[i] = rotY(base[i], sin_t, cos_t);
    }

    volume += pyramidVolume( base, nrOfIPts, height, ipArea );

    // Rotate the face1 polygon.
    if (nrF1 > 2)
    {
        for ( int i = 0; i < nrF1; ++i )
        {
            float3 v = vertices[indices[i]];

            base[i].x = dx.x != 0.0f ? -v.z : v.x;
            base[i].y = dx.y != 0.0f ? -v.z : v.y;

            base[i].z = 0.0f;
            base[i].z += dx.x != 0.0f ? v.x : 0.0f;
            base[i].z += dx.y != 0.0f ? v.y : 0.0f;
            base[i].z += dx.z != 0.0f ? v.z : 0.0f;
        }

        height.x = dx.x != 0.0f ? -vertices[0].z : vertices[0].x;
        height.y = dx.y != 0.0f ? -vertices[0].z : vertices[0].y;

        height.z = 0.0f;
        height.z += dx.x != 0.0f ? vertices[0].x : 0.0f;
        height.z += dx.y != 0.0f ? vertices[0].y : 0.0f;
        height.z += dx.z != 0.0f ? vertices[0].z : 0.0f;

        volume += pyramidVolume( base, nrF1, height, f1Area );
    }

    // Rotate the face2 polygon.
    if (nrF2 > 2)
    {
        for ( int i = 0; i < nrF2; ++i )
        {
            float3 v = vertices[indices[8+i]];

            base[i].x = dy.x != 0.0f ? -v.z : v.x;
            base[i].y = dy.y != 0.0f ? -v.z : v.y;

            base[i].z = 0.0f;
            base[i].z += dy.x != 0.0f ? v.x : 0.0f;
            base[i].z += dy.y != 0.0f ? v.y : 0.0f;
            base[i].z += dy.z != 0.0f ? v.z : 0.0f;
        }

        height.x = dy.x != 0.0f ? -vertices[0].z : vertices[0].x;
        height.y = dy.y != 0.0f ? -vertices[0].z : vertices[0].y;

        height.z = 0.0f;
        height.z += dy.x != 0.0f ? vertices[0].x : 0.0f;
        height.z += dy.y != 0.0f ? vertices[0].y : 0.0f;
        height.z += dy.z != 0.0f ? vertices[0].z : 0.0f;

        volume += pyramidVolume( base, nrF2, height, f2Area );
    }

    // Rotate the face3 polygon.
    if (nrF3 > 2)
    {
        for ( int i = 0; i < nrF3; ++i )
        {
            float3 v = vertices[indices[8*2+i]];

            base[i].x = dz.x != 0.0f ? -v.z : v.x;
            base[i].y = dz.y != 0.0f ? -v.z : v.y;

            base[i].z = 0.0f;
            base[i].z += dz.x != 0.0f ? v.x : 0.0f;
            base[i].z += dz.y != 0.0f ? v.y : 0.0f;
            base[i].z += dz.z != 0.0f ? v.z : 0.0f;
        }

        height.x = dz.x != 0.0f ? -vertices[0].z : vertices[0].x;
        height.y = dz.y != 0.0f ? -vertices[0].z : vertices[0].y;

        height.z = 0.0f;
        height.z += dz.x != 0.0f ? vertices[0].x : 0.0f;
        height.z += dz.y != 0.0f ? vertices[0].y : 0.0f;
        height.z += dz.z != 0.0f ? vertices[0].z : 0.0f;

        volume += pyramidVolume( base, nrF3, height, f3Area );
    }

    return volume;
}

__device__ float pyramidVolume
    ( float3 * base ///< [in] Vertices of the base polygon.
    , int nrOfVerts    ///< [in] Number of vertices in the base.
    , float3 height ///< [in] Highest point of the pyramid.
    , float & baseArea
    )
{
    baseArea = polygonArea( base, nrOfVerts );

    float volume = (1.0f / 3.0f) * baseArea * (height.z - base[0].z);
    return fabs( volume );
}

__device__ float polygonArea
    ( float3 * base
    , int nrOfVerts
    )
{
    // Surveyor's algorithm for the area of the base polygon.
    float baseArea = 0.0f;
    for (int i = 0; i < nrOfVerts; i++)
    {
        float3 v0 = base[i];
        float3 v1 = base[(i+1) % nrOfVerts];

        baseArea += 0.5f * (v0.x * v1.y - v1.x * v0.y);
    }

    return baseArea;
}

///////////////////////////////////////////////////////////////////////////////
///
///////////////////////////////////////////////////////////////////////////////
template <class Node>
__global__ void fillHashMap
    ( Node * nodes
    , HashMap map
    , const uint3 dim
    )
{
    // A single block is running with 1024 threads.
    // The threads are grouped into 32 warps => (32,32)-configuration.
    // <<< 1, dim3(32,32,1) >>>
    // 32 unsigned integers are required to keep track of inter-warp 
    // highest indices.
    // 32 bools are required to keep track of the status of each warp.

    // threadIdx.x & threadIdx.y = indices into the (32,32)-config.
    // blockDim.x & blockDim.y = 32 each
    // gridDim.x = 1
    // Define threadIdx.x as the warp index and threadIdx.y as the thread idx.

    __shared__ uint index[33];

    const uint maxNodeIdx = dim.x * dim.y * dim.z;
    const uint threads = blockDim.x * blockDim.y;
    const uint maxN = maxNodeIdx + (threads - maxNodeIdx % threads);

    index[threadIdx.y + 1] = 0;
    if ( threadIdx.x == 0 && threadIdx.y == 0 ) index[0] = 0;

    for ( uint n = blockDim.x * threadIdx.y + threadIdx.x
        ; n < maxN
        ; n += blockDim.x * blockDim.y )
    {
        __syncthreads();

        uint count = UINT_MAX;

        if ( n < maxNodeIdx )
        {
            const uint nodeType = uint(nodes[n].bid()); 
            const uint ballot = __ballot( nodeType );

            if ( ballot == 0u )
            {
                index[threadIdx.y + 1] = 0;
            }
            else if ( ((1u << threadIdx.x) & ballot) > 0 )
            {
                bool last = true;
                count = 0;
                for ( int i = 0; i < 32; ++i )
                {
                    const bool p = i > threadIdx.x;
                    const bool isect = ((1u << i) & ballot) > 0;
                    count += isect ? !p : 0;
                    last = p && isect ? false : last;
                }

                if ( last ) index[threadIdx.y + 1] = count;
                count -= 1;
            }
        }

        __syncthreads();

        if ( threadIdx.x == 0 && threadIdx.y == 0 )
        {
            for ( int i = 0; i < 32; ++i )
            {
                index[i+1] += index[i];
            }
        }

        __syncthreads();

        if ( count != UINT_MAX )
        {
            map.insert( n, count + index[threadIdx.y] );
        }

        __syncthreads();

        if ( threadIdx.x == 0 && threadIdx.y == 0 )
        {
            index[0] = index[32];
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief "Uses" functions to make the compiler include them into the library.
///
/// Dummy functions to force the compiler to instantiate templated functions.
/// Templated functions only really work if the compiler has access to all of 
/// the code when compiling a project that uses the templated functions. This 
/// is not the case in a library, unless all the code is placed in header 
/// files. To circumvent this, the compiler can be forced to instantiate 
/// certain specific specializations of the templated functions. This allows 
/// other projects to use those instantiations, but nothing else. But since 
/// Nodes only come in a limited number of varieties and are strictly specified 
/// by the library, this doesn't matter much.
///
/// When implementing a templated function that is not part of the Voxelizer 
/// class, the functions should be added here. Then, this function will be 
/// called with each Node type from the masterDummyFunction.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> void dummyFunction()
{
    CommonDevData dd;
    CommonHostData hd;
    Bounds<uint2> ui2b;
    Bounds<uint3> ui3b;
    Node * n = NULL;
    SNode * sn = NULL;
    VoxInt * v = NULL;
    clock_t t = 0;
    bool * b = NULL;
    float * f = NULL;
    uint * u = NULL;
    uchar * c = NULL;
    HashMap h;

    calcNodeList<Node>( dd, v, n, ui2b, t, true );
    launchConvertToFCCGrid<Node>( dd, v, n, ui2b, int(0), t, true );
    procNodeList<Node,SNode>( dd, n, n, b, ui2b, true, sn, t, true );
    launchCalculateFCCBoundaries<Node>( dd, n, n, ui2b, true, t, true );
    calcSurfaceVoxelization<Node>( dd, hd, f, u, n, c, t, true );
    calcOptSurfaceVoxelization<Node,SNode>( dd, hd, f, u, u, u, c, n, ui3b, int(0)
                                    , false, sn, t, true );
    makePaddingZero<Node>( dd, n, n, true, t, true );
    restoreRotatedNodes<Node>( dd, n, n, ui2b, t, true );
    calcSurfNodeCount<Node>( dd, n, t, true );
    populateHashMap<Node>( dd, n, t, true );
}
///////////////////////////////////////////////////////////////////////////////
/// \brief "Uses" functions with every \p Node type to force the compiler to 
///        compile them.
/// 
/// Calls the dummyFunction with all Node types in order to instantiate their 
/// template parameters. If a new node type is added, a new dummyFunction 
/// instantiation needs to be added as well.
///////////////////////////////////////////////////////////////////////////////
void masterDummyFunction()
{
    CommonDevData dd;
    CommonHostData hd;
    Bounds<uint2> ui2b;
    Bounds<uint3> ui3b;
    VoxInt * v = NULL;
    clock_t t = 0;
    //bool * b = NULL;
    float * f = NULL;
    uint * u = NULL;
    //uchar * c = NULL;

    sortWorkQueue( dd, u, u, t, true );
    compactWorkQueue( dd, u, u, u, t, true );
    calcTileOverlap( dd, hd, f, u, u, ui2b, t, true );
    calcWorkQueue( dd, hd, f, u, u, u, u, ui2b, t, true );
    calcVoxelization( dd, hd, f, u, u, u, u, u, v, ui3b, t, true );
    calcTriangleClassification( dd, hd, f, u, u, u, t, true );
}

template void dummyFunction<ShortNode, SurfaceNode>();
template void dummyFunction<LongNode, SurfaceNode>();
template void dummyFunction<PartialNode, SurfaceNode>();
template void dummyFunction<ShortFCCNode, SurfaceNode>();
template void dummyFunction<LongFCCNode, SurfaceNode>();
template void dummyFunction<VolumeNode, SurfaceNode>();
template void dummyFunction<VolumeMapNode, SurfaceNode>();

} // End namespace vox
