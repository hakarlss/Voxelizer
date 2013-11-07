#ifdef _WIN32
#pragma once
#endif

#ifndef GLOBAL_FUNCTIONS_H
#define GLOBAL_FUNCTIONS_H

#include "clean_defs.h"
#include <cuda.h>

namespace vox {

/// Sorts the work queue by tile id.
void sortWorkQueue(
    CommonDevData const & devData,
    uint                * workQueueTriangles_gpu,
    uint                * workQueueTiles_gpu,
    clock_t	              startTime, 
    bool		          verbose );
/// Compacts the work queue by removing duplicate tiles.
void compactWorkQueue(
    CommonDevData & devData,
    uint          * workQueueTiles_gpu,
    uint          * tileList_gpu,
    uint          * tileOffsets_gpu,
    clock_t		    startTime, 
    bool	        verbose );
/// Calculates the number of overlapping tiles for each triangle.
void calcTileOverlap(
    CommonDevData  const & devData,
    CommonHostData const & hostData,
    float          const * vertices_gpu,
    uint           const * indices_gpu,
    uint                 * tileOverlaps_gpu,
    Bounds<uint2>  const & yzSubSpace, 
    clock_t				   startTime, 
    bool				   verbose );
/// Generates the work queue.
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
    bool				   verbose );
/// Generates the grid of integers representing voxels.
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
    bool				   verbose );
/// Performs a simple translation from the integer based voxelization to nodes.
template <class Node> void calcNodeList(
    CommonDevData  const & devData,
    VoxInt         const * voxels_gpu,
    Node                 * nodes_gpu,
    Bounds<uint2>  const & yzSubSpace, 
    clock_t				   startTime, 
    bool				   verbose );
/// Translates an integer-based voxelization to a grid of <tt>FCC Nodes</tt>.
template <class Node> void launchConvertToFCCGrid(
    CommonDevData const & devData, 
    VoxInt        const * voxels_gpu,
    Node                * nodes_gpu,
    Bounds<uint2> const & yzSubSpace, 
    int                   gridType,
    clock_t				  startTime, 
    bool				  verbose );
/// Calculates the boundary ids of each node.
template <class Node> void procNodeList(
    CommonDevData const & devData, 
    Node                * nodes_gpu,
    Node                * nodesCopy_gpu,
    bool                * error_gpu,
    Bounds<uint2> const & yzSubSpace,
    bool                  xSlicing,
    clock_t				  startTime, 
    bool				  verbose );
/// Calculates the boundary ids of each FCC node.
template <class Node> void launchCalculateFCCBoundaries(
    CommonDevData const & devData, 
    Node                * nodes_gpu,
    Node                * nodesCopy_gpu,
    Bounds<uint2> const & yzSubSpace,
    bool                  xSlicing,
    clock_t				  startTime, 
    bool				  verbose );
/// Uses the simple surface voxelizer to produce a surface voxelizaiton.
template <class Node> void calcSurfaceVoxelization(
    CommonDevData  const & devData, 
    CommonHostData const & hostData, 
    float          const * vertices_gpu, 
    uint           const * indices_gpu, 
    Node                 * nodes_gpu, 
    uchar          const * materials_gpu, 
    clock_t				   startTime, 
    bool				   verbose );
/// Classifies the triangles according to their bounding box.
void calcTriangleClassification(
    CommonDevData  const & devData, 
    CommonHostData       & hostData, 
    float          const * vertices_gpu, 
    uint           const * indices_gpu, 
    uint                 * triangleTypes_gpu,
    uint                 * sortedTriangles_gpu,
    clock_t		           startTime, 
    bool		           verbose );
/// Calculates a surface voxelization with the optimized surface voxelizer.
template <class Node> 
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
    clock_t                startTime, 
    bool                   verbose );
/// Makes the outermost nodes of the array zero.
template <class Node>
void makePaddingZero( CommonDevData const & devData,
                      Node                * nodes_gpu,
                      Node                * nodesCopy_gpu,
                      bool                  xSlicing,
                      clock_t               startTime,
                      bool                  verbose );
/// Undoes the rotation when slicing along the x-axis.
template <class Node>
void restoreRotatedNodes( CommonDevData const & devData,
                          Node                * nodes_gpu,
                          Node                * nodesCopy_gpu,
                          Bounds<uint2> const & yzSubSpace,
                          clock_t               startTime,
                          bool                  verbose );

// Solid voxelization kernels and device functions.

/// Kernel that calculates the number of tiles that overlap each triangle.
__global__ void calculateTileOverlap(float const * vertices, 
                                      uint const * indices, 
                                      uint		 * tileOverlaps, 
                                      uint		   nrOfTriangles, 
                                   double3		   minVertex, 
                                    double		   voxelLength,  
                             Bounds<uint3>		   resolution,
                             Bounds<uint2>		   subSpace );
/// Kernel that produces pairs of triangles and tiles that overlap each other.
__global__ void constructWorkQueue(float const * vertices, 
                                    uint const * indices, 
                                    uint	   * workQueueTriangles, 
                                    uint	   * workQueueTiles, 
                                    uint const * offsetBuffer, 
                                    uint		 nrOfTriangles, 
                                 double3		 minVertex, 
                                     int		 firstIndex, 
                                  double		 voxelDistance,
                           Bounds<uint3>		 totalResolution,
                           Bounds<uint2>		 subSpace );
/// Kernel that produces the voxelization of grid of integers.
__global__ void generateVoxelization(float const * vertices, 
                                      uint const * indices, 
                                      uint const * workQueueTriangles, 
                                      uint const * workQueueTiles, 
                                      uint const * tileList, 
                                      uint const * TileOffsets, 
                                    VoxInt		 * voxels, 
                                      uint		   numberOfPolygons, 
                                      uint		   workQueueSize, 
                                   double3		   minVertex, 
                                    double		   voxelDistance, 
                                      bool         left,
                                      bool         right,
                                      bool         up,
                                      bool         down,
                             Bounds<uint3>		   totalResolution,
                             Bounds<uint3>		   subSpace );
/// Kernel that converts voxels in integers to \p Nodes.
template <class Node>
__global__ void constructNodeList2( VoxInt const * voxels, 
                                      Node	     * nodes,
                             Bounds<uint3>		   resolution, 
                             Bounds<uint2>		   yzSubSpace );
/// Kernel that converts voxels in integers to <tt>FCC Nodes</tt>.
template <class Node>
__global__ void convertToFCCGrid ( VoxInt const * voxels
                                 , Node         * nodes
                                 , int            gridType
                                 , Bounds<uint3>  resolution
                                 , Bounds<uint2>  yzSubSpace );
/// Kernel that undoes the rotation when slicing along the x-axis.
template <class Node>
__global__ void unRotateNodes( Node *        inputNodes
                             , Node *        outputNodes
                             , Bounds<uint3> resolution
                             , Bounds<uint2> yzSubSpace );
/// Kernel that calculates the boundary ids of each node.
template <class Node>
__global__ void fillNodeList2( Node       * nodes,
                      Bounds<uint3>         resolution, 
                      Bounds<uint2>         yzSubSpace,
                               bool       * error );
/// Kernel that calculates the boundary ids of each FFC node.
template <class Node>
__global__ void calculateFCCBoundaries( Node * nodes
                                      , Bounds<uint3> resolution
                                      , Bounds<uint2> yzSubSpace );
/// Kernel that performs a simple surface voxelization.
template <class Node>
__global__ void SimpleSurfaceVoxelizer(float const * vertices, 
                                        uint const * indices, 
                                        Node	   * nodes,
                                       uchar const * materials,
                                        uint		 nrOfTriangles,
                                      float3		 modelBBMin,
                                       float		 voxelLength,
                                       uint3		 resolution);
/// Kernel that classifies each triangle according to their bounding box.
__global__ void classifyTriangles(float const * vertices, 
                                   uint const * indices,
                                   uint		  * triTypeBuffer,
                                   uint		    nrOfTriangles,
                                 float3		    modelBBMin,
                                  float		    voxelLength );
/// Kernel that surface voxelizes triangles with a 1D bounding box.
template <class Node>
__global__ void process1DTriangles(float const * vertices, 
                                    uint const * indices, 
                                    uint const * triTypes,
                                    uint const * triangles,
                                   uchar const * materials,
                                    Node	   * nodes,
                                    uint		 triStartIndex,
                                    uint		 triEndIndex,
                                    bool         left,
                                    bool         right,
                                    bool         up,
                                    bool         down,
                                  float3		 modelBBMin,
                                   float		 voxelLength,
                           Bounds<uint3>	     totalResolution,
                           Bounds<uint3>         subSpace,
                                     int         gridType  );
/// Kernel that surface voxelizes triangles with a 2D bounding box.
template <class Node>
__global__ void process2DTriangles(float const * vertices, 
                                    uint const * indices,
                                    uint const * triTypes,
                                    uint const * triangles,
                                   uchar const * materials,
                                    Node	   * nodes,
                                    uint		 triStartIndex,
                                    uint		 triEndIndex,
                                    bool         left,
                                    bool         right,
                                    bool         up,
                                    bool         down,
                                  float3		 modelBBMin,
                                   float		 voxelLength,
                           Bounds<uint3>	     totalResolution,
                           Bounds<uint3>         subSpace,
                                     int         gridType  );
/// Kernel that surface voxelizes triangles with a 3D bounding box.
template <class Node>
__global__ void process3DTriangles(float const * vertices, 
                                    uint const * indices, 
                                    uint const * triTypes,
                                    uint const * triangles,
                                   uchar const * materials,
                                    Node	   * nodes,
                                    uint		 triStartIndex,
                                    uint		 triEndIndex,
                                    bool         left,
                                    bool         right,
                                    bool         up,
                                    bool         down,
                                  float3		 modelBBMin,
                                   float		 voxelLength,
                           Bounds<uint3>	     totalResolution,
                           Bounds<uint3>         subSpace,
                                     int         gridType );
/// Kernel that makes the outermost Nodes zero.
template <class Node>
__global__ void zeroPadding( Node  * nodes,
                             const uint3   dimensions );

template <class Node>
void dummyFunction();

void masterDummyFunction();

} // End namespace vox

#endif
