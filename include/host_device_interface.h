#ifdef _WIN32
#pragma once
#endif

#ifndef VOX_HOST_DEVICE_INTERFACE_H
#define VOX_HOST_DEVICE_INTERFACE_H

#include "common.h"

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
/// 
template <class Node>
void calcSurfNodeCount
    ( CommonDevData & devData
    , Node * nodes
    , clock_t startTime
    , bool verbose
    );
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
template <class Node, class SNode> void procNodeList(
    CommonDevData const & devData, 
    Node                * nodes_gpu,
    Node                * nodesCopy_gpu,
    bool                * error_gpu,
    Bounds<uint2> const & yzSubSpace,
    bool                  xSlicing,
    SNode               * surfNodes,
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
///
template <class Node>
void populateHashMap
    ( CommonDevData       & devData
    , Node                * nodes_gpu
    , clock_t               startTime
    , bool                  verbose );

} // End namespace vox

#endif
