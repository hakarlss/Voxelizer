#ifdef _WIN32
#pragma once
#endif

#ifndef VOX_DEVICE_CODE_H
#define VOX_DEVICE_CODE_H

#include "node_types.h"
#include "common.h"

#include <cuda.h>

namespace vox {

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
template <class Node, class SNode>
__global__ void fillNodeList2( Node       * nodes,
                      Bounds<uint3>         resolution, 
                      Bounds<uint2>         yzSubSpace,
                               bool       * error,
                            HashMap         hashMap,
                              SNode       * surfNodes );
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
template <class Node, class SNode>
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
                                     int         gridType,
                                    bool         countVoxels,
                                 HashMap         hashMap,
                                   SNode       * surfNodes );
/// Kernel that surface voxelizes triangles with a 2D bounding box.
template <class Node, class SNode>
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
                                     int         gridType,
                                    bool         countVoxels,
                                 HashMap         hashMap,
                                   SNode       * surfNodes );
/// Kernel that surface voxelizes triangles with a 3D bounding box.
template <class Node, class SNode>
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
                                     int         gridType,
                                    bool         countVoxels,
                                 HashMap         hashMap,
                                   SNode       * surfNodes );
/// Kernel that makes the outermost Nodes zero.
template <class Node>
__global__ void zeroPadding( Node  * nodes,
                             const uint3   dimensions );

template <class Node>
__global__ void fillHashMap( Node * nodes
                           , HashMap map
                           , const uint3 dim );

/// Implements the tile loop for calculateTileOverlap().
inline __host__ __device__ 
    void traverseTilesForOverlaps( double2       const * edgeNormals
                                 , double        const * distances
                                 , uint	               & nrOfTilesOverlapped
                                 , Bounds<uint2> const & tiles
                                 , Bounds<uint2> const & voxels
                                 , double3       const & minVertex
                                 , double		         d
                                 );
/// Implements the tile loop for constructWorkQueue().
inline __host__ __device__ 
    void traverseTilesForWorkQueue( double2       const * edgeNormals
                                  , double        const * distances
                                  , uint	            * workQueueTriangles
                                  , uint		        * workQueueTiles
                                  , uint		          offset
                                  , uint		          triangle
                                  , Bounds<uint2> const & tiles
                                  , Bounds<uint2> const & voxels
                                  , uint3         const & totalResolution
                                  , double3       const & minVertex
                                  , double		          d
                                  );
/// Loads the vertices of a triangle into memory.
inline __host__ __device__ 
    void fetchTriangleVertices( float   const * vertices
                              , uint    const * indices
                              , double3		  * triangle
                              , uint		    triIdx
                              );
/// Translates a permutation to a boundary id.
inline __host__ __device__ 
    uchar getOrientation( uchar permutation );
/// Calculates the bounding box of a triangle.
inline __host__ __device__ 
    void getTriangleBounds( float3 const * vertices, Bounds<float3> & bounds );
/// Calculates the bounding box of a triangle.
inline __host__ __device__ 
    void getTriangleBounds( double3 const * vertices
                          , Bounds<double2> & bounds
                          );
/// Takes a triangle's bounding box and converts it to voxel coordinates.
inline __host__ __device__ 
    void getVoxelBounds( Bounds<float3> const & triBB   
                       , float3		    const & modelBBMin
                       , Bounds<uint3>        & voxBB
                       , float				    d
                       );

/// Takes a triangle's bounding box and converts it to voxel coordinates.
inline __host__ __device__
    void getVoxelBoundsDouble( Bounds<double2> const & triBB
                             , double3         const & modelBBMin
                             , Bounds<double2>       & voxBB
                             , double                  d 
                             );
/// Takes a triangle's bounding box and converts it to voxel coordinates.
inline __host__ __device__ 
    Bounds<int3> getVoxelBoundsHalf( Bounds<float3> const & triBB
                                   , float3         const & modelBBMin
                                   , float		            d
                                   );
/// Calculates part of the bounding box of a triangle in voxel coodinates.
inline __host__ __device__ 
    int getVoxCBoundMin( float v, float m, float d );
/// Calculates part of the bounding box of a triangle in voxel coodinates.
inline __host__ __device__ 
    int getVoxCBoundMax( float v, float m, float d );
/// Determines the <em>dominant axis</em> of a triangle's normal.
inline __host__ __device__ 
    MainAxis determineDominantAxis( float3 triNormal );
/// Calculates the minimum corner of a single voxel.
inline __host__ __device__ 
    const float3 getSingleVoxelBounds( int    x
                                     , int    y
                                     , int    z
                                     , float3 modelBBMin
                                     , float  d
                                     );
/// Calculates part of the minimum corner of a single voxel.
inline __host__ __device__ 
    float getSingleVoxelBoundsComponent( int   coord
                                       , float modelBBMinComp
                                       , float d
                                       );
/// Tests if a voxel overlaps a triangle.
inline __host__ __device__ 
    bool voxelOverlapsTriangle( float2 ds
                              , OverlapData const * data
                              , float3 triNormal
                              , float3 p
                              );
/// Sets up one of the overlap tests.
inline __host__ __device__ 
    OverlapData setupOverlapTest( float3 const * tri
                                , float3 n
                                , float d
                                , MainAxis axis
                                );
/// Sets up the plane / voxel overlap test.
inline __host__ __device__ 
    float2 setupPlaneOverlapTest( float3 const * tri, float3 n, float d );
/// Sets up the full-voxel overlap test in the YZ-plane.
inline __host__ __device__ 
    OverlapData setupYZOverlapTest( float3 const * tri, float3 n, float d );
/// Sets up the full-voxel overlap test in the ZX-plane.
inline __host__ __device__ 
    OverlapData setupZXOverlapTest( float3 const * tri, float3 n, float d );
/// Sets up the full-voxel overlap test in the XY-plane.
inline __host__ __device__ 
    OverlapData setupXYOverlapTest( float3 const * tri, float3 n, float d );
/// Sets up the voxel center overlap test in the YZ-plane.
inline __host__ __device__ 
    OverlapData setupSimpleYZOverlapTest( float3 const * tri
                                        , float3 n
                                        , float d
                                        );
/// Sets up the voxel center overlap test in the ZX-plane.
inline __host__ __device__ 
    OverlapData setupSimpleZXOverlapTest( float3 const * tri
                                        , float3 n
                                        , float d
                                        );
/// Sets up the voxel center overlap test in the XY-plane.
inline __host__ __device__ 
    OverlapData setupSimpleXYOverlapTest( float3 const * tri
                                        , float3 n
                                        , float d
                                        );
/// Performs the plane / voxel overlap test.
inline __host__ __device__ 
    bool planeOverlapTest( float2 ds, float3 triNormal, float3 p );
/// Perform an overlap test along the desired axis.
inline __host__ __device__ 
    bool overlapTest( OverlapData data, float3 p, MainAxis axis );
/// Performs the overlap test in the YZ-plane.
inline __host__ __device__ 
    bool overlapTestYZ( OverlapData data, float3 p );
/// Performs the overlap test in the ZX-plane.
inline __host__ __device__ 
    bool overlapTestZX( OverlapData data, float3 p );
/// Performs the overlap test in the XY-plane.
inline __host__ __device__ 
    bool overlapTestXY( OverlapData data, float3 p );
/// Writes the voxel to memory once an overlap has been confirmed.
template <class Node, class SNode> __device__ 
    void processVoxel( Node * nodes
                     , uchar const * materials
                     , uint triangleIdx
                     , float3 * triangle
                     , float3 triNormal
                     , float3 modelBBMin
                     , float voxelLength
                     , int3 coords
                     , int3 adjustments
                     , int gridType
                     , uint3 resolution
                     , bool countVoxels
                     , HashMap & hashMap
                     , SNode * surfNodes
                     );
/// Determines the type of a triangle's bounding box.
inline __host__ __device__ 
    const TriData analyzeBoundingBox( Bounds<float3> triBB
                                    , float3 triNormal
                                    , float3 modelBBMin
                                    , float d 
                                    );
/// Encodes a TriData to an unsigned int.
inline __host__ __device__ 
    uint encodeTriangleType( TriData data );
/// Decodes an unsigned int to a TriData.
inline __host__ __device__ 
    const TriData decodeTriangleType( uint type );
/// Extracts the bounding box type from an encoded TriData.
inline __host__ __device__ 
    BBType readBBType( uint type );
/// Extracts the dominant axis from an encoded TriData.
inline __host__ __device__ 
    MainAxis readDomAxis( uint type );
/// Extracts the number of voxel columns from an encoded TriData.
inline __host__ __device__ 
    uint readNrOfVoxCols( uint type );
/// Calculates the area of the largest side of a bounding box.
inline __host__ __device__ 
    uint calculateLargestSideOfVoxels( uint3 res );
/// Filters out any voxels not within the subspace.
inline __host__ __device__ 
    void adjustVoxelRange( Bounds<int3> & voxBB
                         , Bounds<uint3> const & subSpace 
                         );
/// Determines the depth of the triangle in voxels along the x-axis.
inline __host__ __device__ 
    int2 determineDepthRangeX( float3 * triangle
                             , float3 p
                             , float3 triNormal
                             , float3 modelBBMin
                             , float d
                             );
/// Determines the depth of the triangle in voxels along the y-axis.
inline __host__ __device__ 
    int2 determineDepthRangeY( float3 * triangle
                             , float3 p
                             , float3 triNormal
                             , float3 modelBBMin
                             , float d
                             );
/// Determines the depth of the triangle in voxels along the z-axis.
inline __host__ __device__ 
    int2 determineDepthRangeZ( float3 * triangle
                             , float3 p
                             , float3 triNormal
                             , float3 modelBBMin
                             , float d
                             );
/// Calculates the intersection between ray and plane.
inline __host__ __device__ 
    float calculateIntersectionWithPlane( float3 p
                                        , float3 v
                                        , float3 n
                                        , float3 d
                                        );
/// Calculates the intersection between a ray along the x-axis and a plane.
inline __host__ __device__ 
    float intersectWithPlaneX( float3 p, float3 v, float3 n );
/// Calculates the intersection between a ray along the y-axis and a plane.
inline __host__ __device__ 
    float intersectWithPlaneY( float3 p, float3 v, float3 n );
/// Calculates the intersection between a ray along the z-axis and a plane.
inline __host__ __device__ 
    float intersectWithPlaneZ( float3 p, float3 v, float3 n );
/// Validates a bounding box.
template <class T> inline __host__ __device__ 
    void boundsCheck( T &min, T &max );
/// Calculates the part of the volume that is solid in a voxel.
inline __host__ __device__ 
    float calculateVoxelPlaneIntersectionVolume( float3 * triangle
                                               , int3 voxel
                                               , float3 triNormal
                                               , float3 modelBBMin
                                               , float d
                                               );
/// Calculates the volume of a voxel in simple cases.
inline __host__ __device__ 
    float boxParallelCutVolume( float voxCenterComponent
                              , float triVertexComponent
                              , float d
                              );
/// Rotates a point around the x-axis.
inline __host__ __device__ 
    float3 rotX( float3 v, float sin_t, float cos_t );
/// Rotates a point around the y-axis.
inline __host__ __device__ 
    float3 rotY( float3 v, float sin_t, float cos_t );
/// Calculates the volume of a pyramid with a polygonal base.
inline __host__ __device__ 
    float volumeOfPyramid( float3 * base, int nrOfVerts, float3 height );
/// Produces the relevant faces of the polyhedron cut from a voxel.
inline __host__ __device__ 
    void constructPolyhedronFaces( float3 * iPts
                                 , float3 * face1
                                 , float3 * face2
                                 , float3 * face3
                                 , int &nrOfIPts
                                 , int &nrF1
                                 , int &nrF2
                                 , int &nrF3
                                 , float3 * voxVerts
                                 , float3 * triangle
                                 , float3 triNormal
                                 );
/// Calculates the volume of the polyhedron cut from a voxel.
inline __host__ __device__ 
    float volumeOfPolyhedron( float3 * voxVerts
                            , float3 * iPts
                            , float3 * face1
                            , float3 * face2
                            , float3 * face3
                            , int nrOfIPts
                            , int nrF1
                            , int nrF2
                            , int nrF3
                            , float3 triNormal
                            );

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
    );

__device__ void constructAllPolyhedronFaces
    ( float3 * vertices
    , char      * indices
    , char      & nrOfIPts  
    , char      & nrF1    
    , char      & nrF2   
    , char      & nrF3   
    , char      & nrF4    
    , char      & nrF5    
    , char      & nrF6     
    , float3 * triangle  
    , float3   triNormal 
    );

__device__ float polyhedronVolume
    ( float3 * vertices
    , char   * indices
    , char     nrOfIPts  
    , char     nrF1  
    , char     nrF2  
    , char     nrF3   
    , float3   triNormal
    , float3 & dx
    , float3 & dy
    , float3 & dz
    , float  & ipArea
    , float  & f1Area
    , float  & f2Area
    , float  & f3Area
    );

__device__ float pyramidVolume
    ( float3 * base 
    , int nrOfVerts    
    , float3 height
    , float & baseArea
    );

__device__ float polygonArea
    ( float3 * base
    , int nrOfVerts
    );

template <class Node>
void dummyFunction();

void masterDummyFunction();

} // End namespace vox

#endif
