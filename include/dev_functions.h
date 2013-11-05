#pragma once

#include "clean_defs.h"
#include "node_types.h"
#include <cuda.h>

namespace vox {

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
template <class Node> __host__ __device__ 
    void processVoxel( Node * nodes
                     , uchar const * materials
                     , uint triangleIdx
                     , float3 * triangle
                     , float3 triNormal
                     , float3 modelBBMin
                     , float voxelLength
                     , int x
                     , int y
                     , int z
                     , int gridType
                     , uint3 resolution
                     );
/// Writes a PartialNode to memory.
template <> __host__ __device__ 
    void processVoxel<PartialNode>( PartialNode * nodes
                                  , uchar const * materials
                                  , uint triangleIdx
                                  , float3 * triangle
                                  , float3 triNormal
                                  , float3 modelBBMin
                                  , float voxelLength
                                  , int x
                                  , int y
                                  , int z
                                  , int gridType
                                  , uint3 resolution
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

} // End namespace vox

#ifndef _DOUBLE2_DOUBLE3_MATH_
///////////////////////////////////////////////////////////////////////////////
/// \brief Basic math operators for double2 and double3 structs.
///
/// If these functions have been defined elsewhere and they cause conflicts 
/// when compiling, simply undefine _DOUBLE2_DOUBLE3_MATH_.
///////////////////////////////////////////////////////////////////////////////
#define _DOUBLE2_DOUBLE3_MATH_
#endif

#ifdef _DOUBLE2_DOUBLE3_MATH_
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 negation.
/// 
/// \param[in] a \p double2.
/// \returns \f$ -\mathbf{\vec{a}} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double2 operator-( double2 &a ) { return make_double2( -a.x, -a.y ); }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 negation.
/// 
/// \param[in] a \p double3.
/// \returns \f$ -\mathbf{\vec{a}} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 operator-( double3 &a ) 
{ 
    return make_double3( -a.x, -a.y, -a.z ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 addition. 
/// 
/// \param[in] a \p double2.
/// \param[in] b \p double2.
/// \returns \f$ \mathbf{\vec{a}} + \mathbf{\vec{b}} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double2 operator+( double2 a, double2 b ) 
{ 
    return make_double2( a.x + b.x, a.y + b.y ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 addition.
/// 
/// \param[in] a \p double3.
/// \param[in] b \p double3.
/// \returns \f$ \mathbf{\vec{a}} + \mathbf{\vec{b}} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 operator+( double3 a, double3 b ) 
{ 
    return make_double3( a.x + b.x, a.y + b.y, a.z + b.z ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 addition with scalar \p double.
/// 
/// \param[in] a \p double2.
/// \param[in] b \p double.
/// \returns \f$ \mathbf{\vec{a}} + b \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double2 operator+( double2 a, double b ) 
{ 
    return make_double2( a.x + b, a.y + b ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 addition with scalar \p double.
/// 
/// \param[in] a \p double3.
/// \param[in] b \p double.
/// \returns \f$ \mathbf{\vec{a}} + b \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 operator+( double3 a, double b ) 
{ 
    return make_double3( a.x + b, a.y + b, a.z + b ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 addition with scalar \p double.
/// 
/// \param[in] b \p double.
/// \param[in] a \p double2.
/// \returns \f$ b + \mathbf{\vec{a}} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double2 operator+( double b, double2 a ) 
{ 
    return make_double2( a.x + b, a.y + b ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 addition with scalar \p double.
/// 
/// \param[in] b \p double.
/// \param[in] a \p double3.
/// \returns \f$ b + \mathbf{\vec{a}} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 operator+( double b, double3 a ) 
{ 
    return make_double3(a.x + b, a.y + b, a.z + b); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 addition assignment.
/// 
/// \f$ \mathbf{\vec{a}} = \mathbf{\vec{a}} + \mathbf{\vec{b}} \f$.
/// \param[in,out] a \p double2.
/// \param[in] b \p double2.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator+=( double2 &a, double2 b ) { a.x += b.x; a.y += b.y; }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 addition assignment.
/// 
/// \f$ \mathbf{\vec{a}} = \mathbf{\vec{a}} + \mathbf{\vec{b}} \f$.
/// \param[in,out] a \p double3.
/// \param[in] b \p double3.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator+=( double3 &a, double3 b ) 
{ 
    a.x += b.x; 
    a.y += b.y; 
    a.z += b.z; 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 addition assignment with scalar \p double.
/// 
/// \f$ \mathbf{\vec{a}} = \mathbf{\vec{a}} + b \f$.
/// \param[in,out] a \p double2.
/// \param[in] b \p double.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator+=( double2 &a, double b ) { a.x += b; a.y += b; }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 addition assignment with scalar \p double.
/// 
/// \f$ \mathbf{\vec{a}} = \mathbf{\vec{a}} + b \f$.
/// \param[in,out] a \p double3.
/// \param[in] b \p double.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator+=( double3 &a, double b ) { a.x += b; a.y += b; a.z += b; }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 subtraction.
/// 
/// \param[in] a \p double2.
/// \param[in] b \p double2.
/// \return \f$ \mathbf{\vec{a}} - \mathbf{\vec{b}} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double2 operator-( double2 a, double2 b ) 
{ 
    return make_double2( a.x - b.x, a.y - b.y ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 subtraction.
/// 
/// \param[in] a \p double3.
/// \param[in] b \p double3.
/// \return \f$ \mathbf{\vec{a}} - \mathbf{\vec{b}} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 operator-( double3 a, double3 b ) 
{ 
    return make_double3( a.x - b.x, a.y - b.y, a.z - b.z ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 subtraction assignment.
/// 
/// \f$ \mathbf{\vec{a}} = \mathbf{\vec{a}} - \mathbf{\vec{b}} \f$.
/// \param[in,out] a \p double2.
/// \param[in] b \p double2.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator-=( double2 &a, double2 b ) { a.x -= b.x; a.y -= b.y; }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 subtraction assignment.
/// 
/// \f$ \mathbf{\vec{a}} = \mathbf{\vec{a}} - \mathbf{\vec{b}} \f$.
/// \param[in,out] a \p double3.
/// \param[in] b \p double3.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator-=( double3 &a, double3 b ) 
{ 
    a.x -= b.x; 
    a.y -= b.y; 
    a.z -= b.z; 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 subtraction with scalar \p double.
/// 
/// \param[in] a \p double2.
/// \param[in] b \p double.
/// \return \f$ \mathbf{\vec{a}} - b \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double2 operator-( double2 a, double b ) 
{
    return make_double2( a.x - b, a.y - b ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 subtraction with scalar \p double.
/// 
/// \param[in] a \p double3.
/// \param[in] b \p double.
/// \return \f$ \mathbf{\vec{a}} - b \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 operator-( double3 a, double b ) 
{ 
    return make_double3( a.x - b, a.y - b, a.z - b ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 subtraction with scalar \p double.
/// 
/// \param[in] b \p double.
/// \param[in] a \p double2.
/// \return \f$ b - \mathbf{\vec{a}} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double2 operator-( double b, double2 a ) 
{ 
    return make_double2( b - a.x, b - a.y ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 subtraction with scalar \p double.
/// 
/// \param[in] b \p double.
/// \param[in] a \p double3.
/// \return \f$ b - \mathbf{\vec{a}} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 operator-( double b, double3 a ) 
{ 
    return make_double3( b - a.x, b - a.y, b - a.z ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 subtraction assignment with scalar \p double.
/// 
/// \f$ \mathbf{\vec{a}} = \mathbf{\vec{a}} - b \f$.
/// \param[in,out] a \p double2.
/// \param[in] b \p double.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator-=( double2 &a, double b ) { a.x -= b;  a.y -= b; }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 subtraction assignment with scalar \p double.
/// 
/// \f$ \mathbf{\vec{a}} = \mathbf{\vec{a}} - b \f$.
/// \param[in,out] a \p double3.
/// \param[in] b \p double.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator-=( double3 &a, double b ) { a.x -= b;  a.y -= b; a.z -= b; }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 element-wise multiplication.
///
/// \param[in] a \p double2.
/// \param[in] b \p double2.
/// \returns \f$ \begin{bmatrix} a_{x} \cdot b_{x} \\ a_{y} \cdot b_{y} 
///          \end{bmatrix} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double2 operator*( double2 a, double2 b ) 
{ 
    return make_double2(a.x * b.x, a.y * b.y); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 element-wise multiplication assignment.
/// 
/// \f$ \mathbf{\vec{a}} = \begin{bmatrix} a_{x} \cdot b_{x} \\ a_{y} 
/// \cdot b_{y} \end{bmatrix} \f$.
/// \param[in,out] a \p double2.
/// \param[in] b \p double2.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator*=( double2 &a, double2 b ) { a.x *= b.x; a.y *= b.y; }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 multiplication with scalar \p double.
/// 
/// \param[in] a \p double2.
/// \param[in] b \p double.
/// \returns \f$ \mathbf{\vec{a}} \cdot b \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double2 operator*( double2 a, double b ) 
{ 
    return make_double2( a.x * b, a.y * b ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 multiplication with scalar \p double.
/// 
/// \param[in] b \p double.
/// \param[in] a \p double2.
/// \returns \f$ b \cdot \mathbf{\vec{a}} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double2 operator*( double b, double2 a ) 
{ 
    return make_double2(b * a.x, b * a.y); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 multiplication assignment with scalar \p double.
/// 
/// \f$ \mathbf{\vec{a}} = \mathbf{\vec{a}} \cdot b \f$.
/// \param[in,out] a \p double2.
/// \param[in] b \p double.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator*=( double2 &a, double b ) { a.x *= b; a.y *= b; }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 element-wise division.
/// 
/// \param[in] a \p double2.
/// \param[in] b \p double2.
/// \returns \f$ \begin{bmatrix} a_{x} / b_{x} \\ a_{y} / b_{y} 
///          \end{bmatrix} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double2 operator/( double2 a, double2 b ) 
{ 
    return make_double2( a.x / b.x, a.y / b.y ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 element-wise division assignment.
///  
/// \f$ \mathbf{\vec{a}} = \begin{bmatrix} a_{x} / b_{x} \\ 
/// a_{y} / b_{y} \end{bmatrix} \f$.
/// \param[in,out] a \p double2.
/// \param[in] b \p double2.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator/=( double2 &a, double2 b ) { a.x /= b.x; a.y /= b.y; }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 division with scalar \p double.
/// 
/// \param[in] a \p double2.
/// \param[in] b \p double.
/// \returns \f$ \mathbf{\vec{a}} \cdot \dfrac{1}{b} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double2 operator/( double2 a, double b ) 
{ 
    return make_double2( a.x / b, a.y / b ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 division assignment with scalar \p double.
/// 
/// \f$ \mathbf{\vec{a}} = \mathbf{\vec{a}} \cdot \dfrac{1}{b} \f$.
/// \param[in,out] a \p double2.
/// \param[in] b \p double.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator/=( double2 &a, double b ) { a.x /= b; a.y /= b; }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 division with scalar \p double.
/// 
/// \param[in] b \p double.
/// \param[in] a \p double2.
/// \returns \f$ b \cdot \begin{bmatrix} 1 / a_{x} \\ 1 / a_{y}
///          \end{bmatrix} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double2 operator/( double b, double2 a ) 
{ 
    return make_double2( b / a.x, b / a.y ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 element-wise multiplication.
/// 
/// \param[in] a \p double3.
/// \param[in] b \p double3.
/// \returns \f$ \begin{bmatrix} a_{x} \cdot b_{x} \\ a_{y} \cdot b_{y} 
///          \\ a_{z} \cdot b_{z} \end{bmatrix} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 operator*( double3 a, double3 b ) 
{ 
    return make_double3( a.x * b.x, a.y * b.y, a.z * b.z ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 element-wise multiplication assignment.
/// 
/// \f$ \mathbf{\vec{a}} = \begin{bmatrix} a_{x} \cdot b_{x} \\ a_{y} 
/// \cdot b_{y} \\ a_{z} \cdot b_{z} \end{bmatrix} \f$.
/// \param[in,out] a \p double3.
/// \param[in] b \p double3.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator*=( double3 &a, double3 b ) 
{ 
    a.x *= b.x; a.y *= b.y; a.z *= b.z; 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 multiplication with scalar \p double.
/// 
/// \param[in] a \p double3.
/// \param[in] b \p double.
/// \returns \f$ \mathbf{\vec{a}} \cdot b \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 operator*( double3 a, double b ) 
{ 
    return make_double3( a.x * b, a.y * b, a.z * b ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 multiplication with scalar \p double.
/// 
/// \param[in] b \p double.
/// \param[in] a \p double3.
/// \returns \f$ b \cdot \mathbf{\vec{a}} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 operator*( double b, double3 a ) 
{ 
    return make_double3( b * a.x, b * a.y, b * a.z ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 multiplication assignment with scalar \p double.
/// 
/// \f$ \mathbf{\vec{a}} = \mathbf{\vec{a}} \cdot b \f$.
/// \param[in,out] a \p double3.
/// \param[in] b \p double.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator*=( double3 &a, double b ) { a.x *= b; a.y *= b; a.z *= b; }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 element-wise division.
/// 
/// \param[in] a \p double3.
/// \param[in] b \p double3.
/// \returns \f$ \begin{bmatrix} a_{x} / b_{x} \\ a_{y} / b_{y} \\ 
///          a_{z} / b_{z} \end{bmatrix} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 operator/( double3 a, double3 b ) 
{ 
    return make_double3( a.x / b.x, a.y / b.y, a.z / b.z ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 element-wise division assignment.
/// 
/// \f$ \mathbf{\vec{a}} = \begin{bmatrix} a_{x} / b_{x} \\ 
/// a_{y} / b_{y} \\ a_{z} / b_{z} \end{bmatrix} \f$.
/// \param[in,out] a \p double3.
/// \param[in] b \p double3.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator/=( double3 &a, double3 b ) 
{ 
    a.x /= b.x; 
    a.y /= b.y; 
    a.z /= b.z; 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 division with scalar \p double.
/// 
/// \param[in] a \p double3.
/// \param[in] b \p double.
/// \returns \f$ \mathbf{\vec{a}} \cdot \dfrac{1}{b} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 operator/( double3 a, double b ) 
{ 
    return make_double3( a.x / b, a.y / b, a.z / b ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 division assignment with scalar \p double.
/// 
/// \f$ \mathbf{\vec{a}} = \mathbf{\vec{a}} \cdot \dfrac{1}{b} \f$.
/// \param[in,out] a \p double3.
/// \param[in] b \p double.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    void operator/=( double3 &a, double b ) { a.x /= b; a.y /= b; a.z /= b; }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 division with scalar \p double.
/// 
/// \param[in] b \p double.
/// \param[in] a \p double3.
/// \returns \f$ b \cdot \begin{bmatrix} 1 / a_{x} \\ 1 / a_{y} \\ 
///          1 / a_{z} \end{bmatrix} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 operator/( double b, double3 a ) 
{ 
    return make_double3( b / a.x, b / a.y, b / a.z ); 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 factory function.
/// 
/// \param[in] a \p float3 to be converted to \p double3.
/// \return \p a converted to \p double3.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 make_double3( float3 a ) { return make_double3( a.x, a.y, a.z ); }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double2 dot product.
/// 
/// \param[in] a \p double2.
/// \param[in] b \p double2.
/// \return \f$ \sum_{i=0}^{1} a_{i} \cdot b_{i} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double dot( double2 a, double2 b ) { return a.x * b.x + a.y * b.y; }
///////////////////////////////////////////////////////////////////////////////
/// \brief \p double3 dot product.
/// 
/// \param[in] a \p double3.
/// \param[in] b \p double3.
/// \return \f$ \sum_{i=0}^{2} a_{i} \cdot b_{i} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double dot( double3 a, double3 b ) 
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z; 
}
///////////////////////////////////////////////////////////////////////////////
/// \brief Normalizes a \p double2 vector.
/// 
/// \param[in] v \p double2.
/// \return \f$ \mathbf{\vec{v}} \cdot \left( \sqrt{ \mathbf{\vec{v}} \cdot 
///         \mathbf{\vec{v}} } \right)^{-1} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double2 normalize( double2 v ) { return v / sqrt( dot( v, v ) ); }
///////////////////////////////////////////////////////////////////////////////
/// \brief Normalizes a \p double3 vector.
/// 
/// \param[in] v \p double3.
/// \return \f$ \mathbf{\vec{v}} \cdot \left( \sqrt{ \mathbf{\vec{v}} \cdot 
///         \mathbf{\vec{v}} } \right)^{-1} \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 normalize( double3 v ) { return v / sqrt( dot( v, v ) ); }
///////////////////////////////////////////////////////////////////////////////
/// \brief Cross product for \p double3.
/// 
/// \param[in] a \p double3.
/// \param[in] b \p double3.
/// \return \f$ a \times b \f$.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ 
    double3 cross( double3 a, double3 b ) 
{ 
    return make_double3( a.y*b.z - a.z*b.y
                       , a.z*b.x - a.x*b.z
                       , a.x*b.y - a.y*b.x ); 
}
#endif // _DOUBLE2_DOUBLE3_MATH_
