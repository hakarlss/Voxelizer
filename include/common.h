#ifdef _WIN32
#pragma once
#endif

#ifndef VOX_COMMON_H
#define VOX_COMMON_H

#include <stdint.h>

#include <limits>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sm_20_atomic_functions.h>

#include <exception>
#include <string>
#include <iostream>
#include <ctime>
#include <math.h>

#include <algorithm>

#define UINT32_MAX  ((uint32_t)-1)
#define UINT64_MAX  ((uint64_t)-1)

namespace vox
{

typedef unsigned int uint;   ///< \a uint is a 32-bit unsigned integer.
typedef uint16_t ushort; ///< \a ushort is a 16-bit unsigned integer.
typedef uint8_t uchar;   ///< \a uchar is an 8-bit unsigned integer.

#define VOX32 1 // Internal representation of the voxelization.

#if defined(VOX32)
typedef uint32_t VoxInt;
#define VOX_BPI 32
#define VOX_MAX UINT32_MAX
#define VOX_DIV 5
#elif defined(VOX64)    // Not supported yet.
typedef uint64_t VoxInt;
#define VOX_BPI 64
#define VOX_MAX UINT64_MAX
#define VOX_DIV 6
#else
#error "VOX32 or VOX64 needs to be defined!"
#endif

///////////////////////////////////////////////////////////////////////////////
/// \brief Describes the bounding box of a triangle. 
/// 
/// Used in the surface voxelizer to mark and sort triangles by their bounding 
/// box.
///////////////////////////////////////////////////////////////////////////////
enum BBType
{
    BBType_Degen = 0,   ///< Degenerate triangle.
    BBType_1D,          ///< One dimensional bounding box.
    BBType_2D,          ///< Two dimensional bounding box.
    BBType_3D           ///< Full, three dimensional bounding box.
};
///////////////////////////////////////////////////////////////////////////////
/// \brief Describes one of the three main axes. 
/// 
/// Mostly deprecated now.
///////////////////////////////////////////////////////////////////////////////
enum MainAxis
{
    xAxis = 0,  ///< X-axis.
    yAxis,      ///< Y-axis.
    zAxis       ///< Z-axis.
};
///////////////////////////////////////////////////////////////////////////////
/// \brief Compiles a few pieces of information about a triangle. 
/// 
/// Used in the surface voxelizer.
///////////////////////////////////////////////////////////////////////////////
struct TriData
{
    BBType bbType;      ///< Bounding box type of the triangle.
    MainAxis domAxis;   ///< \brief Dominant axis, i.e. which main axis of the 
                        ///<        normal is the largest.
                        ///<
    uint nrOfVoxCols;   ///< \brief How many voxel columns the triangle spans 
                        ///<        along its dominant axis.
                        ///<
};
///////////////////////////////////////////////////////////////////////////////
/// \brief Struct that collects the data needed to voxelize, including the 
///        plane overlap test.
///////////////////////////////////////////////////////////////////////////////
struct TestData
{
    float3 n;		///< Triangle normal.
    float2 d;		///< Plane overlap distances.
    float2 ne[9];	///< Edge normals.
    float de[9];	///< Edge distances.
    float3 p;		///< Voxel minimum corner.
};
///////////////////////////////////////////////////////////////////////////////
/// \brief Struct that collects the data needed to voxelize in the special case 
///        of a triangle with a 2D bounding box, but without the plane overlap 
///        test.
///////////////////////////////////////////////////////////////////////////////
struct TestData2D
{
    float3 n;		///< Triangle normal.
    float2 ne[3];	///< Edge normals.
    float de[3];	///< Edge distances.
    float3 p;		///< Voxel minimum corner.
};
///////////////////////////////////////////////////////////////////////////////
/// \brief Struct that collects the data needed to voxelize in the general 
///        case, but without the plane overlap test.
///////////////////////////////////////////////////////////////////////////////
struct TestData3D
{
    float3 n;		///< Triangle normal.
    float2 ne[9];	///< Edge normals.
    float de[9];	///< Edge distances.
    float3 p;		///< Voxel minimum corner.
};
///////////////////////////////////////////////////////////////////////////////
/// \brief Struct that only collects the edge normals and distances used in the 
///        overlap testing.
///////////////////////////////////////////////////////////////////////////////
struct OverlapData
{
    float2 ne[3];	///< Edge normals.
    float de[3];	///< Edge distances.
};
///////////////////////////////////////////////////////////////////////////////
/// \brief Struct to represent anything that can be described with a minimum 
///        and maximum value. 
/// 
/// Used extensively to represent bounding boxes in three, two 
/// and one dimensions.
///
/// \tparam T Type of the \p min and \p max members. 
///////////////////////////////////////////////////////////////////////////////
template <class T>
struct Bounds
{
    T min;  ///< Minimum value of the bound.
    T max;  ///< Maximum value of the bound.
};
///////////////////////////////////////////////////////////////////////////////
/// \brief Functor that returns \p true if the given \p TriData encoding 
/// represents the desired \p ::BBType. 
///
/// Used with the thrust counting alogorithm to calculate where 
/// the different \p ::BBType begin and end once they've been sorted.
///
/// \tparam T The wanted ::BBType.
/// \tparam B the particular amount of bits that need to be shifted to get at 
///           the \p ::BBType from the encoding. It is set to <tt>VOX_BPI - 
///           2</tt> in the code where it is used.
///////////////////////////////////////////////////////////////////////////////
template <BBType T, int B>
struct is_BB
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Tests if the encoded \p TriData equals the desired \p ::BBType.
    ///
    /// \param[in] x The \p TriData encoded into an \p uint.
    ///
    /// \return \p true if \p x decodes to \p T.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ bool operator()(const uint &x)
    {
        return (x >> B) == T;
    }
};
///////////////////////////////////////////////////////////////////////////////
/// \brief Functor that tests if a bid equals the given number.
///
/// \tparam T Type of node.
/// \tparam I Integer to test against.
///////////////////////////////////////////////////////////////////////////////
template <class T, int I>
struct NodeBidEquals
{
    /// Returns true if c.bid() equals I.
    __host__ __device__ bool operator()( const T & c ) { return c.bid() == I; }
};
///////////////////////////////////////////////////////////////////////////////
/// \brief Exception class used by the Voxelizer. 
///
/// Nothing fancy, its just using a \p string instead of a pointer to \p char.
///////////////////////////////////////////////////////////////////////////////
class Exception : public std::exception {
public:
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructor for strings.
    ///
    /// \param[in] message Error message.
    ///////////////////////////////////////////////////////////////////////////
    Exception(std::string message) throw(): msg(message) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructor for C-style strings.
    ///
    /// \param[in] message Error message.
    ///////////////////////////////////////////////////////////////////////////
    Exception(char const * message) throw(): msg(std::string(message)) {}
    /// Default destructor.
    ~Exception() throw() {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Gets the error message.
    ///
    /// \return Error message.
    ///////////////////////////////////////////////////////////////////////////
    const char * what() const throw() { return msg.c_str(); }
private:
    const std::string msg; ///< Error message.
};
///////////////////////////////////////////////////////////////////////////////
/// \brief Checks for cuda errors.
///
/// \throws Exception if CUDA reported an error.
///
/// \param[in] loc A string that describes the location where the function is 
///                used.
///////////////////////////////////////////////////////////////////////////////
inline void checkCudaErrors( std::string loc )
{
    cudaError_t e = cudaPeekAtLastError();

    if ( e != cudaSuccess )
    {
        std::string errorString = cudaGetErrorString( e );
        std::string msg = errorString + " @ " + loc;
        std::cout << msg;
        throw Exception( msg );
    }
}
///////////////////////////////////////////////////////////////////////////////
/// \brief A list of primes to be used with the hash map. 
///
/// Not really used anywhere at the moment, but it could be used to change 
/// the prime used in the hash function in case there are too many collisions 
/// with the current one.
///////////////////////////////////////////////////////////////////////////////
const uint primeList[204] = {
    970194959, 970194983, 970194989, 970195021, 970195073, 970195123, 
    970195133, 970195139, 970195147, 970195169, 970195183, 970195207, 
    970195243, 970195279, 970195297, 970195337, 970195357, 970195363, 
    970195379, 970195381, 970195393, 970195399, 970195427, 970195433, 
    970195439, 970195441, 970195451, 970195481, 970195483, 970195489, 
    970195507, 970195517, 970195529, 970195531, 970195543, 970195553, 
    970195621, 970195631, 970195673, 970195679, 970195693, 970195697, 
    970195741, 970195801, 970195813, 970195847, 970195859, 970195861, 
    970195873, 970195903, 970195909, 970195921, 970195927, 970195939, 
    970195999, 970196009, 970196011, 970196027, 970196039, 970196081, 
    970196083, 970196093, 970196099, 970196141, 970196147, 970196153, 
    970196173, 970196197, 970196233, 970196237, 970196251, 970196263, 
    970196267, 970196287, 970196303, 970196317, 970196321, 970196333, 
    970196347, 970196431, 970196453, 970196467, 970196471, 970196473, 
    970196497, 970196501, 970196503, 970196527, 970196567, 970196581, 
    970196587, 970196597, 970196627, 970196657, 970196693, 970196723, 
    970196737, 970196761, 970196771, 970196789, 970196797, 970196803, 
    970196819, 970196837, 970196881, 970196891, 970196923, 970196951, 
    970196977, 970196989, 970196999, 970197029, 970197031, 970197047, 
    970197061, 970197097, 970197113, 970197127, 970197161, 970197187, 
    970197251, 970197257, 970197271, 970197281, 970197287, 970197299, 
    970197329, 970197367, 970197377, 970197401, 970197409, 970197413, 
    970197427, 970197443, 970197457, 970197511, 970197533, 970197563, 
    970197601, 970197629, 970197643, 970197647, 970197653, 970197673, 
    970197677, 970197733, 970197737, 970197743, 970197751, 970197791, 
    970197821, 970197847, 970197887, 970197901, 970197911, 970197971, 
    970198003, 970198027, 970198051, 970198057, 970198063, 970198087, 
    970198093, 970198133, 970198147, 970198169, 970198199, 970198217, 
    970198249, 970198291, 970198297, 970198343, 970198373, 970198433, 
    970198511, 970198513, 970198529, 970198561, 970198573, 970198591, 
    970198601, 970198609, 970198651, 970198667, 970198679, 970198709, 
    970200463, 970200479, 970200487, 970200493, 970200593, 970200599, 
    970200619, 970200641, 970200683, 970200703, 970200739, 970200757, 
    970200767, 970200773, 970200797, 970200799, 974786081, 974786101
};

///////////////////////////////////////////////////////////////////////////////
/// \brief CUDA hash map implementation for (uint,uint)-mappings.
///
/// This is a hash map with a simple hash function that can map between 32-bit
/// unsigned integers. It stores (key,value)-pairs as 64-bit unsigned integers 
/// in a table. It needs to be manually allocated and deallocated from the 
/// host side prior to use, but after that it can be inserted into and applied 
/// on the device.
///////////////////////////////////////////////////////////////////////////////
class HashMap
{
public:
    /// Default constructor.
    __host__ __device__
    HashMap() {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Proper constructor for initializing the hash map.
    ///
    /// The hash map only needs three arguments to initialize: The size of the 
    /// hash map (which is recommended to be set to twice the intended amount 
    /// of elements), the maximum number of reshash attempts on collisions 
    /// before giving up, and a large prime number for the hash function.
    ///
    /// \param[in] size Size of the hash maps array.
    /// \param[in] maxProbes Max number of collisions allowed.
    /// \param[in] p A large prime number for the hash function.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__
    HashMap( uint size
           , int maxProbes = 100
           , uint p = 982451653
           ) : table(NULL)
             , size(size)
             , p(p)
             , maxProbes(maxProbes) {}
    /// Copy constructor.
    __host__ __device__
    HashMap( const HashMap & other ): size(other.size)
                                    , maxProbes(other.maxProbes)
                                    , p(other.p)
                                    , table(other.table) {}
    /// Default destructor.
    __host__ __device__
    ~HashMap() {}

#ifdef __CUDACC__
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Insert a pair into the hash map.
    ///
    /// Attempts to insert a pair into the hash map \p maxProbes times, using 
    /// a different hash each time a collision occurs. If none of the attempts 
    /// succeed, returns false. Returns true on a successful insertion.
    ///
    /// \param[in] id Key to insert with.
    /// \param[in] value Value to insert to the given key.
    ///
    /// \return \p true if insertion succeeded, \p false otherwise.
    ///////////////////////////////////////////////////////////////////////////
    __device__
    bool insert( uint32_t id, uint32_t value )
    {
        for ( int attempt = 1; attempt < maxProbes; ++attempt )
        {
            uint32_t hashCode = hash( id, attempt );
            
            // Insert with an atomic compare and swap to make it parallel-
            // friendly.
            uint64_t old = 
                atomicCAS((unsigned long long int*) &table[hashCode]
                         , (unsigned long long int)HashMap::EMPTY
                         , (unsigned long long int)((uint64_t(id) << 32) | uint64_t(value)) 
                         );

            if ( old == HashMap::EMPTY )
            {
                return true;
            }
        }
        return false;
    }
#endif
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Host version of the insertion function.
    ///
    /// Needs to have the hash table allocated on host memory for it to work.
    /// Returns the number of collisions before success (or at failure) for 
    /// debugging purposes.
    ///
    /// \param[in] id Key to insert with.
    /// \param[in] value Value to insert to the given key.
    ///
    /// \return number of collisions before returning.
    ///////////////////////////////////////////////////////////////////////////
    __host__
    uint insertHost( uint32_t id, uint32_t value )
    {
        uint collisions = 0;
        for ( int attempt = 1; attempt < maxProbes; ++attempt )
        {
            uint32_t hashCode = hash( id, attempt );
            
            uint64_t old = table[hashCode];

            if ( old == HashMap::EMPTY )
            {
                table[hashCode] = (uint64_t(id) << 32) | uint64_t(value);
                return collisions;
            }
            collisions++;
        }
        return collisions;
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Retrieves the value associated with the given key.
    ///
    /// \param[in] id The key to look with.
    ///
    /// \return The value that was associated with the given key, or 
    ///         \p UINT32_MAX if the key could not be found.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__
    uint32_t get( uint32_t id )
    {
        for ( int attempt = 1; attempt < maxProbes; ++attempt )
        {
            uint32_t hashCode = hash( id, attempt );

            uint64_t item = table[hashCode];
            if ( item == HashMap::EMPTY )
                return UINT32_MAX;
            else if ( uint32_t(item >> 32) == id )
                return uint32_t(item & (UINT64_MAX >> 32));
        }

        return UINT32_MAX;
    }
    /// Renews the prime number used in the hash function.
    __host__
    void renewHashPrime()
    {
        p = primeList[(p*p) % 204];
    }
    /// Allocates memory for the hash table on the device.
    __host__
    void allocate()
    {
        cudaMalloc( &table, sizeof(uint64_t)*size );
        cudaMemset( table, UINT32_MAX, sizeof(uint64_t)*size );
    }
    /// Allocates memory for the hash table on host memory.
    __host__
    void allocateHost()
    {
        table = new uint64_t[size];
        for ( uint i = 0; i < size; ++i )
            table[i] = HashMap::EMPTY;
    }
    /// Frees the allocated memory on the device.
    __host__
    void deallocate()
    {
        cudaFree( table );
    }
    /// Frees the allocated memory on host memory.
    __host__
    void deallocateHost()
    {
        delete[] table;
    }
    /// Clears the hash map, but only for host memory.
    __host__
    void clearHost()
    {
        for ( uint i = 0; i < size; ++i )
        {
            table[i] = HashMap::EMPTY;
        }
    }
    /// Converts a hash map on device memory to host memory.
    __host__
    void convertToHostMemory()
    {
        uint64_t * devPtr = table;
        table = new uint64_t[size];
        cudaMemcpy( table, devPtr, size * sizeof(uint64_t), cudaMemcpyDeviceToHost );
        cudaFree( devPtr );
    }

    static const uint64_t EMPTY = UINT64_MAX;   ///< Empty table entry.

private:
    uint64_t * table;   ///< Table of (key,value)-pairs.
    uint32_t size;      ///< Size of the table.
    uint32_t p;         ///< Current prime number.
    int maxProbes;      ///< Maximum tries before giving up.

    ///////////////////////////////////////////////////////////////////////////
    /// \brief The hash function
    ///
    /// Creates a hash by multiplying the key with a large prime and adding 
    /// the square of the current attempt to the result, and then taking the 
    /// modulo of the size of the table.
    ///
    /// \param[in] id Key.
    /// \param[in] attempt Current attempt.
    ///
    /// \return A hash code.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__
    uint32_t hash( uint32_t id, uint32_t attempt )
    {
        return (id*p + attempt*attempt) % size;
    }
};
///////////////////////////////////////////////////////////////////////////////
/// \brief A container for variables that need to have their own versions for 
///        each device.
///////////////////////////////////////////////////////////////////////////////
struct CommonDevData
{
    int dev;						///< Device number.
    cudaDeviceProp devProps;		///< Device properties.
    size_t freeMem;					///< Free memory.
    size_t maxMem;					///< Maximum memory.
    uint blocks;				    ///< Number of blocks.
    uint threads;					///< Number of threads per block.

    ///< \brief Minimum corner of the model's bounding box. 
    ///<
    ///< Matches exactly the voxelization space, meaning that there is no 
    ///< padding or anything extra that won't make it to the final 
    ///< voxelization.
    ///<
    double3		  minVertex;
    ///< \brief Extended minimum corner of the model's bounding box. 
    ///<
    ///< If there is a neighboring subspace to the left or below, then this 
    ///< differs from minVertex. The extension also happens during slicing.
    ///<
    double3       extMinVertex;
    uint		  workQueueSize;            ///< Size of the work queue.
    uint		  maxWorkQueueSize;         ///< \brief How much is allocated 
                                            ///<        for the work queue.
                                            ///<
    /// Which triangle is the first to have non-zero overlapping tiles.
    int			  firstTriangleWithTiles;
    /// Essentially the size of the compacted tile list.
    uint		  nrValidElements; 
    /// Dimensions of the model's bounding box in voxels.
    Bounds<uint3> resolution;
    /// Dimensions of the extended bounding box in voxels.
    Bounds<uint3> extResolution;
    /// Dimensions of the array that stores the voxels. Includes padding.
    Bounds<uint3> allocResolution;
    /// Where a subspace is relative to the whole space.
    uint3         location;
    
    bool          left;     ///< If there is a subspace to the left.
    bool          right;    ///< If there is a subspace to the right.
    bool          up;       ///< If there is a subspace above.
    bool          down;     ///< If there is a subspace below.

    uint nrOfNodes;
    uint nrOfSurfaceNodes;

    HashMap hashMap;
};
///////////////////////////////////////////////////////////////////////////////
/// \brief A container for many variables that don't need their own versions 
///        for each device.
///////////////////////////////////////////////////////////////////////////////
struct CommonHostData
{
    uint		  nrOfTriangles;        ///< Number of triangles.
    uint		  nrOfVertices;         ///< Number of vertices.
    uint		  nrOfUniqueMaterials;  ///< Number of different materials.
    double3		  minVertex;            ///< Minimum bounding box corner.
    double3		  maxVertex;            ///< Maximum bounding box corner.
    double		  voxelLength;          ///< Distance between voxel centers.
    Bounds<uint3> resolution;           ///< Dimensions of the voxelization.

    // Surface voxelization data.

    uint		  start1DTris;          ///< Index where 1D Triangles start.
    uint		  end1DTris;            ///< Index where 1D Triangles end.
    uint		  start2DTris;          ///< Index where 2D Triangles start.
    uint		  end2DTris;            ///< Index where 2D Triangles end.
    uint		  start3DTris;          ///< Index where 3D Triangles start.
    uint		  end3DTris;            ///< Index where 3D Triangles end.
};

///////////////////////////////////////////////////////////////////////////////
/// Traverses each vertex of the triangle and finds the minimum and maximum
/// coordinate components and uses them to construct the minimum and maximum
/// corners of the bounding box.
/// This version uses \p double3.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ void getTriangleBounds
    ( double3 const * vertices ///< [in] Vertices of the triangle.
    , Bounds<double3> & bounds ///< [out] Bounding box of the triangle.
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
/// The minimum corner is floored and the maximum corner is ceiled.
/// Expects the triangle's bounding box to be made of \p double3 and returns a
/// bounding box made of \p uint3.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ void getVoxelBounds
    ( Bounds<double3> const & triBB   ///< [in] Triangle's bounding box.
    , double3 const & modelBBMin      /**< [in] Minimum corner of the device's
                                               voxelization space. */
    , Bounds<uint3> & voxBB          /**< [out] Triangle's bounding
                                                box in voxel coordinates. */
    , double d                        ///< [in] Distance between voxel centers.
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
/// Clipping value function.
/// From: http://stackoverflow.com/questions/9323903/most-efficient-elegant-way-to-clip-a-number
///////////////////////////////////////////////////////////////////////////////
template <typename T> __host__ __device__
T clip(const T& n, const T& lower, const T& upper) {
  return std::max(lower, std::min(n, upper));
}

///////////////////////////////////////////////////////////////////////////////
/// Similar function to getVoxelBounds(), but the bounding boxes are calculated
/// according to where the triangle resides (without extra voxels that are
/// outside the triangle's bounding box).
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ void getVoxelBounds_v2
    ( Bounds<double3> const & triBB   ///< [in] Triangle's bounding box.
    , double3 const & modelBBMin      /**< [in] Minimum corner of the device's
                                               voxelization space. */
    , Bounds<uint3> & voxBB          /**< [out] Triangle's bounding
                                                box in voxel coordinates. */
    , double d                      /**< [in] Distance between voxel centers.*/
	, Bounds<uint3> const & voxelization_limits ///< [in] The limits of the
									/// voxelization - the voxel BB will not be
									/// off this domain.
    )
{
    /* Convert to fractional voxel coordinates, then take their floor for both
     * the minimum and the maximum. */
	Bounds<double3> triBB_voxel_space;
	triBB_voxel_space.min = make_double3( (triBB.min.x - modelBBMin.x) / d,
										  (triBB.min.y - modelBBMin.y) / d,
										  (triBB.min.z - modelBBMin.z) / d );
	triBB_voxel_space.max = make_double3( (triBB.max.x - modelBBMin.x) / d,
										  (triBB.max.y - modelBBMin.y) / d,
										  (triBB.max.z - modelBBMin.z) / d );

    voxBB.min = make_uint3( uint( floorf( triBB_voxel_space.min.x ) )
                          , uint( floorf( triBB_voxel_space.min.y ) )
                          , uint( floorf( triBB_voxel_space.min.z ) ));
    voxBB.max = make_uint3( uint( floorf( triBB_voxel_space.max.x ) )
                          , uint( floorf( triBB_voxel_space.max.y ) )
                          , uint( floorf( triBB_voxel_space.max.z ) ) );
    /* Now, if the minimum  touches the voronoi cell surface, then a conserva
     * tive approach will decrement it. */
    // Let's use this one... Sound ok-ish for a double precision
    double epsilon__ = 1e-10;
    // See http://stackoverflow.com/questions/15313808/how-to-check-if-float-is-a-whole-number
    if (  fabs(triBB_voxel_space.min.x - round(triBB_voxel_space.min.x)) < epsilon__ ){
		voxBB.min.x = clip(voxBB.min.x - 1, voxelization_limits.min.x, voxelization_limits.max.x);
	}
	if (  fabs(triBB_voxel_space.min.y - round(triBB_voxel_space.min.y)) < epsilon__ ){
		voxBB.min.y = clip(voxBB.min.y - 1, voxelization_limits.min.y, voxelization_limits.max.y);
	}
	if (  fabs(triBB_voxel_space.min.z - round(triBB_voxel_space.min.z)) < epsilon__ ){
		voxBB.min.z = clip(voxBB.min.z - 1, voxelization_limits.min.z, voxelization_limits.max.z);
	}
	/*Similar, for the maximum where the voxel max will be incremented. */
	if (  fabs(triBB_voxel_space.max.x - round(triBB_voxel_space.max.x)) < epsilon__ ){
		voxBB.max.x = clip(voxBB.max.x + 1, voxelization_limits.min.x, voxelization_limits.max.x);
	}
	if (  fabs(triBB_voxel_space.max.y - round(triBB_voxel_space.max.y)) < epsilon__ ){
		voxBB.max.y = clip(voxBB.max.y + 1, voxelization_limits.min.y, voxelization_limits.max.y);
	}
	if (  fabs(triBB_voxel_space.max.z - round(triBB_voxel_space.max.z)) < epsilon__ ){
		voxBB.max.z = clip(voxBB.max.z + 1, voxelization_limits.min.z, voxelization_limits.max.z);
	}
    return;
}

} // End namespace vox

#endif
