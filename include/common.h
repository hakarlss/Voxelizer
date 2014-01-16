#ifdef _WIN32
#pragma once
#endif

#ifndef VOX_COMMON_H
#define VOX_COMMON_H

#if defined(unix) || defined(__unix__) || defined(__unix)
#include <tr1/cstdint>
#else
#include <cstdint>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <sm_12_atomic_functions.h>

#include <exception>
#include <string>
#include <iostream>

#include <ctime>

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
///
///////////////////////////////////////////////////////////////////////////////
template <class T, int I>
struct NodeBidEquals
{
    //neq_x( T value ): x(value) {}

    __host__ __device__ bool operator()( const T & c ) { return c.bid() == I; }

    //T x;
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
///
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
///
///////////////////////////////////////////////////////////////////////////////
class HashMap
{
public:
    __host__ __device__
    HashMap() {}

    __host__ __device__
    HashMap( uint size
           , int maxProbes = 100
           , uint p = 982451653
           ) : table(NULL)
             , size(size)
             , p(p)
             , maxProbes(maxProbes) {}

    __host__ __device__
    HashMap( const HashMap & other ): size(other.size)
                                    , maxProbes(other.maxProbes)
                                    , p(other.p)
                                    , table(other.table) {}

    __host__ __device__
    ~HashMap() {}

#ifdef __CUDACC__
    __device__
    bool insert( uint32_t id, uint32_t value )
    {
        for ( int attempt = 1; attempt < maxProbes; ++attempt )
        {
            uint32_t hashCode = hash( id, attempt );
            
            uint64_t old = 
                atomicCAS( &table[hashCode]
                         , HashMap::EMPTY
                         , (uint64_t(id) << 32) | uint64_t(value) 
                         );

            if ( old == HashMap::EMPTY )
            {
                return true;
            }
        }
        return false;
    }
#endif

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

    __host__
    void renewHashPrime()
    {
        p = primeList[(p*p) % 204];
    }

    __host__
    void allocate()
    {
        cudaMalloc( &table, sizeof(uint64_t)*size );
        cudaMemset( table, UINT32_MAX, sizeof(uint64_t)*size );
    }

    __host__
    void allocateHost()
    {
        table = new uint64_t[size];
        for ( uint i = 0; i < size; ++i )
            table[i] = HashMap::EMPTY;
    }

    __host__
    void deallocate()
    {
        cudaFree( table );
    }

    __host__
    void deallocateHost()
    {
        delete[] table;
    }

    __host__
    void clearHost()
    {
        for ( uint i = 0; i < size; ++i )
        {
            table[i] = HashMap::EMPTY;
        }
    }

    __host__
    void convertToHostMemory()
    {
        uint64_t * devPtr = table;
        table = new uint64_t[size];
        cudaMemcpy( table, devPtr, size * sizeof(uint64_t), cudaMemcpyDeviceToHost );
        cudaFree( devPtr );
    }

    static const uint64_t EMPTY = UINT64_MAX;

private:
    uint64_t * table;
    uint32_t size, p;
    int maxProbes;

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

} // End namespace vox

#endif
