#pragma once

#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <exception>
#include <string>

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
        std::string msg = cudaGetErrorString( e );
        throw Exception( msg + " @ " + loc );
    }
}
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
