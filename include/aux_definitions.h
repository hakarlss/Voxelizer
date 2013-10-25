#ifndef _VOX_AUX_DEFS_H_
#define _VOX_AUX_DEFS_H_
                                                                               
#include <cuda.h>
#include <cuda_runtime.h>
#include <exception>
#include <string>
#include <stdint.h>

#include <map>
#include <iterator>
#include <iostream>
#include <memory>
#include <vector>

namespace vox {

typedef uint32_t uint;
typedef uint16_t ushort;
typedef uint8_t uchar;

#define VOX32 1 // Internal representation of the voxelization.
//#define VOX64 1 // 64-bit requires compute 3.5+ and is not implemented yet.

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

/**************************************************************************//**
 * \brief Describes the bounding box of a triangle. 
 * 
 * Used in the surface voxelizer to mark and sort triangles by their bounding 
 * box.
 *****************************************************************************/
enum BBType
{
    BBType_Degen = 0,   ///< Degenerate triangle.
    BBType_1D,          ///< One dimensional bounding box.
    BBType_2D,          ///< Two dimensional bounding box.
    BBType_3D           ///< Full, three dimensional bounding box.
};
/**************************************************************************//**
 * \brief Describes one of the three main axes. 
 * 
 * Mostly deprecated now.
 *****************************************************************************/
enum MainAxis
{
    xAxis = 0,  ///< X-axis.
    yAxis,      ///< Y-axis.
    zAxis       ///< Z-axis.
};
/**************************************************************************//**
 * \brief Compiles a few pieces of information about a triangle. 
 * 
 * Used in the surface voxelizer.
 *****************************************************************************/
struct TriData
{
    BBType bbType;      ///< Bounding box type of the triangle.
    MainAxis domAxis;   /**< \brief Dominant axis, i.e. which main axis of the 
                                    normal is the largest. */
    uint nrOfVoxCols;   /**< \brief How many voxel columns the triangle spans 
                                    along its dominant axis. */
};
/**************************************************************************//**
 * \brief Should be deprecated.
 *****************************************************************************/
enum NodeType
{
    ShortNodeType = 0,
    LongNodeType
};
/**************************************************************************//**
 * \brief Struct that collects the data needed to voxelize, including the plane
 * overlap test.
 *****************************************************************************/
struct TestData
{
    float3 n;		///< Triangle normal.
    float2 d;		///< Plane overlap distances.
    float2 ne[9];	///< Edge normals.
    float de[9];	///< Edge distances.
    float3 p;		///< Voxel minimum corner.
};
/**************************************************************************//**
 * \brief Struct that collects the data needed to voxelize in the special case 
 * of a triangle with a 2D bounding box, but without the plane overlap test.
 *****************************************************************************/
struct TestData2D
{
    float3 n;		///< Triangle normal.
    float2 ne[3];	///< Edge normals.
    float de[3];	///< Edge distances.
    float3 p;		///< Voxel minimum corner.
};
/**************************************************************************//**
 * \brief Struct that collects the data needed to voxelize in the general case, 
 * but without the plane overlap test.
 *****************************************************************************/
struct TestData3D
{
    float3 n;		///< Triangle normal.
    float2 ne[9];	///< Edge normals.
    float de[9];	///< Edge distances.
    float3 p;		///< Voxel minimum corner.
};
/**************************************************************************//**
 * \brief Struct that only collects the edge normals and distances used in the 
 * overlap testing.
 *****************************************************************************/
struct OverlapData
{
    float2 ne[3];	///< Edge normals.
    float de[3];	///< Edge distances.
};
/**************************************************************************//**
 * \brief Struct to represent anything that can be described with a minimum and 
 * maximum value. 
 * 
 * Used extensively to represent bounding boxes in three, two 
 * and one dimensions.
 *
 * \tparam T Type of the \p min and \p max members. 
 *****************************************************************************/
template <class T>
struct Bounds
{
    T min;  ///< Minimum value of the bound.
    T max;  ///< Maximum value of the bound.
};
/**************************************************************************//**
 * \brief Functor that returns \p true if the given \p TriData encoding 
 * represents the desired \p ::BBType. 
 *
 * Used with the thrust counting alogorithm to calculate where 
 * the different \p ::BBType begin and end once they've been sorted.
 *
 * \tparam T The wanted ::BBType.
 * \tparam B the particular amount of bits that need to be shifted to get at 
 *           the \p ::BBType from the encoding. It is set to <tt>VOX_BPI - 2</tt> 
 *           in the code where it is used.
 *****************************************************************************/
template <BBType T, int B>
struct is_BB
{
    /**********************************************************************//**
     * \brief Tests if the encoded \p TriData equals the desired \p ::BBType.
     * \param[in] x The \p TriData encoded into an \p uint.
     * \return \p true if \p x decodes to \p T.
     *************************************************************************/
    __host__ __device__ bool operator()(const uint &x)
    {
        return (x >> B) == T;
    }
};
/**************************************************************************//**
 * \brief Exception class used by the Voxelizer. 
 *
 * Nothing fancy, its just using a \p string instead of a pointer to \p char.
 *****************************************************************************/
class Exception : public std::exception {
public:
    ~Exception() throw() {}
    /** \brief Constructor for strings.
        
        \param[in] message Error message. */
    Exception(std::string message): msg(message) {}
    /** \brief Constructor for C-style strings.
        
        \param[in] message Error message. */
    Exception(char const * message): msg(std::string(message)) {}
    /** \brief Gets the error message.
        
        \return Error message. */
    char const * what() const throw() { return msg.c_str(); }
private:
    std::string const msg; ///< Error message.
};
/**************************************************************************//**
 * \brief Allocates an object and returns it in a \p unique_ptr.
 *
 * \tparam Type or class of the object.
 * \return \p std::unique_ptr containing the object.
 *****************************************************************************/
template<typename T>
std::unique_ptr<T> make_unique()
{
    return std::unique_ptr<T>( new T() );
}
/**************************************************************************//**
 * \brief Allocates an array of objects and returns it in a \p unique_ptr.
 *
 * \tparam Type or class of the object.
 * \return \p std::unique_ptr containing the array of objects.
 *****************************************************************************/
template<typename T>
std::unique_ptr<T[]> make_unique( std::size_t count )
{
    return std::unique_ptr<T[]>( new T[count]() );
}
/**************************************************************************//**
 * \brief <em>Unique smart pointer</em> implementation for CUDA memory.
 *
 * Functions largely in the same way as std::unique_ptr, but comes with 
 * automatic allocation and deallocation of CUDA memory upon construction and 
 * destruction. Also implements a few memory operations, such as copying to and 
 * from host memory and zeroing all data.
 *
 * \tparam Type or class of the object.
 *****************************************************************************/
template <typename T>
class DevPtr
{
public:
    /// Default constructor.
    DevPtr(): _ptr( nullptr ), _bytes( 0 ), _device( -1 ) {}
    /**********************************************************************//**
     * \brief Constructor that allocates memory.
     *
     * \param[in] count Number of elements to be allocated.
     * \param[in] device Which device the memory should be allocated on.
     *                   Default value uses the currently active device.
     *************************************************************************/
    DevPtr( std::size_t count, int device = -1 ): _ptr( nullptr )
                                                , _bytes( sizeof(T)*count )
    {
        setDevice( device );
        allocate();
    }
    /**********************************************************************//**
     * \brief Constructor that takes a pointer to already allocated memory.
     *
     * Make sure \p count and \p device are correct. If they aren't, CUDA will 
     * likely crash when performing certain operations, such as copying or 
     * unallocating the memory.
     *
     * \param[in] ptr Device pointer to allocated device memory.
     * \param[in] count Number of elements that have been allocated.
     * \param[in] device Which device the memory has been allocated on.
     *                   Default value uses the currently active device.
     *************************************************************************/
    DevPtr( T * ptr, std::size_t count, int device = -1 )
        : _ptr( ptr )
        , _bytes( sizeof(T)*count )
    {
        setDevice( device );
    }
    /**********************************************************************//**
     * \brief Move constructor.
     *
     * Simply swaps all data between \p this and \p rhs.
     *
     * \param[in,out] rhs The \p DevPtr which should be moved to \p this.
     *************************************************************************/
    DevPtr( DevPtr && rhs )
    {
        std::swap( _ptr, rhs._ptr );
        std::swap( _bytes, rhs._bytes );
        std::swap( _device, rhs._device );
    }
    /// Default destructor: Unallocates memory on card.
    ~DevPtr() { unallocate(); }
    /**********************************************************************//**
     * \brief Move-assignment operator.
     *
     * Simply swaps all data between \p this and \p rhs.
     *
     * \param[in,out] rhs The \p DevPtr which should be moved to \p this.
     *
     * \return Reference to \p this.
     *************************************************************************/
    DevPtr & operator=( DevPtr && rhs )
    {
        std::swap( _ptr, rhs._ptr );
        std::swap( _bytes, rhs._bytes );
        std::swap( _device, rhs._device );

        return *this;
    }
    /**********************************************************************//**
     * \brief Copies contents to the given destination.
     *
     * Make sure the data allocated at \p dstPtr is large enough to fit the 
     * data on the device.
     *
     * \param[in] dstPtr Pointer to memory on the host to where the data on 
     *                   the device should be copied to.
     *************************************************************************/
    void copyTo( T * dstPtr )
    {
        int currentDevice = 0;
        cudaGetDevice( &currentDevice );
        if ( _device != currentDevice )
        {
            cudaSetDevice( _device );
            cudaMemcpy( dstPtr
                      , _ptr
                      , _bytes
                      , cudaMemcpyDeviceToHost );
            cudaSetDevice( currentDevice );
        }
        else
            cudaMemcpy( dstPtr
                      , _ptr
                      , _bytes
                      , cudaMemcpyDeviceToHost );
    }
    /**********************************************************************//**
     * \brief Copies contents from the given source.
     *
     * Make sure the data allocated at \p srcPtr is small enough to fit the 
     * allocated memory this \p DevPtr has.
     *
     * \param[in] srcPtr Pointer to memory on the host from where the device 
     *                   should copy.
     *************************************************************************/
    void copyFrom( T * srcPtr )
    {
        int currentDevice = 0;
        cudaGetDevice( &currentDevice );
        if ( _device != currentDevice )
        {
            cudaSetDevice( _device );
            cudaMemcpy( _ptr
                      , srcPtr
                      , _bytes
                      , cudaMemcpyHostToDevice );
            cudaSetDevice( currentDevice );
        }
        else
            cudaMemcpy( _ptr
                      , srcPtr
                      , _bytes
                      , cudaMemcpyHostToDevice );
    }
    /// Sets all bits to zero.
    void zero()
    {
        int currentDevice = 0;
        cudaGetDevice( &currentDevice );
        if ( _device != currentDevice )
        {
            cudaSetDevice( _device );
            cudaMemset( _ptr, 0, _bytes );
            cudaSetDevice( currentDevice );
        }
        else
            cudaMemset( _ptr, 0, _bytes );
    }
    /// Unallocates the data and resets to default state.
    void reset() { unallocate(); }
    /**********************************************************************//**
     * \brief Unallocates the current memory and then allocates new memory with 
     *        the given parameters.
     *
     * \param[in] count Number of elements that the new array should contain.
     * \param[in] device Which device the memory should be allocated on.
     *************************************************************************/
    void reset( std::size_t count, int device = -1 )
    {
        unallocate();

        _bytes = sizeof(T) * count;
        setDevice( device );

        allocate();
    }
    /**********************************************************************//**
     * \brief Unallocates the current memory and assigns a new pointer, which 
     *        points to already allocated memory.
     *
     * Make sure \p count and \p device are correct. If they aren't, CUDA will 
     * likely crash when performing certain operations, such as copying or 
     * unallocating the memory.
     *
     * \param[in] ptr Pointer to already allocated memory.
     * \param[in] count Number of elements that the allocated array contains.
     * \param[in] device Which device the memory has been allocated on.
     *************************************************************************/
    void reset( T * ptr, std::size_t count, int device = -1 )
    {
        unallocate();

        _bytes = sizeof(T) * count;
        setDevice( device );
        _ptr = ptr;
    }
    /**********************************************************************//**
     * \brief Stops managing the memory on the device.
     *
     * Calling this function returns the pointer to device memory and prevents 
     * the DevPtr from deallocating it.
     *
     * \return Pointer to the allocated device memory.
     *************************************************************************/
    T * release()
    {
        T * result = _ptr;

        _ptr = nullptr;

        return result;
    }
    /**********************************************************************//**
     * \brief Returns the pointer to device memory.
     *
     * \return Pointer to the allocated device memory.
     *************************************************************************/
    T * get() { return _ptr; } 
    /**********************************************************************//**
     * \brief Returns the pointer to device memory.
     *
     * \return Pointer to the allocated device memory.
     *************************************************************************/
    std::size_t bytes() { return _bytes; }
    /**********************************************************************//**
     * \brief Returns the pointer to device memory.
     *
     * \return Pointer to the allocated device memory.
     *************************************************************************/
    std::size_t size() { return _bytes / sizeof(T); }
private:
    /// Disabled copy constructor.
    DevPtr( const DevPtr & rhs ) {}
    /// Disabled copy assignment operator.
    DevPtr & operator=( const DevPtr & rhs ) { return *this; }
    /// Allocates memory on the device.
    void allocate()
    {
        int currentDevice = 0;
        cudaGetDevice( &currentDevice );
        if ( _device != currentDevice )
        {
            cudaSetDevice( _device );
            cudaMalloc( &_ptr, _bytes );
            cudaSetDevice( currentDevice );
        }
        else
            cudaMalloc( &_ptr, _bytes );
    }
    /// Unallocates the memory that was allocated upon construction.
    void unallocate()
    {
        if ( _ptr == nullptr )
            return;

        int currentDevice;
        cudaGetDevice( &currentDevice );

        if ( currentDevice != _device )
        {
            cudaSetDevice( _device );
            cudaFree( _ptr );
            cudaSetDevice( currentDevice );
        }
        else
            cudaFree( _ptr );

        _ptr = nullptr;
    }
    /**********************************************************************//**
     * \brief Initializes the device.
     *
     * If the given device id is negative, then this \p DevPtr is associated 
     * with the currently active device. Otherwise, the given device id is 
     * used.

     * \param[in] device Which device id this \p DevPtr should use.
     *************************************************************************/
    void setDevice( int device )
    {
        if ( device < 0 )
            cudaGetDevice( &_device );
        else
            _device = device;
    }

    T * _ptr;            ///< Device pointer.
    std::size_t _bytes;  ///< Allocated bytes.
    int _device;         ///< Device id.
};

/**************************************************************************//**
 * \brief A container for all variables that need to be duplicated for each 
 * additional device used.
 *
 * \tparam Node type
 *****************************************************************************/
template <class Node>
struct DevContext
{
    DevContext() {}
    ~DevContext() {}

    int dev;						///< Device number.
    cudaDeviceProp devProps;		///< Device properties.
    size_t freeMem;					///< Free memory.
    size_t maxMem;					///< Maximum memory.
    uint blocks;				    ///< Number of blocks.
    uint threads;					///< Number of threads per block.

    // GPU pointers.
    
    DevPtr<float> vertices_gpu;          ///< Vertex data on GPU.
    DevPtr<uint> indices_gpu;		     ///< Index data on GPU.
    DevPtr<uchar> materials_gpu;		 ///< Material data on GPU.
    DevPtr<uint> tileOverlaps_gpu;	     ///< Tile overlap buffer on GPU.
    DevPtr<uint> workQueueTriangles_gpu; ///< Work queue triangle ids on GPU.
    DevPtr<uint> workQueueTiles_gpu;	 ///< Work queue tile ids on GPU.
    DevPtr<uint> offsetBuffer_gpu;	     ///< Offset buffer on GPU.
    DevPtr<uint> tileList_gpu;	         ///< Compacted tile lsit tiles on GPU.
    DevPtr<uint> tileOffsets_gpu;        /**< Compacted tile list offsets 
                                              on GPU. */
    DevPtr<VoxInt> voxels_gpu;		     ///< Voxel data as integers on GPU.
    DevPtr<Node> nodes_gpu;			     ///< Node data on GPU.
    DevPtr<Node> nodesCopy_gpu;          /**< Copy of nodes used in 
                                              slicing on GPU. */
    DevPtr<bool> error_gpu;			     ///< Error boolean on GPU.
    DevPtr<uint> triangleTypes_gpu;      /**< Triangle classification 
                                              array on GPU. */
    DevPtr<uint> sortedTriangles_gpu;    /**< Triangle ids sorted by 
                                              classification on GPU. */
    
    // Host pointers.
    std::vector<uint> tileOverlaps;       ///< Tile overlaps on host.
    std::vector<uint> offsetBuffer;       ///< Offset buffer on host.
    std::vector<uint> workQueueTriangles; ///< Work queue triangles on host.
    std::vector<uint> workQueueTiles;     ///< Work queue tiles on host.
    std::vector<uint> tileList;           ///< Tile list on host.
    std::vector<uint> tileOffsets;        ///< Tile offsets on host.
    bool error;                           ///< Error boolean on host.

    // Host variables.
    /** \brief Minimum corner of the model's bounding box. 
        
        Matches exactly the voxelization space, meaning that there is no 
        padding or anything extra that won't make it to the final 
        voxelization. */
    double3		  minVertex;
    /** \brief Extended minimum corner of the model's bounding box. 
               
        If there is a neighboring subspace to the left or below, then this 
        differs from minVertex. The extension also happens during slicing. */
    double3       extMinVertex;
    uint		  workQueueSize;            ///< Size of the work queue.
    uint		  maxWorkQueueSize;         /**< \brief How much is allocated 
                                                        for the work queue. */
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

private:
    DevContext( const DevContext & rhs ) {}
    DevContext & operator=( const DevContext & rhs ) { return *this; }
};
/**************************************************************************//**
 * \brief A container for many variables that don't need their own versions for 
 *        each device.
 *****************************************************************************/
struct HostContext
{
    std::vector<float> vertices;  ///< Vertices of the model.
    std::vector<uint> indices;    ///< Triangle indices for the model.
    std::vector<uchar> materials; ///< Triangle-material mappings.

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
