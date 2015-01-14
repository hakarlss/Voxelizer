#ifdef _WIN32
#pragma once
#endif

#ifndef VOX_HOST_SUPPORT_H
#define VOX_HOST_SUPPORT_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <exception>
#include <string>
#include <sstream>

#include <iostream>
#include <vector>

#include <boost/shared_array.hpp>

#include "common.h"

namespace vox {

template <typename T>
std::string toString( T val )
{ 
    std::stringstream ss;
    ss << val;
    return ss.str();
}

template <typename T>
boost::shared_array<T> make_shared( int count )
{
    return boost::shared_array<T>( new T[count]() );
}

/// Options or state-tracking variables for the voxelizer.
struct Options
{
    bool nodeOutput;           ///< Whether to output integers or \p Nodes.
    bool materials;            ///< Do \p Nodes have materials associated?
    bool slices;               ///< If a \a slice is being voxelized.
    uint slice;                ///< Which \a slice is being voxelized.
    int  sliceDirection;       ///< What kind of \a slice is being made.
    bool slicePrepared;        ///< Has the first \a slice been produced?
    bool sliceOOB;             ///< Is the \a slice index out of bounds?
    bool verbose;              ///< Should messages be output?
    bool printing;             ///< Should various data be printed to file?
    bool voxelDistanceGiven;   ///< \brief \p True if the user has specified 
                               ///<        the distance between voxels.
                               ///<
    bool simulateMultidevice;  ///< \brief Simulates multiple devices on one 
                               ///<        device.
                               ///<
    bool orientations;  	   ///< \brief Whether or not the default voxelization
							   ///< worker calculates the orientations
    bool displace_VoxSpace_dX_2;///< \brief Whether or not the triangle-voxel
							   ///< comparison point is the center of the voxel
							   ///< or the lower corner. This is achieved by
							   ///< displacing the voxelization space by dX/2.
	bool is_displaced;		   ///< \brief If the space has already been
							   ///< incremented by dX/2. Avoids multiple dis-
							   ///< splacements among sub-domains. Used only if
							   ///< displace_VoxSpace_dX_2 is true.
};
///////////////////////////////////////////////////////////////////////////////
/// \brief Exits the program, printing an error message if an error occurred.
///
/// \param[in] code Error code of some CUDA function.
/// \param[in] file Filename of the source this function is called in.
/// \param[in] line Line number where this function is called.
/// \param[in] abort \p true to abort on any error, \p false to continue.  
///////////////////////////////////////////////////////////////////////////////
inline void gpuAssert( cudaError_t code
                     , const char * file
                     , int line
                     , bool abort = true )
{
    if ( code != cudaSuccess )
    {
        std::string msg = std::string( file ) + "(" + toString( line ) + 
            "): " + std::string( cudaGetErrorString( code ) );

        if ( abort ) throw Exception( msg );
    }
}
/// Macro that embeds source information into the gpuAsser-function.
#define GPU_ERRCHK(ans) gpuAssert((ans), __FILE__, __LINE__)
///////////////////////////////////////////////////////////////////////////////
/// \brief <em>Unique smart pointer</em> implementation for CUDA memory.
///
/// Functions largely in the same way as std::unique_ptr, but comes with 
/// automatic allocation and deallocation of CUDA memory upon construction and 
/// destruction. Also implements a few memory operations, such as copying to 
/// and from host memory and zeroing all data.
///
/// \tparam Type or class of the object.
///////////////////////////////////////////////////////////////////////////////
template <typename T>
class DevPtr
{
public:
    /// Default constructor.
    DevPtr() throw(): _ptr( NULL ), _bytes( 0 ), _device( 0 ) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructor that allocates memory.
    ///
    /// \throws Exception if CUDA reports an error. Tries to make sure that any 
    ///                   memory that may have been allocated in this 
    ///                   constructor is freed before throwing.
    /// 
    /// \param[in] count Number of elements to be allocated.
    /// \param[in] device Which device the memory should be allocated on.
    ///////////////////////////////////////////////////////////////////////////
    DevPtr( std::size_t count, int device )
          : _ptr( NULL ), _bytes( sizeof(T)*count ), _device( device )
    {
        try { allocate(); }
        catch ( ... ) { GPU_ERRCHK( cudaFree( _ptr ) ); throw; }
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructor that takes a pointer to already allocated memory.
    ///
    /// Make sure \p count and \p device are correct. If they aren't, CUDA will 
    /// likely crash when performing certain operations, such as copying or 
    /// unallocating the memory.
    /// 
    /// \param[in] ptr Device pointer to allocated device memory.
    /// \param[in] count Number of elements that have been allocated.
    /// \param[in] device Which device the memory has been allocated on.
    ///////////////////////////////////////////////////////////////////////////
    DevPtr( T * ptr, std::size_t count, int device ) throw()
          : _ptr( ptr ), _bytes( sizeof(T)*count ), _device ( device )
    {
    }
    /// Default destructor: Unallocates memory on card.
    ~DevPtr() throw()
    {
        try { unallocate(); } catch ( ... ) {}
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Copies contents to the given destination.
    ///
    /// Make sure the data allocated at \p dstPtr is large enough to fit the 
    /// data on the device.
    ///
    /// \throws Exception if CUDA reports an error.
    /// 
    /// \param[in] dstPtr Pointer to memory on the host to where the data on 
    ///                   the device should be copied to.
    ///////////////////////////////////////////////////////////////////////////
    void copyTo( T * dstPtr )
    {
        int currentDevice = 0;
        GPU_ERRCHK( cudaGetDevice( &currentDevice ) );
        if ( _device != currentDevice )
        {
            GPU_ERRCHK( cudaSetDevice( _device ) );
            GPU_ERRCHK( cudaMemcpy( dstPtr
                                  , _ptr
                                  , _bytes
                                  , cudaMemcpyDeviceToHost ) );
            GPU_ERRCHK( cudaSetDevice( currentDevice ) );
        }
        else
            GPU_ERRCHK( cudaMemcpy( dstPtr
                                  , _ptr
                                  , _bytes
                                  , cudaMemcpyDeviceToHost ) );
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Copies contents from the given source.
    ///
    /// Make sure the data allocated at \p srcPtr is small enough to fit the 
    /// allocated memory this \p DevPtr has.
    ///
    /// \throws Exception if CUDA reports an error.
    ///
    /// \param[in] srcPtr Pointer to memory on the host from where the device 
    ///                   should copy.
    ///////////////////////////////////////////////////////////////////////////
    void copyFrom( T * srcPtr )
    {
        int currentDevice = 0;
        GPU_ERRCHK( cudaGetDevice( &currentDevice ) );
        if ( _device != currentDevice )
        {
            GPU_ERRCHK( cudaSetDevice( _device ) );
            GPU_ERRCHK( cudaMemcpy( _ptr
                                  , srcPtr
                                  , _bytes
                                  , cudaMemcpyHostToDevice ) );
            GPU_ERRCHK( cudaSetDevice( currentDevice ) );
        }
        else
            GPU_ERRCHK( cudaMemcpy( _ptr
                                  , srcPtr
                                  , _bytes
                                  , cudaMemcpyHostToDevice ) );
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Sets all bits to zero.
    ///
    /// \throws Exception if CUDA reports an error.
    ///////////////////////////////////////////////////////////////////////////
    void zero()
    {
        int currentDevice = 0;
        GPU_ERRCHK( cudaGetDevice( &currentDevice ) );
        if ( _device != currentDevice )
        {
            GPU_ERRCHK( cudaSetDevice( _device ) );
            GPU_ERRCHK( cudaMemset( _ptr, 0, _bytes ) );
            GPU_ERRCHK( cudaSetDevice( currentDevice ) );
        }
        else
            GPU_ERRCHK( cudaMemset( _ptr, 0, _bytes ) );
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Sets all bits to zero.
    ///
    /// \throws Exception if CUDA reports an error.
    ///////////////////////////////////////////////////////////////////////////
    void setAllTo( int byte )
    {
        int currentDevice = 0;
        GPU_ERRCHK( cudaGetDevice( &currentDevice ) );
        if ( _device != currentDevice )
        {
            GPU_ERRCHK( cudaSetDevice( _device ) );
            GPU_ERRCHK( cudaMemset( _ptr, byte, _bytes ) );
            GPU_ERRCHK( cudaSetDevice( currentDevice ) );
        }
        else
            GPU_ERRCHK( cudaMemset( _ptr, byte, _bytes ) );
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Unallocates the data and resets to default state.
    ///
    /// \throws Exception if CUDA reports an error.
    ///////////////////////////////////////////////////////////////////////////
    void reset() { unallocate(); }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Unallocates the current memory and then allocates new memory with 
    ///        the given parameters.
    ///
    /// \throws Exception if CUDA reports an error.
    /// 
    /// \param[in] count Number of elements that the new array should contain.
    /// \param[in] device Which device the memory should be allocated on.
    ///////////////////////////////////////////////////////////////////////////
    void reset( std::size_t count, int device )
    {
        unallocate();

        _bytes = sizeof(T) * count;
        _device = device;

        allocate();
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Unallocates the current memory and assigns a new pointer, which 
    ///        points to already allocated memory.
    ///
    /// Make sure \p count and \p device are correct. If they aren't, CUDA will 
    /// likely crash when performing certain operations, such as copying or 
    /// unallocating the memory.
    ///
    /// \throws Exception if CUDA reports an error.
    ///
    /// \param[in] ptr Pointer to already allocated memory.
    /// \param[in] count Number of elements that the allocated array contains.
    /// \param[in] device Which device the memory has been allocated on.
    ///////////////////////////////////////////////////////////////////////////
    void reset( T * ptr, std::size_t count, int device )
    {
        unallocate();

        _bytes = sizeof(T) * count;
        _device = device;
        _ptr = ptr;
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Stops managing the memory on the device.
    ///
    /// Calling this function returns the pointer to device memory and prevents 
    /// the DevPtr from deallocating it.
    ///
    /// \return Pointer to the allocated device memory.
    ///////////////////////////////////////////////////////////////////////////
    T * release() throw()
    {
        T * result = _ptr;

        _ptr = NULL;

        return result;
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the pointer to device memory.
    ///
    /// \return Pointer to the allocated device memory.
    ///////////////////////////////////////////////////////////////////////////
    T * get() const throw() { return _ptr; } 
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the pointer to device memory.
    ///
    /// \return Pointer to the allocated device memory.
    ///////////////////////////////////////////////////////////////////////////
    std::size_t bytes() const throw() { return _bytes; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the pointer to device memory.
    ///
    /// \return Pointer to the allocated device memory.
    ///////////////////////////////////////////////////////////////////////////
    std::size_t size() const throw() { return _bytes / sizeof(T); }
private:
    /// Disabled copy constructor.
    DevPtr( const DevPtr & rhs ) throw() {}
    /// Disabled copy assignment operator.
    DevPtr & operator=( const DevPtr & rhs ) throw() { return *this; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Allocates memory on the device.
    ///
    /// \throws Exception if CUDA reports an error.
    ///////////////////////////////////////////////////////////////////////////
    void allocate()
    {
        int currentDevice = 0;
        GPU_ERRCHK( cudaGetDevice( &currentDevice ) );
        if ( _device != currentDevice )
        {
            GPU_ERRCHK( cudaSetDevice( _device ) );
            GPU_ERRCHK( cudaMalloc( &_ptr, _bytes ) );
            GPU_ERRCHK( cudaSetDevice( currentDevice ) );
        }
        else
            GPU_ERRCHK( cudaMalloc( &_ptr, _bytes ) );
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Unallocates the memory tied to this DevPtr.
    ///
    /// \throws Exception if CUDA reports an error.
    ///////////////////////////////////////////////////////////////////////////
    void unallocate()
    {
        if ( _ptr == NULL )
            return;

        int currentDevice;
        GPU_ERRCHK( cudaGetDevice( &currentDevice ) );

        if ( currentDevice != _device )
        {
            GPU_ERRCHK( cudaSetDevice( _device ) );
            GPU_ERRCHK( cudaFree( _ptr ) );
            GPU_ERRCHK( cudaSetDevice( currentDevice ) );
        }
        else
            GPU_ERRCHK( cudaFree( _ptr ) );

        _ptr = NULL;
    }

    T * _ptr;            ///< Device pointer.
    std::size_t _bytes;  ///< Allocated bytes.
    int _device;         ///< Device id.
};

///////////////////////////////////////////////////////////////////////////////
/// \brief A container for all variables that need to be duplicated for each 
/// additional device used.
///
/// \tparam Node type
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
struct DevContext
{
    /// Default constructor.
    DevContext() {}
    /// Default destructor.
    ~DevContext() {}

    // GPU pointers.
    
    DevPtr<float> vertices_gpu;          ///< Vertex data on GPU.
    DevPtr<uint> indices_gpu;		     ///< Index data on GPU.
    DevPtr<uchar> materials_gpu;		 ///< Material data on GPU.
    DevPtr<uint> tileOverlaps_gpu;	     ///< Tile overlap buffer on GPU.
    DevPtr<uint> workQueueTriangles_gpu; ///< Work queue triangle ids on GPU.
    DevPtr<uint> workQueueTiles_gpu;	 ///< Work queue tile ids on GPU.
    DevPtr<uint> offsetBuffer_gpu;	     ///< Offset buffer on GPU.
    DevPtr<uint> tileList_gpu;	         ///< Compacted tile lsit tiles on GPU.
    DevPtr<uint> tileOffsets_gpu;        ///< \brief Compacted tile list  
                                         ///<        offsets on GPU.
                                         ///<
    DevPtr<VoxInt> voxels_gpu;		     ///< Voxel data as integers on GPU.
    DevPtr<Node> nodes_gpu;			     ///< Node data on GPU.
    DevPtr<Node> nodesCopy_gpu;          ///< \brief Copy of nodes used in 
                                         ///<        slicing on GPU.
                                         ///<
    DevPtr<SNode> surfNodes_gpu;         ///<
    DevPtr<bool> error_gpu;			     ///< Error boolean on GPU.
    DevPtr<uint> triangleTypes_gpu;      ///< \brief Triangle classification 
                                         ///<        array on GPU.
                                         ///<
    DevPtr<uint> sortedTriangles_gpu;    ///< \brief Triangle ids sorted by 
                                         ///<        classification on GPU.
                                         ///<

    HashMap hashMap;                     ///<
    
    // Host pointers.

    std::vector<uint> tileOverlaps;       ///< Tile overlaps on host.
    std::vector<uint> offsetBuffer;       ///< Offset buffer on host.
    std::vector<uint> workQueueTriangles; ///< Work queue triangles on host.
    std::vector<uint> workQueueTiles;     ///< Work queue tiles on host.
    std::vector<uint> tileList;           ///< Tile list on host.
    std::vector<uint> tileOffsets;        ///< Tile offsets on host.
    bool error;                           ///< Error boolean on host.

    // Host variables.

    CommonDevData data;     ///< \brief Contains various variables that need 
                            ///<        to be passed on to kernel functions.
                            ///<

private:
    /// Disabled copy constructor.
    DevContext( const DevContext & rhs ) throw() {}
    /// Disabled copy assignment operator.
    DevContext & operator=( const DevContext & rhs ) throw() { return *this; }
};

template <typename T>
struct SplitData
{
    T counts;
    boost::shared_array<Bounds<T> > splits;
};

} // End namespace vox

#endif
