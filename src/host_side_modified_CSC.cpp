#include "voxelizer.h"
#include "common.h"
#include "host_device_interface.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/count.h>

#include <boost/thread.hpp>
#include <boost/bind.hpp>

namespace vox {

///////////////////////////////////////////////////////////////////////////////
/// Constructs the voxelizer with the minimum amount of required data.
/// Does not use the given arrays beyond this constructor: Copies are made
/// and the voxelizer uses them for its own purposes.
///
/// \param[in] _vertices A pointer to an array of vertices. Three consequtive 
///                      floats define the thre coordinates of a vertex.
/// \param[in] _indices  A pointer to an array of indices. Three consecutive 
///                      values define the vertices of a triangle of the model.
/// \param[in] _nrOfVertices The total number of vertices defined in _vertices.
/// \param[in] _nrOfTriangles The total number of triangles defined in 
///                           _indices.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> Voxelizer<Node, SNode>::Voxelizer(
    const float * _vertices, 
    const uint  * _indices, 
    uint		  _nrOfVertices, 
    uint		  _nrOfTriangles ): defaultDevice(0)
                                  , hostVars(CommonHostData())
                                  , options(Options())
{
    this->hostVars.nrOfVertices = _nrOfVertices;
    this->hostVars.nrOfTriangles = _nrOfTriangles;

    this->initVariables();

    // Copy the arrays to own memory so as to not be reliant on their
    // possibly fickle existence.
    this->vertices = 
        std::vector<float>( _vertices, _vertices + 3*_nrOfVertices );
    this->indices = 
        std::vector<uint>( _indices, _indices + 3*_nrOfTriangles );

    this->calculateBoundingBox();
}
///////////////////////////////////////////////////////////////////////////////
/// Constructs the voxelizer as well as supplies material information.
/// Does not use the given arrays beyond this contructor: Copies are made
/// and the voxelizer uses them for its own purposes.
///
/// \param[in] _vertices A pointer to an array of vertices. Three consequtive 
///                      floats define the thre coordinates of a vertex.
/// \param[in] _indices  A pointer to an array of indices. Three consecutive 
///                      values define the vertices of a triangle of the model.
/// \param[in] _materials A pointer to an array of material indices. Each index 
///                       associates a material to a triangle with the same 
///                       index.
/// \param[in] _nrOfVertices The total number of vertices defined in _vertices.
/// \param[in] _nrOfTriangles The total number of triangles defined in 
///                           _indices.
/// \param[in] _nrOfUniqueMaterials The total number of unique materials 
///                                 provided. Some \p Node implementations have 
///                                 a limit.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> Voxelizer<Node, SNode>::Voxelizer(
    float const * _vertices, 
    uint  const * _indices, 
    uchar const * _materials, 
    uint		 _nrOfVertices, 
    uint		 _nrOfTriangles, 
    uint		 _nrOfUniqueMaterials ): defaultDevice(0)
                                       , hostVars(CommonHostData())
                                       , options(Options())
{
    this->hostVars.nrOfVertices = _nrOfVertices;
    this->hostVars.nrOfTriangles = _nrOfTriangles;

    this->initVariables();

    // Copy the arrays to own memory so as to not be reliant on their
    // possibly fickle existence.
    this->vertices = 
        std::vector<float>( _vertices, _vertices + 3*_nrOfVertices );
    this->indices = 
        std::vector<uint>( _indices, _indices + 3*_nrOfTriangles );

    this->calculateBoundingBox();

    this->setMaterials( _materials, _nrOfUniqueMaterials );
}
///////////////////////////////////////////////////////////////////////////////
/// Initializes various variables, including the number of available devices on 
/// the computer, courtesy of CUDA. 
/// they don't point anywhere.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::initVariables()
{
    this->fatalError = false;

    cudaGetDeviceCount( &this->nrOfDevices );

    this->hostVars.resolution.min = make_uint3( 0 );
    this->hostVars.resolution.max = make_uint3( 256, 0, 0 );
    this->nrOfDevicesInUse = 0;
}

///////////////////////////////////////////////////////////////////////////////
/// Deallocates any and all dynamically allocated memory that can be connected
/// to any device. This includes some host memory that has to have 
/// separate copies per device.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::deallocate()
{
    this->devices.reset();
}

///////////////////////////////////////////////////////////////////////////////
/// Only deallocates dynamically allocated memory that is no longer useful 
/// after the plain voxelization phase. It is used to clear up memory so 
/// that subsequent phases, such as surface voxelization and the various 
/// node phases have more memory to work with.
///
/// \param[in] device The device this function applies to.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
void Voxelizer<Node, SNode>::deallocateVoxelizationData( DevContext<Node,SNode> & device )
{
    // Calling reset() on the DevPtr-class causes its destructor to free the 
    // allocated device memory.
    device.tileOverlaps_gpu.reset();
    device.workQueueTriangles_gpu.reset();
    device.workQueueTiles_gpu.reset();
    device.offsetBuffer_gpu.reset();
    device.tileList_gpu.reset();
    device.tileOffsets_gpu.reset();

    // Replacing the vectors should cause them to be destructed, and 
    // consequently have their allocated memory freed.
    device.tileOverlaps = std::vector<uint>();
    device.offsetBuffer = std::vector<uint>();
    device.workQueueTriangles = std::vector<uint>();
    device.workQueueTiles = std::vector<uint>();
    device.tileList = std::vector<uint>();
    device.tileOffsets = std::vector<uint>();
}
///////////////////////////////////////////////////////////////////////////////
/// Checks if the amount of requested devices can be satisfied and allocates
/// the device contexts for the requested devices. Also initializes the 
/// variables in the device contexts to some sensible default values.
/// 
/// \param[in] nrOfUsedDevices The requested number of devices.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::initDevices( uint nrOfUsedDevices )
{
    if ( this->options.verbose )
        std::cout << "Found " << this->nrOfDevices << " CUDA capable "
                     "device(s).\n";

    if ( this->nrOfDevices == 0 )
        throw Exception( "Couldn't find any CUDA capable devices.\n" );

    if ( nrOfUsedDevices > uint(this->nrOfDevices) && 
         !this->options.simulateMultidevice )
        throw Exception( "Not enough devices to satisfy request.\n" );

    // Make as many DevContexts as there are devices being used.
    this->devices.reset( new DevContext<Node,SNode>[nrOfUsedDevices]() );
    this->nrOfDevicesInUse = nrOfUsedDevices;

    for ( int d = 0; d < this->nrOfDevicesInUse; ++d )
    {
        int dev = d;

        if ( this->options.simulateMultidevice )
            dev = this->defaultDevice;

        cudaSetDevice(dev);

        this->devices[d].data.dev = dev;

        cudaGetDeviceProperties( &this->devices[d].data.devProps, 0 );
        cudaMemGetInfo( &this->devices[d].data.freeMem
                      , &this->devices[d].data.maxMem );
        
        // Blocks and threads are currently not optimized per device.
        this->devices[d].data.blocks = 64;
        this->devices[d].data.threads = 512;

        checkCudaErrors( "initDevices" );
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Updates the material to triangle mapping. The supplied array needs to be 
/// of the same format as in the extended constructor, i.e. have one unsigned 
/// char per triangle id. The materials array is copied to local storage, so 
/// it can be safely deleted after the call. Automatically enables the material 
/// calculations.
///
/// \param[in] _materials Array of unsigned characters, where each char 
///                       corresponds to some material id and each index to a 
///                       triangle id.
/// \param[in] _nrOfUniqueMaterials The number of unique material ids in the 
///                                 array. Hard-capped at 256, but individual 
///                                 \p Node implementations may limit it more.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::setMaterials(
    uchar const * _materials, 
    uint		  _nrOfUniqueMaterials )
{
    if (_materials == NULL)
        throw Exception( "No valid materials provided (Received null pointer)"
                         " @ setMaterials." );
    if ( _nrOfUniqueMaterials > uint( Node::maxMat() + 1 ) )
        throw Exception( "The number of unique materials exceeds the "
                         "capabilities of the node type @ setMaterials." );

    this->hostVars.nrOfUniqueMaterials = _nrOfUniqueMaterials;

    // Copy over the materials into a local copy.
    this->materials = 
        std::vector<uchar>( _materials
                          , _materials + this->hostVars.nrOfTriangles );

    // Free any old materials that may have been uploaded to the GPUs.
    for ( int d = 0; d < this->nrOfDevicesInUse; ++d )
        this->devices[d].materials_gpu.reset();

    // Automatically enable the inclusion of material information to the 
    // output.
    this->options.materials = true;
}
///////////////////////////////////////////////////////////////////////////////
/// Excplicitly enables or disables the calculation and inclusion of material 
/// information into the voxelizer's output. Only applies if the output is a 
/// \p Node type: Integer-voxelization cannot contain material information.
///
/// \param[in] _materials \p true to enable materials, \p false to disable 
/// them.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::setMaterialOutput( bool _materials ) throw()
{
    this->options.materials = _materials;
}
///////////////////////////////////////////////////////////////////////////////
/// Explicitly enables or disables the calculation of orientations in the 
/// output of the voxelizer. 
///
/// \param[in] _orientations \p true to enable orientations, \p false to disable 
/// them.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::setOrientationsOutput( bool _orientations ) throw()
{
    this->options.orientations = _orientations;
}
///////////////////////////////////////////////////////////////////////////////
/// Explicitly enables or disables the displacement of the comparison point to
/// the center of the voxel. Otherwise, the lower corner of the voxel is used
/// for comparison.
///
/// \param[in] _displace_VoxSpace_dX_2 \p true to use center of voxel, \p false 
/// to use the lower corner of the voxel.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::setDisplace_VoxSpace_dX_2( bool _displace_VoxSpace_dX_2 ) throw()
{
    this->options.displace_VoxSpace_dX_2 = _displace_VoxSpace_dX_2;
}
///////////////////////////////////////////////////////////////////////////////
/// Explicitly sets whether the dX/2 displacement already occurred.
///
/// \param[in] _is_displaced \p true if displacement occurred.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::setIs_displaced( bool _is_displaced ) throw()
{
    this->options.is_displaced = _is_displaced;
}
///////////////////////////////////////////////////////////////////////////////
/// Calculates the plain, integer voxelization given a \a subspace and some 
/// other parameters. Relies on certain data structures being allocated and 
/// certain values being calculated, so it cannot be called on its own, but is 
/// instead meant to be called from a managing function.
/// After returning, <tt>devices[device].voxels_gpu</tt> has been updated with 
/// voxel data within the space defined by \p yzSubSpace.
///
/// \param[in] yzSubSpace Minimum and maximum voxel coordinates of the \a 
///                       subspace where the voxelization is taking place. The 
///                       x-axis is always fully voxelized, so only the 
///                       (y,z)-coordinates are specified. Upper bound 
///                       exclusive.
/// \param[in] xRes The length of the x-partitions.
/// \param[in] nrOfXSlices How many x-partitions the space is divided into.
/// \param[in] device Which device the voxelization is performed on.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::performVoxelization(
    Bounds<uint2>         yzSubSpace,
    uint				  xRes,
    uint				  nrOfXSlices,
    DevContext<Node,SNode>    & device )
{
	if (this->options.displace_VoxSpace_dX_2 && !this->options.is_displaced)
	{
		// Move space half a voxel so that intersection point is the center of the voxel:
		double3 displace_ext_by = make_double3( 0.5 * this->hostVars.voxelLength, 0.5 * this->hostVars.voxelLength, 0.5 * this->hostVars.voxelLength);
		device.data.extMinVertex += displace_ext_by;
		this->options.is_displaced = true;
	}
	
    // Initialize values here, since the function may be called successively.
    device.data.workQueueSize = 0;
    device.data.firstTriangleWithTiles = -1;

    if (this->options.printing) this->printGeneralInfo( device );

    // Process triangles in parallel, and determine the number of tiles 
    // overlapping each triangle.
    calcTileOverlap( device.data,
                     this->hostVars, 
                     device.vertices_gpu.get(),
                     device.indices_gpu.get(),
                     device.tileOverlaps_gpu.get(),
                     yzSubSpace,
                     this->startTime, 
                     this->options.verbose );

    // Copy the tile overlap information to the host.
    device.tileOverlaps_gpu.copyTo( device.tileOverlaps.data() );

    // Calculate the offset buffer.
    this->prepareForConstructWorkQueue( device );

    // There's nothing to voxelize if the work queue will be empty.
    if (device.data.workQueueSize == 0)
        return;

    // Reuse previously allocated memory if the new data fits.
    // Otherwise reallocate.
    this->reAllocDynamicMem( device, 1.5f );

    if (this->options.printing) this->printTileOverlaps( device, xAxis );

    // Upload offset buffer contents to GPU.
    device.offsetBuffer_gpu.copyFrom( device.offsetBuffer.data() );

    if (this->options.printing) this->printOffsetBuffer( device, xAxis );

    // Process triangles in parallel, and construct the work queue.
    calcWorkQueue( device.data, 
                   this->hostVars, 
                   device.vertices_gpu.get(),
                   device.indices_gpu.get(),
                   device.workQueueTriangles_gpu.get(),
                   device.workQueueTiles_gpu.get(),
                   device.offsetBuffer_gpu.get(),
                   yzSubSpace, 
                   this->startTime, 
                   this->options.verbose );

    if (this->options.printing)
    {
        device.workQueueTriangles_gpu.
            copyTo( device.workQueueTriangles.data() );
        device.workQueueTiles_gpu.copyTo( device.workQueueTiles.data() );

        this->printWorkQueue( device, xAxis );
    }

    // Sort the work queue by tile, instead of by triangle.
    sortWorkQueue( device.data,
                   device.workQueueTriangles_gpu.get(),
                   device.workQueueTiles_gpu.get(),
                   this->startTime, 
                   this->options.verbose );

    if ( this->options.printing )
    {
        device.workQueueTriangles_gpu.
            copyTo( device.workQueueTriangles.data() );
        device.workQueueTiles_gpu.copyTo( device.workQueueTiles.data() );

        this->printSortedWorkQueue( device, xAxis );
    }

    // Compact the work queue so that each tile only appears once and has 
    // an offset into the work queue where the tile's data begins.
    compactWorkQueue( device.data,
                      device.workQueueTiles_gpu.get(),
                      device.tileList_gpu.get(),
                      device.tileOffsets_gpu.get(),
                      this->startTime, 
                      this->options.verbose );

    if ( this->options.printing )
    {
        device.tileList_gpu.copyTo( device.tileList.data() );
        device.tileOffsets_gpu.copyTo( device.tileOffsets.data() );

        this->printCompactedList( device, xAxis );
    }

    // The actual voxelization is done here. Recalculate the voxelization for 
    // each x-partition. Unfortunate, but necessary.
    Bounds<uint3> subSpace;
    for (uint i = 0; i < nrOfXSlices; ++i) {
        subSpace.min = make_uint3( i * xRes
                                 , yzSubSpace.min.x
                                 , yzSubSpace.min.y );
        subSpace.max = make_uint3( (i + 1) * xRes
                                 , yzSubSpace.max.x
                                 , yzSubSpace.max.y );
			
        calcVoxelization( device.data, 
                          this->hostVars, 
                          device.vertices_gpu.get(),
                          device.indices_gpu.get(),
                          device.workQueueTriangles_gpu.get(),
                          device.workQueueTiles_gpu.get(),
                          device.tileList_gpu.get(),
                          device.tileOffsets_gpu.get(),
                          device.voxels_gpu.get(),
                          subSpace, 
                          this->startTime, 
                          this->options.verbose );
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Allocates memory whose size won't change even between successive calls to 
/// the voxelizer, provided the voxelization happens in different subspaces
/// but the same voxel or node array. The allocated memory mainly depends on 
/// the dimensions of the voxel / node array or some property of the input
/// model, such as the number of triangles.
///
/// \param[in] device Which device the allocations happen on.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::allocStaticMem( DevContext<Node,SNode> & device )
{
    // Calculates the allocated dimensions from the bounding box.
    uint3 res = 
        device.data.allocResolution.max - device.data.allocResolution.min;

    // Since the FCC grid is made up of 4 voxel grids, it has 4 times the 
    // nodes, arranged so that the x- and z-resolutions double in size.
    if ( Node::isFCCNode() )
        device.data.nrOfNodes = 2*res.x * res.y * 2*res.z;
    else
        device.data.nrOfNodes = res.x * res.y * res.z;

    // When voxelizing slices along the x-direction, a copy of the nodes
    // needs to be made at some point. This doesn't need to be done when 
    // voxelizing along the other directions.
    if ( this->options.slices && this->options.sliceDirection == 0 && 
         this->options.nodeOutput )
         device.nodesCopy_gpu.reset( device.data.nrOfNodes, device.data.dev );

    // Allocate nodes if we are to produce an array of nodes as output.
    if ( this->options.nodeOutput )
        device.nodes_gpu.reset( device.data.nrOfNodes, device.data.dev );

    // VOX_DIV depends on the size of the integers on the sytem.
    // Currently essentially divides by 32, regardless of architecture.
    uint nrOfVoxelInts = ( res.x * res.y * res.z ) >> VOX_DIV;

    // Allocate memory for the voxels on the selected device.
    device.voxels_gpu.reset( nrOfVoxelInts, device.data.dev );

    // Make all integers in the voxel array zero.
    device.voxels_gpu.zero();

    // Allocates various memory only used when calculating materials.
    if ( this->options.materials )
    {
        device.materials_gpu.reset( this->hostVars.nrOfTriangles
                                  , device.data.dev );

        device.materials_gpu.copyFrom( &this->materials[0] );

        device.triangleTypes_gpu.reset( this->hostVars.nrOfTriangles
                                      , device.data.dev );

        device.triangleTypes_gpu.zero();

        device.sortedTriangles_gpu.reset( this->hostVars.nrOfTriangles
                                        , device.data.dev );
    }

    // Allocate space for various data structures.
    device.vertices_gpu.reset( 3 * this->hostVars.nrOfVertices
                             , device.data.dev );

    device.indices_gpu.reset( 3 * this->hostVars.nrOfTriangles
                            , device.data.dev );

    device.tileOverlaps_gpu.reset( this->hostVars.nrOfTriangles
                                 , device.data.dev );

    device.offsetBuffer_gpu.reset( this->hostVars.nrOfTriangles
                                 , device.data.dev );

    device.vertices_gpu.copyFrom( &this->vertices[0] );

    device.indices_gpu.copyFrom( &this->indices[0] );

    device.tileOverlaps.reserve( this->hostVars.nrOfTriangles );
    
    device.offsetBuffer.reserve( this->hostVars.nrOfTriangles );
}
///////////////////////////////////////////////////////////////////////////////
/// When repeatedly calling \p performVoxelization() in order to consecutively 
/// voxelize parts of the voxel array, allocated memory may be 
/// reused as long as the new data isn't larger than the old one. This 
/// function does nothing if the memory can be reused, and allocates new 
/// memory if the new data requires it. A multiplier can be supplied that 
/// allocates more memory than needed, with the intention of reducing the 
/// subsequent amount of reallocations needed.
/// 
/// \param[in] device The affected device.
/// \param[in] multiplier Directly multiplies the size of any allocated memory 
///                       with the multiplier.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::reAllocDynamicMem(
    DevContext<Node,SNode> & device, 
    float multiplier )
{
    // Most of the allocations depend on the workQueueSize. MaxWorkQueueSize
    // is the current workQueueSize.
    if ( device.data.workQueueSize <= device.data.maxWorkQueueSize )
        return;
    else
    {
        device.data.maxWorkQueueSize = 
            uint( multiplier * float( device.data.workQueueSize ) );
    }

    // Reallocate memory.
    device.workQueueTriangles_gpu.reset( device.data.maxWorkQueueSize
                                       , device.data.dev );

    device.workQueueTiles_gpu.reset( device.data.maxWorkQueueSize
                                   , device.data.dev );

    device.tileList_gpu.reset( device.data.maxWorkQueueSize, device.data.dev );

    device.tileOffsets_gpu.reset( device.data.maxWorkQueueSize
                                , device.data.dev );

    device.workQueueTriangles.reserve( device.data.maxWorkQueueSize );

    device.workQueueTiles.reserve( device.data.maxWorkQueueSize );

    device.tileList.reserve( device.data.maxWorkQueueSize );

    device.tileOffsets.reserve( device.data.maxWorkQueueSize );

    if (this->options.verbose) 
        std::cout << "Allocated work queue size = " << 
        device.data.maxWorkQueueSize << "\n";
}
///////////////////////////////////////////////////////////////////////////////
/// Produces a plain, integer-voxelization given the supplied parameters.
///
/// \throws Exception if CUDA fails for whatever reason.
/// 
/// \param[in] maxDimension The length of the longest side of the model's
///                         bounding box, measured in voxel centers, starting 
///                         from the very edge and ending at the opposite edge.
/// \param[in] devConfig Determines how the y- and z-axes are split among 
///                      different GPUs. A (1,2)-value means that both devices 
///                      will render the entire y-axis, but the z-axis is split 
///                      evenly among the two axes. Multiplying the x and y 
///                      values together gives the amount of spaces the voxel 
///                      array is split into, as well as the required number of 
///                      devices.
/// \param[in] voxSplitRes The internal size of a single voxelization pass. 
///                        Having a smaller size forces the voxelization to be 
///                        run in multiple, consecutive passes. If the size is 
///                        larger than the total voxelization size, then the 
///                        voxelization is performed parallelly in one go. The 
///                        \p voxSplitRes and \p matSplitRes sizes have nothing 
///                        to do with multiple devices, but instead define a 
///                        minimum workload for all devices.
///
/// \return <tt>vector<NodePointer></tt> that contains the device pointers and 
///         other relevant information, such as the size of the array.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
std::vector<NodePointer<Node> > Voxelizer<Node, SNode>::voxelize
    ( uint  maxDimension
    , uint2 devConfig
    , uint3 voxSplitRes )
{
    std::vector<NodePointer<Node> > result;

    if ( Node::isFCCNode() )
    {
        std::cout << "Please use a standard Node-type when producing a "
                     "plain voxelization.\n";
        return result;
    }

    this->options.nodeOutput = false;

    this->setResolution( maxDimension );

    // Voxelize.

    this->voxelizeEntry( devConfig
                       , voxSplitRes
                       , make_uint3( 1024 )
                       , NULL );

    result = this->collectData();

    this->deallocate();

    // Return the pointer to the voxel data.
    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Produces a plain, integer-voxelization given the supplied parameters.
///
/// \throws Exception if CUDA fails for whatever reason.
///
/// \param[in] cubeLength The distance between sampling points, or the length 
///                       of the sides of the voxels.
/// \param[in] devConfig Determines how the y- and z-axes are split among 
///                      different GPUs. A (1,2)-value means that both devices 
///                      will render the entire y-axis, but the z-axis is split 
///                      evenly among the two axes. Multiplying the x and y 
///                      values together gives the amount of spaces the voxel 
///                      array is split into, as well as the required number of 
///                      devices.
/// \param[in] voxSplitRes The internal size of a single voxelization pass. 
///                        Having a smaller size forces the voxelization to be 
///                        run in multiple, consecutive passes. If the size is 
///                        larger than the total voxelization size, then the 
///                        voxelization is performed parallelly in one go. The 
///                        \p voxSplitRes and \p matSplitRes sizes have nothing 
///                        to do with multiple devices, but instead define a 
///                        minimum workload for all devices.
///
/// \return <tt>vector<NodePointer></tt> that contains the device pointers and 
///         other relevant information, such as the size of the array.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
std::vector<NodePointer<Node> >  Voxelizer<Node, SNode>::voxelize( double cubeLength,
                                uint2 devConfig,
                                uint3 voxSplitRes )
{
    std::vector<NodePointer<Node> > result;

    if ( Node::isFCCNode() )
    {
        std::cout << "Please use a standard Node-type when producing a "
                     "plain voxelization.\n";
        return result;
    }

    this->options.nodeOutput = false;
    this->options.voxelDistanceGiven = true;
    this->hostVars.voxelLength = cubeLength;

    // Voxelize.

    this->voxelizeEntry( devConfig
                       , voxSplitRes
                       , make_uint3( 1024 )
                       , NULL );

    result = this->collectData();

    this->deallocate();

    // Return the pointer to the voxel data.
    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Entry point to the voxelization. Every voxelizing function calls this 
/// function at some point after setting the relevant options to its own mode
/// of voxelization. This function essentially sets up all the relevant 
/// datastructures so that each device can run independently.
/// 
/// The tasks the thread does before calling \p voxelizeWorker() or \p 
/// fccWorker():
///   - Initialize devices to be used.
///   - Calculate dimensions of the voxelization.
///   - Determine the need for splits, and subdivide the voxelization space.
///   - Tell each device which subdivision they're responsible for.
///
/// Once these tasks have been finished, a thread is launched for each 
/// additional device used (over the first one) and each thread executes the 
/// worker function which calculates the voxelization on each device. If there 
/// is only one device, then only the main thread will call the worker 
/// function. After this function has returned, the voxelization is complete 
/// and can be retrieved from the various data structures.
///
/// \todo Voxelizing to file.
///
/// \throws Exception if CUDA fails for whatever reason.
///
/// \param[in] deviceConfig The number of times the voxelization space needs to 
///                         be subdivided in both the y- and z-directions.
/// \param[in] voxSplitRes Maximum internal voxelization dimensions.
/// \param[in] matSplitRes Maximum internal voxelization dimensions for the more
///                        demanding material calculations.
/// \param[in] filename Not in use. Aborts if not set to NULL.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::voxelizeEntry(
    uint2		 deviceConfig, 
    uint3        voxSplitRes,
    uint3        matSplitRes,
    char const * filename )
{
    //boost::thread_group threads;

    if ( filename != NULL )
        throw Exception("Voxelization to file not yet supported.");
    
    uint nrOfUsedDevices = deviceConfig.x * deviceConfig.y;

    // Always initialize devices, UNLESS slicing is enabled and a previous 
    // slice has already been voxelized.
    if ( !( this->options.slicePrepared && this->options.slices ) )
        this->initDevices( nrOfUsedDevices );

    // If we're voxelizing slices along the x-axis, then the vertices need to 
    // be rotated on the first execution. Subsequent executions should NOT
    // cause any changes to the vertices.
    if ( this->options.slices &&
         !this->options.slicePrepared && 
         this->options.sliceDirection == 0 )
    {
        // We need the min- and maxVertices in order to be able to perform 
        // the rotation.
        this->calculateBoundingBox();

        this->rotateVertices();

        this->calculateBoundingBox();

        if ( this->options.verbose )
            std::cout << "Rotated the vertices.\n";
    }

    // Determine the distance between voxel centers through the number of voxel 
    // centers along the longest side of the bounding box, unless it has been 
    // specified from the start. Then, determine the number of voxels along 
    // each axis.
    if ( this->options.voxelDistanceGiven )
        this->determineDimensions( this->hostVars.voxelLength );
    else
        this->determineDimensions();

    // Split along the x-direction to find out how many splits are needed.
    uint xSplits = splitResolutionByMaxDim( voxSplitRes.x  
                                          , this->hostVars.resolution ).counts;

    if ( this->options.verbose )
        std::cout << "Split the x-resolution into " << xSplits << " parts.\n";

    // Adjust the resolution so that the x-direction is divisible by 
    // 32 times the number of splits along it.
    this->adjustResolution( xSplits );

    // Temp variable so we don't change hostVars.resolution.
    Bounds<uint3> splittable = this->hostVars.resolution;

    // When slicing, set the minimum corner of the bounding box to slice and 
    // the maximum corner to slice + 1, effectively creating a one voxel thick
    // slice along the chosen direction. If the given slice is zero or out of 
    // bounds, set the appropriate flags so that a zero array can be 
    // returned.
    if ( this->options.slices ) {
        if ( this->options.sliceDirection == 0 ||
             this->options.sliceDirection == 1 ) 
        { // X or Y
            if ( this->options.slice == 0 || 
                 this->options.slice > this->hostVars.resolution.max.y )
            {   // Voxelize as if slice was 0 and set the OOB flag.
                this->options.slice = 0;
                this->options.sliceOOB = true;
            }
            else
                this->options.slice -= 1; // Account for the padding.

            // Define a slice.
            splittable.min.y = this->options.slice;
            splittable.max.y = this->options.slice + 1;
        }
        else 
        { // Z
            if ( this->options.slice == 0 || 
                 this->options.slice > this->hostVars.resolution.max.z )
            {   // Voxelize as if slice was 0 and set the OOB flag.
                this->options.slice = 0;
                this->options.sliceOOB = true;
            }
            else
                this->options.slice -= 1; // Account for the padding.

            // Define a slice.
            splittable.min.z = this->options.slice;
            splittable.max.z = this->options.slice + 1;
        }
    }

    // Subdivide the voxelization space according to the device configuration.
    {
        boost::shared_array<Bounds<uint3> > parts = 
            splitResolutionByNrOfParts( make_uint3( 1 
                                                  , deviceConfig.x
                                                  , deviceConfig.y )
                                      , splittable );

        // Assign the different volumes to different devices.
        for ( int i = 0; i < this->nrOfDevicesInUse; ++i )
        {
            // The location of the subspace as a coordinate in an "array" of
            // subspaces. Relies on them being constructed in a certain order. 
            int y = i % deviceConfig.x;
            int z = i / deviceConfig.x;

            this->devices[i].data.location = make_uint3( 0, y, z );

            this->devices[i].data.resolution = parts[i];

            // Also update the minimum corners to reflect the new bounding box.
            this->devices[i].data.minVertex.x = this->hostVars.minVertex.x + 
                double(this->devices[i].data.resolution.min.x) * 
                this->hostVars.voxelLength;
            this->devices[i].data.minVertex.y = this->hostVars.minVertex.y + 
                double(this->devices[i].data.resolution.min.y) * 
                this->hostVars.voxelLength;
            this->devices[i].data.minVertex.z = this->hostVars.minVertex.z + 
                double(this->devices[i].data.resolution.min.z) * 
                this->hostVars.voxelLength;
        }
    }

    // Need to pass the size of the maximum allowable x-length later.
    uint xRes = this->hostVars.resolution.max.x / xSplits;

    this->startTime = clock();

    // The FCC voxelization uses a different worker function.
    //void (Voxelizer<Node, SNode>::*workerFunc)( uint, uint, uint3, uint3
    //                                   , DevContext<Node,SNode> & );

    boost::function<void ( Voxelizer<Node, SNode> * voxelizer
                         , uint xRes
                         , uint xSplits
                         , uint3 voxSplitRes
                         , uint3 matSplitRes
                         , DevContext<Node,SNode> & device )> workerFunc;


    if ( Node::isFCCNode() )
        workerFunc = &Voxelizer<Node, SNode>::fccWorker;
    else if ( Node::usesTwoArrays() )
        workerFunc = &Voxelizer<Node, SNode>::twoNodeArraysWorker;
    else
        workerFunc = &Voxelizer<Node, SNode>::voxelizeWorker;

    //std::cout<< "DEBUG: voxelizeEntry: Before worker threads are launched: option.simulateMultidevice " << this->options.simulateMultidevice << std::endl;
    if ( this->options.simulateMultidevice )
    {
        //std::cout<< "DEBUG: NO Multidevice - nrOfDevicesInUse = " << this->nrOfDevicesInUse << std::endl;
        for ( int i = 0; i < this->nrOfDevicesInUse; ++i )
        {
            //std::cout<< "\tDEBUG: launching thread __ " << i << std::endl;
            workerFunc( this
                      , xRes
                      , xSplits
                      , voxSplitRes
                      , matSplitRes
                      , this->devices[i] );

            //(this->*workerFunc)( xRes
            //                 , xSplits
            //                 , voxSplitRes
            //                 , matSplitRes
            //                 , **it );
        }
    }
    else
    {
        //std::cout<< "\tDEBUG: launching main thread 0 ..." << std::endl;
        // Process device 0 in the main thread.
        workerFunc( this
                  , xRes
                  , xSplits
                  , voxSplitRes
                  , matSplitRes
                  , this->devices[0] );

        if (this->nrOfDevicesInUse > 1){
            boost::thread_group threads;
            // If there are more than one device, launch new threads to process the 
            // additional devices.
            std::cout<< "DEBUG: Multidevice - nrOfDevicesInUse = " << this->nrOfDevicesInUse << std::endl;
            for ( int i = 1; i < this->nrOfDevicesInUse; ++i )
            {
                //std::cout<< "\tDEBUG: launching thread " << i << std::endl;
                threads.create_thread( boost::bind( workerFunc 
                                                  , this
                                                  , xRes
                                                  , xSplits
                                                  , voxSplitRes
                                                  , matSplitRes
                                                  , boost::ref( this->devices[i] ) 
                                                  ) 
                                     );
            }
            // Wait for all threads to finish.
            std::cout<< "DEBUG: waiting for threads to join ..." << std::endl;
            threads.join_all(); // THIS IS WHERE IT CRASHES! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!           !!
            //std::cout<< "DEBUG: threads joined ..." << std::endl;
        }
    }
    //std::cout<< "DEBUG END: voxelizeEntry: All good... " << std::endl;
}
///////////////////////////////////////////////////////////////////////////////
/// A worker function that orchestrates the actual voxelization. This function 
/// can be called by multiple threads simultaneously, as long as the devices
/// the threads use are different.
///
/// \throws Exception if CUDA fails for whatever reason.
///
/// \param[in] xRes The maximum allowable size along the x-axis for one 
///                 voxelization round.
/// \param[in] xSplits The number of splits along the x-axis.
/// \param[in] voxSplitRes The maximum allowable sizes along each direction for 
///                        one voxelization round.
/// \param[in] matSplitRes The maximum allowable sizes along each direction for 
///                        one voxelization round.
/// \param[in] device The device the voxelization should be performed on.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::voxelizeWorker( 
    uint  xRes,
    uint  xSplits,
    uint3 voxSplitRes,
    uint3 matSplitRes,
    DevContext<Node,SNode> & device )
{
    // All processing in this thread will use this device.
    cudaSetDevice( device.data.dev );

    // The left, right, up and down bools tell if, in a multidevice
    // environment, some other device is voxelizing a subspace that is 
    // adjacent to this device's subspace in the given direction. Since each 
    // device voxelizes the entire x-direction, each device can only have at 
    // most 4 neighboring subspaces.
    device.data.left = false;
    device.data.right = false;
    device.data.up = false;
    device.data.down = false;

    // Rename to a shorter name.
    Bounds<uint3> partition = device.data.resolution;
    //uint3 res = partition.max - partition.min; // Dimensions of the box.

    // Determine subspace neighborhoods.
    if ( partition.min.y == this->hostVars.resolution.min.y && 
         partition.max.y != this->hostVars.resolution.max.y )
        device.data.right = true;
    else if ( partition.max.y == this->hostVars.resolution.max.y && 
              partition.min.y != this->hostVars.resolution.min.y )
        device.data.left = true;
    else if ( partition.min.y != this->hostVars.resolution.min.y && 
              partition.max.y != this->hostVars.resolution.max.y ) 
    {
        device.data.left = true;
        device.data.right = true;
    }

    if ( partition.min.z == this->hostVars.resolution.min.z && 
         partition.max.z != this->hostVars.resolution.max.z )
        device.data.up = true;
    else if ( partition.max.z == this->hostVars.resolution.max.z && 
              partition.min.z != this->hostVars.resolution.min.z )
        device.data.down = true;
    else if ( partition.min.z != this->hostVars.resolution.min.z && 
              partition.max.z != this->hostVars.resolution.max.z ) 
    {
        device.data.down = true;
        device.data.up = true;
    }

    // Insert padding to get the correct allocation size.
    device.data.allocResolution = device.data.resolution;

    // Padding or extension.
	device.data.allocResolution.max.y += 2;
	device.data.allocResolution.max.z += 2;

    // Allocate memory, except when slices are enabled and the allocation
    // has already been performed.
    if ( !this->options.slicePrepared ) {
        this->allocStaticMem( device );
    }
    else {
        // When using slicing and the first slice has been calculated, some 
        // data needs to be reset in order for everything to work currectly.
        uint3 dim = device.data.allocResolution.max - 
                    device.data.allocResolution.min;
        if ( this->options.materials )
            device.triangleTypes_gpu.zero();

        device.voxels_gpu.zero();

        // Since the ownership of results is given to the user, we need to 
        // reallocate these.
        device.data.nrOfNodes = dim.x * dim.y * dim.z;
        if ( this->options.sliceDirection == 0 ) {
            device.nodesCopy_gpu.reset( device.data.nrOfNodes, device.data.dev );
        }
        else
        {
            device.nodes_gpu.reset( device.data.nrOfNodes, device.data.dev );
        }
    }

    device.data.maxWorkQueueSize = 0;

    // The extended partition will be a space much like the original voxel 
    // space, but extended in the directions it has neighboring subspaces. 
    // The idea is to have some overlap between subspaces so that the node
    // validation phase can correctly deduce the node neighborhoods at the 
    // edges of the subspace. Extending the space essentially makes the 
    // voxelizer voxelize a slice further.
    Bounds<uint3> extPartition = {
        make_uint3( partition.min.x, partition.min.y, partition.min.z ),
        make_uint3( partition.max.x, partition.max.y, partition.max.z )
    };

    extPartition.min.y -= device.data.left ? 1 : 0;
    extPartition.min.z -= device.data.down ? 1 : 0;
    extPartition.max.y += device.data.right ? 1 : 0;
    extPartition.max.z += device.data.up ? 1 : 0;

    // Update minimum bounds.
    device.data.extMinVertex = make_double3(
        this->hostVars.minVertex.x + double(extPartition.min.x) * 
            this->hostVars.voxelLength,
        this->hostVars.minVertex.y + double(extPartition.min.y) * 
            this->hostVars.voxelLength,
        this->hostVars.minVertex.z + double(extPartition.min.z) * 
            this->hostVars.voxelLength );

    //uint3 extRes = extPartition.max - extPartition.min;

    // Split the subspace into further subspaces as demanded by the maximum 
    // voxelization size. These subsubspaces will be voxelized in sequence.
    SplitData<uint2> extYzSplits = 
        splitResolutionByMaxDim( make_uint2( voxSplitRes.y
                                           , voxSplitRes.z ) 
                               , extPartition );

    device.data.extResolution = extPartition;

    if (this->options.verbose) {
        std::cout << "Left : " << ( device.data.left ? "Yes" : "No" ) << "\n";
        std::cout << "Right: " << ( device.data.right ? "Yes" : "No" ) << "\n";
        std::cout << "Down : " << ( device.data.down ? "Yes" : "No" ) << "\n";
        std::cout << "Up   : " << ( device.data.up ? "Yes" : "No" ) << "\n";

        std::cout << "Res, min: (" << partition.min.x << ", " << 
            partition.min.y << ", " << partition.min.z << ")\n";
        std::cout << "Res, max: (" << partition.max.x << ", " << 
            partition.max.y << ", " << partition.max.z << ")\n";

        std::cout << "Ext res, min: (" << extPartition.min.x << ", " << 
            extPartition.min.y << ", " << extPartition.min.z << ")\n";
        std::cout << "Ext res, max: (" << extPartition.max.x << ", " << 
            extPartition.max.y << ", " << extPartition.max.z << ")\n";

        std::cout << "MinVertex: (" << device.data.extMinVertex.x << ", " << 
            device.data.extMinVertex.y << ", " << device.data.extMinVertex.z << ")\n";
    }

    // Perform voxelization.
    for ( uint i = 0; i < extYzSplits.counts.x * extYzSplits.counts.y; ++i )
        this->performVoxelization( extYzSplits.splits[i]
                                 , xRes
                                 , xSplits
                                 , device );

    // Deallocate the plain voxelization exclusive data structures, unless we
    // are voxelizing into slices.
    if ( !this->options.slices )
        this->deallocateVoxelizationData( device );

    // Quit now if only producing a plain voxelization.
    if (!this->options.nodeOutput)
        return;

    // Using the entire array now, so allocPartition is the right bounding 
    // box to use. Split it into smaller spaces according to the maximum size 
    // demands and process the individual spaces sequentially, just like in 
    // the plain voxelization.
    Bounds<uint3> allocPartition = device.data.allocResolution;
    uint3 allocRes = allocPartition.max - allocPartition.min;

    SplitData<uint2> allocYzSplits = 
        splitResolutionByMaxDim( make_uint2( voxSplitRes.y 
                                           , voxSplitRes.z ) 
                               , allocPartition );

    //uint nrOfNodes = allocRes.x * allocRes.y * allocRes.z;

    if ( !this->options.slicePrepared )
        device.error_gpu.reset( 1, device.data.dev );

    device.nodes_gpu.zero();

    // Perform a simple translation from the integer representation to a Node 
    // representation. No materials or neighborhoods are calculated yet.
    for ( uint i = 0; i < allocYzSplits.counts.x * allocYzSplits.counts.y; ++i )
    {
        calcNodeList( device.data,
                      device.voxels_gpu.get(),
                      device.nodes_gpu.get(),
                      allocYzSplits.splits[i],
                      this->startTime,
                      this->options.verbose );
    }

    // Calculate materials.
    if (this->options.materials)
    {
        // Classifies triangles according to their bounding boxes. This makes 
        // it possible to group similar triangles together to increase 
        // performance.
        calcTriangleClassification( device.data 
                                  , this->hostVars
                                  , device.vertices_gpu.get()
                                  , device.indices_gpu.get()
                                  , device.triangleTypes_gpu.get()
                                  , device.sortedTriangles_gpu.get()
                                  , this->startTime 
                                  , this->options.verbose );
        // Yet another subdivision. Since the surface voxelizer functions in 
        // all three dimensions, instead of just two like in the plain 
        // voxelizer, there are performance pressures to use smaller spaces
        // when voxelizing. The division is done according to matSplitRes.
        SplitData<uint3> splits = 
            this->splitResolutionByMaxDim( matSplitRes
                                         , extPartition );

        for ( uint i = 0
            ; i < splits.counts.x * splits.counts.y * splits.counts.z
            ; ++i )
        {
            calcOptSurfaceVoxelization<Node, SNode>
                ( device.data
                , this->hostVars
                , device.vertices_gpu.get()
                , device.indices_gpu.get()
                , device.triangleTypes_gpu.get()
                , device.sortedTriangles_gpu.get()
                , device.materials_gpu.get()
                , device.nodes_gpu.get()
                , splits.splits[i]
                , 1
                , false
                , device.surfNodes_gpu.get()
                , this->startTime
                , this->options.verbose );

            // Because pending kernel calls execute so quickly after another, 
            // the device driver times out even though individual calls return 
            // before the timeout window. Due to this, the device needs to be 
            // synchronized every now and again to force CUDA to relinquish 
            // control over the device.
#ifdef _WIN32
            cudaDeviceProp devProps = device.data.devProps;
            if (devProps.kernelExecTimeoutEnabled > 0 && i % 2 == 0)
                cudaDeviceSynchronize();
#endif
        }
    }
    
    // If slicing along the x-direction, undo the rotation of the model.
    if ( this->options.slices && this->options.sliceDirection == 0 ) {
        // Rotates the nodes in nodes_gpu (from the rotated state) and 
        // copies the rotated nodes to nodesCopy_gpu (in a "normal" state). 
        // The two arrays are identical in size, but the node arrangement is 
        // different.
        for ( uint i = 0
            ; i < allocYzSplits.counts.x * allocYzSplits.counts.y
            ; ++i )
            restoreRotatedNodes( device.data
                               , device.nodes_gpu.get()
                               , device.nodesCopy_gpu.get()
                               , allocYzSplits.splits[i]
                               , this->startTime
                               , this->options.verbose );

        // In addition to the nodes, the bounding box also needs to be 
        // recalculated. And, since the bounding box changes, the internal 
        // subdivisions also need to be recalculated.

        // First rotate the min and max corners.
        Bounds<uint3> temp = {
            this->unRotateCoords( allocPartition.min, allocRes.x ),
            this->unRotateCoords( allocPartition.max - 1, allocRes.x )
        };

        // Then get the new min and max corners.
        device.data.allocResolution.min = min( temp.min, temp.max );
        device.data.allocResolution.max = max( temp.min, temp.max ) + 1;

        allocPartition = device.data.allocResolution;
        allocRes = allocPartition.max - allocPartition.min;

        allocYzSplits = splitResolutionByMaxDim( make_uint2( voxSplitRes.y 
                                                           , voxSplitRes.z ) 
                                               , allocPartition );
    }
    
    // Calculate orientations. The algorithm should repeatedly call 
    // procNodeList while the error_gpu is true.
	if (this->options.orientations)
	{
		for ( uint i = 0
			; i < allocYzSplits.counts.x * allocYzSplits.counts.y
			; ++i )
		{
			do
			{
				device.error = false;

				device.error_gpu.copyFrom( &device.error );

				procNodeList( device.data
							, device.nodes_gpu.get()
							, device.nodesCopy_gpu.get()
							, device.error_gpu.get()
							, allocYzSplits.splits[i]
							, this->options.slices && 
							  this->options.sliceDirection == 0
							, device.surfNodes_gpu.get()
							, this->startTime 
							, this->options.verbose );

				device.error_gpu.copyTo( &device.error );

			} while ( device.error == true );
		}
	}
	
    // Zero the padding. Since part of the border may contain nodes that 
    // overlap with another subspace, the entire border needs to be set to 
    // zero, unless there are no neighboring subspaces.
    if ( device.data.left || device.data.right || 
         device.data.up || device.data.down )
    {
        makePaddingZero( device.data
                       , device.nodes_gpu.get()
                       , device.nodesCopy_gpu.get()
                       , this->options.slices && 
                         this->options.sliceDirection == 0
                       , this->startTime
                       , this->options.verbose );
    }
    // If the slice requested is zero or outside of the bounds of the 
    // voxelization, just return an empty array. This also emulates the zero 
    // padding at the first and last slices.
    if ( this->options.sliceOOB )
    {
        if ( this->options.sliceDirection == 0 )
        {
            device.nodesCopy_gpu.zero();
        }
        else
        {
            device.nodes_gpu.zero();
        }
    }

    if (this->options.verbose)
    {
        cudaDeviceSynchronize();
        std::cout << "Voxelization finished in " << 
                ( (double)( clock() - this->startTime ) / CLOCKS_PER_SEC ) << 
                " seconds\n\n";
    }
}
///////////////////////////////////////////////////////////////////////////////
/// A worker function that orchestrates the actual voxelization. This function 
/// can be called by multiple threads simultaneously, as long as the devices
/// the threads use are different. This is a specialized version for FCC nodes.
///
/// \throws Exception if CUDA fails for whatever reason.
/// 
/// \param[in] xRes The maximum allowable size along the x-axis for one 
///                 voxelization round.
/// \param[in] xSplits The number of splits along the x-axis.
/// \param[in] voxSplitRes The maximum allowable sizes along each direction for 
///                        one voxelization round.
/// \param[in] matSplitRes The maximum allowable sizes along each direction for 
///                        one voxelization round.
/// \param[in] device The device the voxelization should be performed on.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::fccWorker( 
    uint  xRes,
    uint  xSplits,
    uint3 voxSplitRes,
    uint3 matSplitRes,
    DevContext<Node,SNode> & device )
{
    // All processing in this thread will use this device.
    cudaSetDevice( device.data.dev );

    // The left, right, up and down bools tell if, in a multidevice
    // environment, some other device is voxelizing a subspace that is 
    // adjacent to this device's subspace in the given direction. Since each 
    // device voxelizes the entire x-direction, each device can only have at 
    // most 4 neighboring subspaces.
    device.data.left = false;
    device.data.right = false;
    device.data.up = false;
    device.data.down = false;

    // Rename to a shorter name.
    Bounds<uint3> partition = device.data.resolution;
    //uint3 res = partition.max - partition.min; // Dimensions of the box.

    // Determine subspace neighborhoods.
    if ( partition.min.y == this->hostVars.resolution.min.y && 
         partition.max.y != this->hostVars.resolution.max.y )
        device.data.right = true;
    else if ( partition.max.y == this->hostVars.resolution.max.y && 
              partition.min.y != this->hostVars.resolution.min.y )
        device.data.left = true;
    else if ( partition.min.y != this->hostVars.resolution.min.y && 
              partition.max.y != this->hostVars.resolution.max.y ) 
    {
        device.data.left = true;
        device.data.right = true;
    }

    if ( partition.min.z == this->hostVars.resolution.min.z && 
         partition.max.z != this->hostVars.resolution.max.z )
        device.data.up = true;
    else if ( partition.max.z == this->hostVars.resolution.max.z && 
              partition.min.z != this->hostVars.resolution.min.z )
        device.data.down = true;
    else if ( partition.min.z != this->hostVars.resolution.min.z && 
              partition.max.z != this->hostVars.resolution.max.z ) 
    {
        device.data.down = true;
        device.data.up = true;
    }

    // Insert padding to get the correct allocation size.
    device.data.allocResolution = device.data.resolution;

    // Padding or extension.
    device.data.allocResolution.max.y += 2;
    device.data.allocResolution.max.z += 2;

    // Allocate memory, except when slices are enabled and the allocation
    // has already been performed.
    if ( !this->options.slicePrepared ) {
        this->allocStaticMem(device);
    }
    else {
        // When using slicing and the first slice has been calculated, some 
        // data needs to be reset in order for everything to work correctly.
        uint3 dim = 
            device.data.allocResolution.max - device.data.allocResolution.min;
        if ( this->options.materials )
            device.triangleTypes_gpu.zero();

        device.voxels_gpu.zero();

        // Since the ownership of results is given to the user, we need to 
        // reallocate these.
        device.data.nrOfNodes = 4 * dim.x * dim.y * dim.z;

        if ( this->options.sliceDirection == 0 ) {
            device.nodesCopy_gpu.reset( device.data.nrOfNodes, device.data.dev );
        }
        else
        {
            device.nodes_gpu.reset( device.data.nrOfNodes, device.data.dev );
        }
    }

    device.data.maxWorkQueueSize = 0;

    // The extended partition will be a space much like the original voxel 
    // space, but extended in the directions it has neighboring subspaces. 
    // The idea is to have some overlap between subspaces so that the node
    // validation phase can correctly deduce the node neighborhoods at the 
    // edges of the subspace. Extending the space essentially makes the 
    // voxelizer voxelize a slice further.
    Bounds<uint3> extPartition = {
        make_uint3( partition.min.x, partition.min.y, partition.min.z ),
        make_uint3( partition.max.x, partition.max.y, partition.max.z )
    };

    extPartition.min.y -= device.data.left ? 1 : 0;
    extPartition.min.z -= device.data.down ? 1 : 0;
    extPartition.max.y += device.data.right ? 1 : 0;
    extPartition.max.z += device.data.up ? 1 : 0;

    // Update minimum bounds.
    device.data.extMinVertex = make_double3(
        this->hostVars.minVertex.x + double(extPartition.min.x) * 
            this->hostVars.voxelLength,
        this->hostVars.minVertex.y + double(extPartition.min.y) * 
            this->hostVars.voxelLength,
        this->hostVars.minVertex.z + double(extPartition.min.z) * 
            this->hostVars.voxelLength );

    //uint3 extRes = extPartition.max - extPartition.min;

    // Split the subspace into further subspaces as demanded by the maximum 
    // voxelization size. These subsubspaces will be voxelized in sequence.
    SplitData<uint2> extYzSplits = 
        splitResolutionByMaxDim( make_uint2( voxSplitRes.y
                                           , voxSplitRes.z ) 
                               , extPartition );

    device.data.extResolution = extPartition;

    if (this->options.verbose) {
        std::cout << "Left : " << ( device.data.left ? "Yes" : "No" ) << "\n";
        std::cout << "Right: " << ( device.data.right ? "Yes" : "No" ) << "\n";
        std::cout << "Down : " << ( device.data.down ? "Yes" : "No" ) << "\n";
        std::cout << "Up   : " << ( device.data.up ? "Yes" : "No" ) << "\n";

        std::cout << "Res, min: (" << partition.min.x << ", " << 
            partition.min.y << ", " << partition.min.z << ")\n";
        std::cout << "Res, max: (" << partition.max.x << ", " << 
            partition.max.y << ", " << partition.max.z << ")\n";

        std::cout << "Ext res, min: (" << extPartition.min.x << ", " << 
            extPartition.min.y << ", " << extPartition.min.z << ")\n";
        std::cout << "Ext res, max: (" << extPartition.max.x << ", " << 
            extPartition.max.y << ", " << extPartition.max.z << ")\n";

        std::cout << "MinVertex: (" << device.data.extMinVertex.x << ", " << 
            device.data.extMinVertex.y << ", " << device.data.extMinVertex.z 
            << ")\n";
    }

    // Using the entire array now, so allocPartition is the right bounding 
    // box to use. Split it into smaller spaces according to the maximum size 
    // demands and process the individual spaces sequentially, just like in 
    // the plain voxelization.
    Bounds<uint3> allocPartition = device.data.allocResolution;
    uint3 allocRes = allocPartition.max - allocPartition.min;

    SplitData<uint2> allocYzSplits = 
        splitResolutionByMaxDim( make_uint2( voxSplitRes.y 
                                           , voxSplitRes.z ) 
                               , allocPartition );

    //uint nrOfNodes = 4 * allocRes.x * allocRes.y * allocRes.z;

    
    //if ( !this->options.slicePrepared ) {
    //    this->mmgr->allocHost( &device.error, 1, "Error bool" );
    //    this->mmgr->allocDevice( &device.error_gpu
    //                           , 1
    //                           , device
    //                           , "Error bool" );
    //}

    device.nodes_gpu.zero();

    const bool xSlicing = this->options.slices && 
                          this->options.sliceDirection == 0;

    // Voxelize the four different grids and process them into the FCC grid.
    double3 boundingBoxMin = device.data.extMinVertex;
    double halfWidth = this->hostVars.voxelLength / 2.0;
    for ( int gridType = 1; gridType < 5; ++gridType )
    {
        // Shift the voxelization if gridType != 1. If slicing along x, 
        // the alternating (by z) pattern in which the nodes are arranged is 
        // reversed due to the rotation.
        if ( gridType == 1 )
            device.data.extMinVertex = boundingBoxMin +
                make_double3( 0.0
                            , 0.0
                            , xSlicing ? halfWidth : 0.0 );
        else if ( gridType == 2 )
            device.data.extMinVertex = boundingBoxMin + 
                make_double3( halfWidth
                            , halfWidth
                            , xSlicing ? halfWidth : 0.0 );
        else if ( gridType == 3 )
            device.data.extMinVertex = boundingBoxMin + 
                make_double3( 0.0
                            , halfWidth
                            , xSlicing ? 0.0 : halfWidth );
        else // if ( gridType == 4 )
            device.data.extMinVertex = boundingBoxMin + 
                make_double3( halfWidth
                            , 0.0
                            , xSlicing ? 0.0 : halfWidth );

        for ( uint i = 0; i < extYzSplits.counts.x * extYzSplits.counts.y; ++i )
            this->performVoxelization( extYzSplits.splits[i]
                                     , xRes
                                     , xSplits
                                     , device );

        for ( uint i = 0
            ; i < allocYzSplits.counts.x * allocYzSplits.counts.y
            ; ++i )
            launchConvertToFCCGrid( device.data
                                  , device.voxels_gpu.get()
                                  , device.nodes_gpu.get()
                                  , allocYzSplits.splits[i]
                                  , xSlicing ? (gridType + 1) % 4 + 1 : 
                                               gridType
                                  , this->startTime
                                  , this->options.verbose );

        this->resetDataStructures( device );
    }
    device.data.extMinVertex = boundingBoxMin;

    // Deallocate the plain voxelization exclusive data structures, unless we
    // are voxelizing into slices.
    
    if ( !this->options.slices )
        this->deallocateVoxelizationData( device );

    // Change the dimensions and partitions to match the widened dimensions 
    // of the Node grid.
    device.data.allocResolution.min.x *= 2;
    device.data.allocResolution.max.x *= 2;
    device.data.allocResolution.min.z *= 2;
    device.data.allocResolution.max.z *= 2;

    allocPartition = device.data.allocResolution;
    allocRes = device.data.allocResolution.max -
        device.data.allocResolution.min;

    for ( uint i = 0
        ; i < allocYzSplits.counts.x * allocYzSplits.counts.y
        ; ++i )
    {
        allocYzSplits.splits[i].min.y *= 2;
        //allocYzSplits.second[i].max.y -= 1;
        allocYzSplits.splits[i].max.y *= 2;
        //allocYzSplits.second[i].max.y += 1;
    }

    // Calculate materials.
    if (this->options.materials)
    {
        // Yet another subdivision. Since the surface voxelizer functions 
        // in all three dimensions, instead of just two like in the plain 
        // voxelizer, there are performance pressures to use smaller spaces
        // when voxelizing. The division is done according to matSplitRes.
        SplitData<uint3> splits = 
            this->splitResolutionByMaxDim( matSplitRes, extPartition );

        // Calculate four voxelizations, each shifted a little in relation to 
        // the others. They are also sparser than the final outcome.
        for ( int gridType = 1; gridType < 5; ++gridType )
        {
            // Shift the voxelization if gridType != 1. If slicing along x, 
            //  the alternating (by z) pattern in which the nodes are arranged is 
            //  reversed due to the rotation.
            if ( gridType == 1 )
                device.data.extMinVertex = boundingBoxMin +
                    make_double3( 0
                                , 0
                                , xSlicing ? halfWidth : 0.0 );
            else if ( gridType == 2 )
                device.data.extMinVertex = boundingBoxMin + 
                    make_double3( halfWidth
                                , halfWidth
                                , xSlicing ? halfWidth : 0.0 );
            else if ( gridType == 3 )
                device.data.extMinVertex = boundingBoxMin + 
                    make_double3( 0.0
                                , halfWidth
                                , xSlicing ? 0.0 : halfWidth );
            else // if ( gridType == 4 )
                device.data.extMinVertex = boundingBoxMin + 
                    make_double3( halfWidth
                                , 0.0
                                , xSlicing ? 0.0 : halfWidth );

            // Classifies triangles according to their bounding boxes. This 
            // makes it possible to group similar triangles together to 
            // increase performance.
            calcTriangleClassification( device.data
                                      , this->hostVars
                                      , device.vertices_gpu.get()
                                      , device.indices_gpu.get()
                                      , device.triangleTypes_gpu.get()
                                      , device.sortedTriangles_gpu.get()
                                      , this->startTime 
                                      , this->options.verbose );

            for ( uint i = 0
                ; i < splits.counts.x * splits.counts.y * splits.counts.z
                ; ++i )
            {
                calcOptSurfaceVoxelization<Node, SNode>
                    ( device.data
                    , this->hostVars
                    , device.vertices_gpu.get()
                    , device.indices_gpu.get()
                    , device.triangleTypes_gpu.get()
                    , device.sortedTriangles_gpu.get()
                    , device.materials_gpu.get()
                    , device.nodes_gpu.get()
                    , splits.splits[i]
                    , xSlicing ? (gridType + 1) % 4 + 1
                               : gridType
                    , false
                    , device.surfNodes_gpu.get()
                    , this->startTime
                    , this->options.verbose );

                // Because pending kernel calls execute so quickly after 
                // another, the device driver times out even though individual 
                // calls return before the timeout window. Due to this, the 
                // device needs to be synchronized every now and again to force 
                // CUDA to relinquish control over the device. Seems to be a 
                // Windows issue.
#ifdef _WIN32
                cudaDeviceProp devProps = device.data.devProps;
                if (devProps.kernelExecTimeoutEnabled > 0 && i % 2 == 0)
                    cudaDeviceSynchronize();
#endif
            }
        }
    }

    //cudaMemcpy( device.nodesCopy_gpu.get()
    //          , device.nodes_gpu.get()
    //          , device.nodes_gpu.bytes()
    //          , cudaMemcpyDeviceToDevice );
    //
    //
    //return;
    
    // If slicing along the x-direction, undo the rotation of the model.
    if ( this->options.slices && this->options.sliceDirection == 0 ) {
        // Rotates the nodes in nodes_gpu (from the rotated state) and 
        // copies the rotated nodes to nodesCopy_gpu (in a "normal" state). 
        // The two arrays are identical in size, but the node arrangement is 
        // different.
        for ( uint i = 0
            ; i < allocYzSplits.counts.x * allocYzSplits.counts.y
            ; ++i )
            restoreRotatedNodes( device.data
                               , device.nodes_gpu.get()
                               , device.nodesCopy_gpu.get()
                               , allocYzSplits.splits[i]
                               , this->startTime
                               , this->options.verbose );

        // In addition to the nodes, the bounding box also needs to be 
        // recalculated. And, since the bounding box changes, the internal 
        // subdivisions also need to be recalculated.

        // First rotate the min and max corners.
        Bounds<uint3> temp = {
            this->unRotateCoords( allocPartition.min, allocRes.x ),
            this->unRotateCoords( allocPartition.max - 1, allocRes.x )
        };

        // Then get the new min and max corners.
        device.data.allocResolution.min = 
            make_uint3( temp.min.x - 1, temp.max.y, temp.min.z );
        device.data.allocResolution.max = 
            make_uint3( temp.max.x + 1, temp.min.y + 1, temp.max.z + 1 );

        allocPartition = device.data.allocResolution;
        allocRes = allocPartition.max - allocPartition.min;

        allocYzSplits = splitResolutionByMaxDim( make_uint2( voxSplitRes.y 
                                                           , voxSplitRes.z ) 
                                               , allocPartition );
    }

    // Calculate orientations.
    for ( uint i = 0
        ; i < allocYzSplits.counts.x * allocYzSplits.counts.y
        ; ++i )
    {
        launchCalculateFCCBoundaries( device.data
                                    , device.nodes_gpu.get()
                                    , device.nodesCopy_gpu.get()
                                    , allocYzSplits.splits[i]
                                    , this->options.slices && 
                                      this->options.sliceDirection == 0
                                    , this->startTime 
                                    , this->options.verbose );
    }
    
    // Zero the padding. Since part of the border may contain nodes that 
    // overlap with another subspace, the entire border needs to be set to 
    // zero, unless there are no neighboring subspaces.
    
    if ( device.data.left || device.data.right || 
         device.data.up || device.data.down )
    {
        makePaddingZero( device.data
                       , device.nodes_gpu.get()
                       , device.nodesCopy_gpu.get()
                       , this->options.slices && 
                         this->options.sliceDirection == 0
                       , this->startTime
                       , this->options.verbose );
    }
    
    // If the slice requested is zero of outside of the bounds of the 
    // voxelization, just return an empty array. This also emulates the zero 
    // padding at the first and last slices. 
    if ( this->options.sliceOOB )
    {
        if ( this->options.sliceDirection == 0 )
        {
            device.nodesCopy_gpu.zero();
        }
        else
        {
            device.nodes_gpu.zero();
        }
    }
    

    if ( this->options.verbose )
    {
        cudaDeviceSynchronize();
        std::cout << "Voxelization finished in " << 
                ( (double)( clock() - this->startTime ) / CLOCKS_PER_SEC ) << 
                " seconds\n\n";
    }
}
///////////////////////////////////////////////////////////////////////////////
/// A worker function that orchestrates the actual voxelization. This function 
/// can be called by multiple threads simultaneously, as long as the devices
/// the threads use are different. This is a specialized version for \p Nodes 
/// with two arrays: One for the volumetric voxels and one for the surface 
/// voxels.
///
/// \throws Exception if CUDA fails for whatever reason.
///
/// \param[in] xRes The maximum allowable size along the x-axis for one 
///                 voxelization round.
/// \param[in] xSplits The number of splits along the x-axis.
/// \param[in] voxSplitRes The maximum allowable sizes along each direction for 
///                        one voxelization round.
/// \param[in] matSplitRes The maximum allowable sizes along each direction for 
///                        one voxelization round.
/// \param[in] device The device the voxelization should be performed on.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::twoNodeArraysWorker( 
    uint  xRes,
    uint  xSplits,
    uint3 voxSplitRes,
    uint3 matSplitRes,
    DevContext<Node,SNode> & device )
{
    // All processing in this thread will use this device.
    cudaSetDevice( device.data.dev );

    // The left, right, up and down bools tell if, in a multidevice
    // environment, some other device is voxelizing a subspace that is 
    // adjacent to this device's subspace in the given direction. Since each 
    // device voxelizes the entire x-direction, each device can only have at 
    // most 4 neighboring subspaces.
    device.data.left = false;
    device.data.right = false;
    device.data.up = false;
    device.data.down = false;

    // Rename to a shorter name.
    Bounds<uint3> partition = device.data.resolution;
    //uint3 res = partition.max - partition.min; // Dimensions of the box.

    // Determine subspace neighborhoods.
    if ( partition.min.y == this->hostVars.resolution.min.y && 
         partition.max.y != this->hostVars.resolution.max.y )
        device.data.right = true;
    else if ( partition.max.y == this->hostVars.resolution.max.y && 
              partition.min.y != this->hostVars.resolution.min.y )
        device.data.left = true;
    else if ( partition.min.y != this->hostVars.resolution.min.y && 
              partition.max.y != this->hostVars.resolution.max.y ) 
    {
        device.data.left = true;
        device.data.right = true;
    }

    if ( partition.min.z == this->hostVars.resolution.min.z && 
         partition.max.z != this->hostVars.resolution.max.z )
        device.data.up = true;
    else if ( partition.max.z == this->hostVars.resolution.max.z && 
              partition.min.z != this->hostVars.resolution.min.z )
        device.data.down = true;
    else if ( partition.min.z != this->hostVars.resolution.min.z && 
              partition.max.z != this->hostVars.resolution.max.z ) 
    {
        device.data.down = true;
        device.data.up = true;
    }

    // Insert padding to get the correct allocation size.
    device.data.allocResolution = device.data.resolution;

    // Padding or extension.
    device.data.allocResolution.max.y += 2;
    device.data.allocResolution.max.z += 2;

    // Allocate memory, except when slices are enabled and the allocation
    // has already been performed.
    if ( !this->options.slicePrepared ) {
        this->allocStaticMem( device );
    }
    else {
        // When using slicing and the first slice has been calculated, some 
        // data needs to be reset in order for everything to work currectly.
        uint3 dim = device.data.allocResolution.max - 
                    device.data.allocResolution.min;
        if ( this->options.materials )
            device.triangleTypes_gpu.zero();

        device.voxels_gpu.zero();

        // Since the ownership of results is given to the user, we need to 
        // reallocate these.
        device.data.nrOfNodes = dim.x * dim.y * dim.z;
        if ( this->options.sliceDirection == 0 ) {
            device.nodesCopy_gpu.reset( device.data.nrOfNodes, device.data.dev );
        }
        else
        {
            device.nodes_gpu.reset( device.data.nrOfNodes, device.data.dev );
        }
    }

    device.data.maxWorkQueueSize = 0;

    // The extended partition will be a space much like the original voxel 
    // space, but extended in the directions it has neighboring subspaces. 
    // The idea is to have some overlap between subspaces so that the node
    // validation phase can correctly deduce the node neighborhoods at the 
    // edges of the subspace. Extending the space essentially makes the 
    // voxelizer voxelize a slice further.
    Bounds<uint3> extPartition = {
        make_uint3( partition.min.x, partition.min.y, partition.min.z ),
        make_uint3( partition.max.x, partition.max.y, partition.max.z )
    };

    extPartition.min.y -= device.data.left ? 1 : 0;
    extPartition.min.z -= device.data.down ? 1 : 0;
    extPartition.max.y += device.data.right ? 1 : 0;
    extPartition.max.z += device.data.up ? 1 : 0;

    // Update minimum bounds.
    device.data.extMinVertex = make_double3(
        this->hostVars.minVertex.x + double(extPartition.min.x) * 
            this->hostVars.voxelLength,
        this->hostVars.minVertex.y + double(extPartition.min.y) * 
            this->hostVars.voxelLength,
        this->hostVars.minVertex.z + double(extPartition.min.z) * 
            this->hostVars.voxelLength );

    //uint3 extRes = extPartition.max - extPartition.min;

    // Split the subspace into further subspaces as demanded by the maximum 
    // voxelization size. These subsubspaces will be voxelized in sequence.
    SplitData<uint2> extYzSplits = 
        splitResolutionByMaxDim( make_uint2( voxSplitRes.y
                                           , voxSplitRes.z ) 
                               , extPartition );

    device.data.extResolution = extPartition;

    if (this->options.verbose) {
        std::cout << "Left : " << ( device.data.left ? "Yes" : "No" ) << "\n";
        std::cout << "Right: " << ( device.data.right ? "Yes" : "No" ) << "\n";
        std::cout << "Down : " << ( device.data.down ? "Yes" : "No" ) << "\n";
        std::cout << "Up   : " << ( device.data.up ? "Yes" : "No" ) << "\n";

        std::cout << "Res, min: (" << partition.min.x << ", " << 
            partition.min.y << ", " << partition.min.z << ")\n";
        std::cout << "Res, max: (" << partition.max.x << ", " << 
            partition.max.y << ", " << partition.max.z << ")\n";

        std::cout << "Ext res, min: (" << extPartition.min.x << ", " << 
            extPartition.min.y << ", " << extPartition.min.z << ")\n";
        std::cout << "Ext res, max: (" << extPartition.max.x << ", " << 
            extPartition.max.y << ", " << extPartition.max.z << ")\n";

        std::cout << "MinVertex: (" << device.data.extMinVertex.x << ", " << 
            device.data.extMinVertex.y << ", " << device.data.extMinVertex.z << ")\n";
    }

    // Perform voxelization.
    for ( uint i = 0; i < extYzSplits.counts.x * extYzSplits.counts.y; ++i )
        this->performVoxelization( extYzSplits.splits[i]
                                 , xRes
                                 , xSplits
                                 , device );

    // Deallocate the plain voxelization exclusive data structures, unless we
    // are voxelizing into slices.
    if ( !this->options.slices )
        this->deallocateVoxelizationData( device );

    // Using the entire array now, so allocPartition is the right bounding 
    // box to use. Split it into smaller spaces according to the maximum size 
    // demands and process the individual spaces sequentially, just like in 
    // the plain voxelization.
    Bounds<uint3> allocPartition = device.data.allocResolution;
    uint3 allocRes = allocPartition.max - allocPartition.min;

    SplitData<uint2> allocYzSplits = 
        splitResolutionByMaxDim( make_uint2( voxSplitRes.y 
                                           , voxSplitRes.z ) 
                               , allocPartition );

    //uint nrOfNodes = allocRes.x * allocRes.y * allocRes.z;

    if ( !this->options.slicePrepared )
        device.error_gpu.reset( 1, device.data.dev );

    device.nodes_gpu.zero();

    // Perform a specialized surface voxelization that sets each node, that is 
    // part of the surface voxelization, to one. This is done to identify the 
    // surface nodes.
    {
        // Classifies triangles according to their bounding boxes. This makes 
        // it possible to group similar triangles together to increase 
        // performance. This needs to be done only once.
        calcTriangleClassification( device.data 
                                  , this->hostVars
                                  , device.vertices_gpu.get()
                                  , device.indices_gpu.get()
                                  , device.triangleTypes_gpu.get()
                                  , device.sortedTriangles_gpu.get()
                                  , this->startTime 
                                  , this->options.verbose );
        // Yet another subdivision. Since the surface voxelizer functions in 
        // all three dimensions, instead of just two like in the plain 
        // voxelizer, there are performance pressures to use smaller spaces
        // when voxelizing. The division is done according to matSplitRes.
        SplitData<uint3> splits = 
            this->splitResolutionByMaxDim( matSplitRes
                                         , extPartition );

        for ( uint i = 0
            ; i < splits.counts.x * splits.counts.y * splits.counts.z
            ; ++i )
        {
            calcOptSurfaceVoxelization<Node, SNode>
                ( device.data
                , this->hostVars
                , device.vertices_gpu.get()
                , device.indices_gpu.get()
                , device.triangleTypes_gpu.get()
                , device.sortedTriangles_gpu.get()
                , device.materials_gpu.get()
                , device.nodes_gpu.get()
                , splits.splits[i]
                , 1
                , true // This enables the marking of surface nodes.
                , device.surfNodes_gpu.get()
                , this->startTime
                , this->options.verbose );

            // Because pending kernel calls execute so quickly after another, 
            // the device driver times out even though individual calls return 
            // before the timeout window. Due to this, the device needs to be 
            // synchronized every now and again to force CUDA to relinquish 
            // control over the device.
#ifdef _WIN32
            cudaDeviceProp devProps = device.data.devProps;
            if (devProps.kernelExecTimeoutEnabled > 0 && i % 2 == 0)
                cudaDeviceSynchronize();
#endif
        }
    }

    // Use thrust to count the number of nodes that were earlier marked as 
    // being surface nodes.
    calcSurfNodeCount( device.data
                     , device.nodes_gpu.get()
                     , this->startTime
                     , this->options.verbose );

    if ( this->options.verbose ) 
        std::cout << "Number of surface nodes: " << 
                     device.data.nrOfSurfaceNodes << "\n";

    // Allocate Surface node array and HashMap. Having the HashMap be twice 
    // as large as the data it should contain is a pretty good value that 
    // results in relatively few collisions.
    device.data.hashMap = HashMap( 2 * device.data.nrOfSurfaceNodes );
    device.data.hashMap.allocate();

    device.surfNodes_gpu.reset( device.data.nrOfSurfaceNodes
                              , device.data.dev );
    device.surfNodes_gpu.zero();

    // Read the node array and calculate an index into the surface node array 
    // for each regular node that was earlier marked as being part of the 
    // surface. Essentially fills the HashMap with mappings from regular node 
    // indices to surface node indices.
    populateHashMap( device.data
                   , device.nodes_gpu.get()
                   , this->startTime
                   , this->options.verbose );

    // Perform a simple translation from the integer representation to a Node 
    // representation. No materials or neighborhoods are calculated yet.
    // Essentially complements the earlier surface node data with the data 
    // from the plain voxelization.
    for ( uint i = 0; i < allocYzSplits.counts.x * allocYzSplits.counts.y; ++i )
    {
        calcNodeList( device.data,
                      device.voxels_gpu.get(),
                      device.nodes_gpu.get(),
                      allocYzSplits.splits[i],
                      this->startTime,
                      this->options.verbose );
    }

    // Calculate materials and also perform the cutting algorithm to determine 
    // the inner volume of the voxel, as well as the partial areas of the sides 
    // of the voxel that connect solid voxels together.
    {
        // Yet another subdivision. Since the surface voxelizer functions in 
        // all three dimensions, instead of just two like in the plain 
        // voxelizer, there are performance pressures to use smaller spaces
        // when voxelizing. The division is done according to matSplitRes.
        SplitData<uint3> splits = 
            this->splitResolutionByMaxDim( matSplitRes
                                         , extPartition );

        for ( uint i = 0
            ; i < splits.counts.x * splits.counts.y * splits.counts.z
            ; ++i )
        {
            calcOptSurfaceVoxelization<Node, SNode>
                ( device.data
                , this->hostVars
                , device.vertices_gpu.get()
                , device.indices_gpu.get()
                , device.triangleTypes_gpu.get()
                , device.sortedTriangles_gpu.get()
                , device.materials_gpu.get()
                , device.nodes_gpu.get()
                , splits.splits[i]
                , 1
                , false
                , device.surfNodes_gpu.get()
                , this->startTime
                , this->options.verbose );

            // Because pending kernel calls execute so quickly after another, 
            // the device driver times out even though individual calls return 
            // before the timeout window. Due to this, the device needs to be 
            // synchronized every now and again to force CUDA to relinquish 
            // control over the device.
#ifdef _WIN32
            cudaDeviceProp devProps = device.data.devProps;
            if (devProps.kernelExecTimeoutEnabled > 0 && i % 2 == 0)
                cudaDeviceSynchronize();
#endif
        }
    }
    
    // If slicing along the x-direction, undo the rotation of the model.
    if ( this->options.slices && this->options.sliceDirection == 0 ) {
        // Rotates the nodes in nodes_gpu (from the rotated state) and 
        // copies the rotated nodes to nodesCopy_gpu (in a "normal" state). 
        // The two arrays are identical in size, but the node arrangement is 
        // different.
        for ( uint i = 0
            ; i < allocYzSplits.counts.x * allocYzSplits.counts.y
            ; ++i )
            restoreRotatedNodes( device.data
                               , device.nodes_gpu.get()
                               , device.nodesCopy_gpu.get()
                               , allocYzSplits.splits[i]
                               , this->startTime
                               , this->options.verbose );

        // In addition to the nodes, the bounding box also needs to be 
        // recalculated. And, since the bounding box changes, the internal 
        // subdivisions also need to be recalculated.

        // First rotate the min and max corners.
        Bounds<uint3> temp = {
            this->unRotateCoords( allocPartition.min, allocRes.x ),
            this->unRotateCoords( allocPartition.max - 1, allocRes.x )
        };

        // Then get the new min and max corners.
        device.data.allocResolution.min = min( temp.min, temp.max );
        device.data.allocResolution.max = max( temp.min, temp.max ) + 1;

        allocPartition = device.data.allocResolution;
        allocRes = allocPartition.max - allocPartition.min;

        allocYzSplits = splitResolutionByMaxDim( make_uint2( voxSplitRes.y 
                                                           , voxSplitRes.z ) 
                                               , allocPartition );
    }
    
    // Calculate orientations. The algorithm should repeatedly call 
    // procNodeList while the error_gpu is true. If the algorithm consumes too 
    // many surface nodes, then the nodes with surface bids will start to have 
    // no matching surface nodes. This is identical to the case with other 
    // node types (including this one) where nodes are destroyed in 
    // sufficiently large numbers to move the surface to cover nodes with no 
    // material ids set. There is no built-in mechanism to detect these cases.
    // Usually the surface nodes produce a thick enough layer that the deletion 
    // of a few will not cause any loss of important data.
    for ( uint i = 0
        ; i < allocYzSplits.counts.x * allocYzSplits.counts.y
        ; ++i )
    {
        do
        {
            device.error = false;

            device.error_gpu.copyFrom( &device.error );

            procNodeList( device.data
                        , device.nodes_gpu.get()
                        , device.nodesCopy_gpu.get()
                        , device.error_gpu.get()
                        , allocYzSplits.splits[i]
                        , this->options.slices && 
                          this->options.sliceDirection == 0
                        , device.surfNodes_gpu.get()
                        , this->startTime 
                        , this->options.verbose );

            device.error_gpu.copyTo( &device.error );

        } while ( device.error == true );
    }
    
    // Zero the padding. Since part of the border may contain nodes that 
    // overlap with another subspace, the entire border needs to be set to 
    // zero, unless there are no neighboring subspaces.
    if ( device.data.left || device.data.right || 
         device.data.up || device.data.down )
    {
        makePaddingZero( device.data
                       , device.nodes_gpu.get()
                       , device.nodesCopy_gpu.get()
                       , this->options.slices && 
                         this->options.sliceDirection == 0
                       , this->startTime
                       , this->options.verbose );
    }
    // If the slice requested is zero or outside of the bounds of the 
    // voxelization, just return an empty array. This also emulates the zero 
    // padding at the first and last slices.
    if ( this->options.sliceOOB )
    {
        if ( this->options.sliceDirection == 0 )
        {
            device.nodesCopy_gpu.zero();
        }
        else
        {
            device.nodes_gpu.zero();
        }
    }

    if (this->options.verbose)
    {
        cudaDeviceSynchronize();
        std::cout << "Voxelization finished in " << 
                ( (double)( clock() - this->startTime ) / CLOCKS_PER_SEC ) << 
                " seconds\n\n";
    }
}


///////////////////////////////////////////////////////////////////////////////
/// Produces a \p Node based voxelization given the supplied parameters. The 
/// function returns one \p NodePointer per device, and it contains the device 
/// pointer to the array of \p Nodes as well as the dimensions of the array and 
/// the location of the subspace in relation to the whole array.
///
/// \throws Exception if CUDA encounters any errors.
/// 
/// \param[in] maxDimension The length of the longest side of the model's
///                         bounding box, measured in voxel centers, starting 
///                         from the very edge and ending at the opposite edge.
/// \param[in] devConfig Determines how the y- and z-axes are split among 
///                      different GPUs. A (1,2)-value means that both devices 
///                      will render the entire y-axis, but the z-axis is split 
///                      evenly among the two axes. Multiplying the x and y 
///                      values together gives the amount of spaces the voxel 
///                      array is split into, as well as the required number of 
///                      devices.
/// \param[in] voxSplitRes The internal size of a single voxelization pass. 
///                        Having a smaller size forces the voxelization to be 
///                        run in multiple, consecutive passes. If the size is 
///                        larger than the total voxelization size, then the 
///                        voxelization is performed parallelly in one go. The 
///                        \p voxSplitRes and \p matSplitRes sizes have nothing 
///                        to do with multiple devices, but instead define a 
///                        minimum workload for all devices.
/// \param[in] matSplitRes Same function as \p voxSplitRes, but applies to the 
///                        material calculation phase, which is more demanding 
///                        of the device than the voxelization phase.
/// \return A vector of \p NodePointer, where each \p NodePointer corresponds 
///         to a specific device. All information needed to read the array and 
///         place it in context of the whole can be found in the \p NodePointer
///         struct.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
std::vector<NodePointer<Node> > Voxelizer<Node, SNode>::voxelizeToNodes( 
    uint  maxDimension,
    uint2 devConfig,
    uint3 voxSplitRes,
    uint3 matSplitRes )
{
    std::vector<NodePointer<Node> > result;

    this->options.nodeOutput = true;

    if ( Node::isFCCNode() )
    {
        std::cout << "FCC Nodes can currently only be defined through the "
                     "side length of their cubic cells";
        return result;
    }

    this->setResolution( maxDimension );

    this->voxelizeEntry( devConfig, voxSplitRes, matSplitRes, NULL );

    result = this->collectData();

    this->deallocate();

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Produces a \p Node based voxelization given the supplied parameters. The 
/// function returns one \p NodePointer per device, and it contains the device 
/// pointer to the array of \p Nodes as well as the dimensions of the array and 
/// the location of the subspace in relation to the whole array.
///
/// \throws Exception if CUDA encounters any errors.
/// 
/// \param[in] cubeLength The distance between \p Nodes along any of the three 
///                       main axes.
/// \param[in] devConfig Determines how the y- and z-axes are split among 
///                      different GPUs. A (1,2)-value means that both devices 
///                      will render the entire y-axis, but the z-axis is split 
///                      evenly among the two axes. Multiplying the x and y 
///                      values together gives the amount of spaces the voxel 
///                      array is split into, as well as the required number of 
///                      devices.
/// \param[in] voxSplitRes The internal size of a single voxelization pass. 
///                        Having a smaller size forces the voxelization to be 
///                        run in multiple, consecutive passes. If the size is 
///                        larger than the total voxelization size, then the 
///                        voxelization is performed parallelly in one go. The 
///                        \p voxSplitRes and \p matSplitRes sizes have nothing 
///                        to do with multiple devices, but instead define a 
///                        minimum workload for all devices.
/// \param[in] matSplitRes Same function as \p voxSplitRes, but applies to the 
///                        material calculation phase, which is more demanding 
///                        of the device than the voxelization phase.
/// \return A vector of \p NodePointer, where each \p NodePointer corresponds 
///         to a specific device. All information needed to read the array and 
///         place it in context of the whole can be found in the \p NodePointer
///         struct.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
std::vector<NodePointer<Node> > Voxelizer<Node, SNode>::voxelizeToNodes( 
    double cubeLength,
    uint2 devConfig,
    uint3 voxSplitRes,
    uint3 matSplitRes )
{
    std::vector<NodePointer<Node> > result;

    this->options.nodeOutput = true;
    this->options.voxelDistanceGiven = true;
    this->hostVars.voxelLength = cubeLength;

    this->voxelizeEntry( devConfig, voxSplitRes, matSplitRes, NULL );

    result = this->collectData();

    this->deallocate();

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Identical to \p voxelizeToNodes(), but allocates and copies the \p Nodes to 
/// host memory after voxelization. Also doesn't support multiple devices.
///
/// \throws Exception if CUDA encounters any errors.
///
/// \param[in] maxDimension How many voxel centers there are on the longest side
///                         of the model's bounding box.
/// \param[in] voxSplitRes Internal maximum voxelization size per device.
/// \param[in] matSplitRes Internal maximum voxelization size per device for 
///                        the heavier material calculations.
/// \return A \p NodePointer struct.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
NodePointer<Node> Voxelizer<Node, SNode>::voxelizeToNodesToRAM( 
    uint  maxDimension,
    uint3 voxSplitRes,
    uint3 matSplitRes )
{
    NodePointer<Node> result = NodePointer<Node>();

    if ( Node::isFCCNode() )
    {
        std::cout << "FCC Nodes can currently only be defined through the " << 
                     "length of their cubic cells";
        return result;
    }

    this->options.nodeOutput = true;
    this->setResolution( maxDimension );

    this->voxelizeEntry( make_uint2( 1 ), voxSplitRes, matSplitRes, NULL );

    result = this->collectData( this->devices[0], true );

    this->deallocate();

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Identical to \p voxelizeToNodes(), but allocates and copies the \p Nodes to 
/// host memory after voxelization. Also doesn't support multiple devices.
///
/// \throws Exception if CUDA encounters any errors.
///
/// \param[in] cubeLength Distance between \p Nodes along any of the three 
///                       main axes.
/// \param[in] voxSplitRes Internal maximum voxelization size per device.
/// \param[in] matSplitRes Internal maximum voxelization size per device for 
///                        the heavier material calculations.
/// \return A \p NodePointer struct.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
NodePointer<Node> Voxelizer<Node, SNode>::voxelizeToNodesToRAM( 
    double cubeLength,
    uint3 voxSplitRes,
    uint3 matSplitRes )
{
    NodePointer<Node> result = NodePointer<Node>();

    this->options.nodeOutput = true;
    this->options.voxelDistanceGiven = true;
    this->hostVars.voxelLength = cubeLength;

    this->voxelizeEntry( make_uint2( 1 ), voxSplitRes, matSplitRes, NULL );

    result = this->collectData( this->devices[0], true );

    this->deallocate();

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Partially voxelizes a larger space, as defined by \p maxDimension, and 
/// returns thin slices (3 to 6 \p Nodes thick, depending on the \p Node type)
/// of the total voxelization. The voxelization can be carried out along any 
/// of the three main directions using \p direction, and the n:th slice along 
/// the chosen direction can be voxelizaed by passing n to \p slice.
/// 
/// \warning Voxelizing slices along the x-direction is currently much less 
///          efficient than the other directions due to having to rotate the 
///          vertices before and after the voxelization. The un-rotation also 
///          includes allocation and copying into another slice. It is advised 
///          to manually rotate the model and use one of the y- or z-slicing 
///          directions and then interpret the results accordingly.
///
/// \throws Exception if CUDA encounters any errors along the way.
///
/// \param[in] maxDimension How many voxels there are on the longest side of
///                         the model's bounding box.
/// \param[in] direction Along which axis the two-dimensional slice should 
///                      sweep. 0 stands for x, 1 for y and 2 for z.
///                      Slicing along the x-axis is rather inefficient, so 
///                      consider manually rotating the model and using one of 
///                      the other directions instead.
/// \param[in] slice The coordinate of the current slice. Ranges from 0 to 
///                  however long the bounding box is along the chosen axis.
/// \param[in] devConfig Choose how many devices should be used.
/// \param[in] voxSplitRes Determines on how large chunks the voxelization is 
///                        carried out internally. Smaller chunks cause more
///                        serialization, but may be more stable and less 
///                        likely to choke on its own workload.
/// \param[in] matSplitRes Same as \p voxSplitRes, but for the part of the 
///                        program that calculates the materials. Generally 
///                        heavier than the plain voxelization, so the numbers
///                        should be at most as large.
/// \return A vector of NodePointer that contains the device pointers for 
///         each device, as well as other useful information, such as the 
///         dimensions of the node array. The array has zero padding on 
///         all sides, resulting in a three-voxel-thick slice, where the 
///         middle slice is the actual result.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
std::vector<NodePointer<Node> >  Voxelizer<Node, SNode>::voxelizeSlice
    ( uint  maxDimension
    , int   direction 
    , uint  slice
    , uint  devConfig
    , uint2 voxSplitRes
    , uint2 matSplitRes 
    )
{
    std::vector<NodePointer<Node> > result;

    if ( Node::isFCCNode() )
    {
        std::cout << "FCC Nodes can currently only be defined through the " << 
                     "length of their cubic cells";
        return result;
    }

    this->options.nodeOutput = true;
    this->options.slices = true;
    this->options.sliceOOB = false;
    this->options.slice = slice;

    this->setResolution( maxDimension );

    // If the direction changes, then everything will have to be
    // recalculated, so make slicePrepared = false.
    if (this->options.sliceDirection != direction)
    {
        this->options.slicePrepared = false;
    }

    this->options.sliceDirection = direction;

    // Devconfig.
    uint2 dc = make_uint2( 1 );
    // Voxel split resolution.
    const uint3 vsr = make_uint3( voxSplitRes.x
                                , voxSplitRes.y
                                , voxSplitRes.y );
    // Material split resolution.
    const uint3 msr = make_uint3( matSplitRes.x
                                , matSplitRes.y
                                , matSplitRes.y );

    // Slice sweeps along the x-axis.
    if ( direction == 0 || direction == 1 )
    {   // X-axis or Y-axis.
        dc.x = devConfig;
    }
    else 
    { // Z-axis.
        dc.y = devConfig;
    }

    this->voxelizeEntry( dc, vsr, msr, NULL );

    result = this->collectData();

    // Once the first slice has been calculated, certain work can be skipped
    // in subsequent calls.
    this->options.slicePrepared = true;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Partially voxelizes a larger space, as defined by \p cubeLength, and 
/// returns thin slices (3 to 6 \p Nodes thick, depending on the \p Node type)
/// of the total voxelization. The voxelization can be carried out along any 
/// of the three main directions using \p direction, and the n:th slice along 
/// the chosen direction can be voxelizaed by passing n to \p slice.
/// 
/// \warning Voxelizing slices along the x-direction is currently much less 
///          efficient than the other directions due to having to rotate the 
///          vertices before and after the voxelization. The un-rotation also 
///          includes allocation and copying into another slice. It is advised 
///          to manually rotate the model and use one of the y- or z-slicing 
///          directions and then interpret the results accordingly.
///
/// \throws Exception if CUDA encounters any errors along the way.
///
/// \param[in] cubeLength Distance between \p Nodes along any of the three main 
///                       axes.
/// \param[in] direction Along which axis the two-dimensional slice should 
///                      sweep. 0 stands for x, 1 for y and 2 for z.
///                      Slicing along the x-axis is rather inefficient, so 
///                      consider manually rotating the model and using one of 
///                      the other directions instead.
/// \param[in] slice The coordinate of the current slice. Ranges from 0 to 
///                  however long the bounding box is along the chosen axis.
/// \param[in] devConfig Choose how many devices should be used.
/// \param[in] voxSplitRes Determines on how large chunks the voxelization is 
///                        carried out internally. Smaller chunks cause more
///                        serialization, but may be more stable and less 
///                        likely to choke on its own workload.
/// \param[in] matSplitRes Same as \p voxSplitRes, but for the part of the 
///                        program that calculates the materials. Generally 
///                        heavier than the plain voxelization, so the numbers
///                        should be at most as large.
/// \return A vector of NodePointer that contains the device pointers for 
///         each device, as well as other useful information, such as the 
///         dimensions of the node array. The array has zero padding on 
///         all sides, resulting in a three-voxel-thick slice, where the 
///         middle slice is the actual result.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
std::vector<NodePointer<Node> >  Voxelizer<Node, SNode>::voxelizeSlice
    ( double cubeLength
    , int   direction 
    , uint  slice
    , uint  devConfig
    , uint2 voxSplitRes
    , uint2 matSplitRes 
    )
{
    std::vector<NodePointer<Node> > result;

    this->options.nodeOutput = true;
    this->options.slices = true;
    this->options.sliceOOB = false;
    this->options.slice = slice;

    // If the direction changes, then everything will have to be
    // recalculated, so make slicePrepared = false.
    if ( this->options.sliceDirection != direction || 
         cubeLength != this->hostVars.voxelLength )
    {
        this->options.slicePrepared = false;
    }

    this->options.voxelDistanceGiven = true;
    this->hostVars.voxelLength = cubeLength;

    this->options.sliceDirection = direction;

    // Devconfig.
    uint2 dc = make_uint2( 1 );
    // Voxel split resolution.
    const uint3 vsr = make_uint3( voxSplitRes.x
                                , voxSplitRes.y
                                , voxSplitRes.y );
    // Material split resolution.
    const uint3 msr = make_uint3( matSplitRes.x
                                , matSplitRes.y
                                , matSplitRes.y );

    // Slice sweeps along the x-axis.
    if ( direction == 0 || direction == 1 )
    {   // X-axis or Y-axis.
        dc.x = devConfig;
    }
    else 
    { // Z-axis.
        dc.y = devConfig;
    }

    this->voxelizeEntry( dc, vsr, msr, NULL );

    result = this->collectData();

    // Once the first slice has been calculated, certain work can be skipped
    // in subsequent calls.
    this->options.slicePrepared = true;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Functionally identical to \p voxelizeSlice(), but returns only a single 
/// \p NodePointer and multi-device voxelization is disabled.
///
/// \warning Voxelizing slices along the x-direction is currently much less 
///          efficient than the other directions due to having to rotate the 
///          vertices before and after the voxelization. The un-rotation also 
///          includes allocation and copying into another slice. It is advised 
///          to manually rotate the model and use one of the y- or z-slicing 
///          directions and then interpret the results accordingly.
///
/// \throws Exception if CUDA encounters an error.
///
/// \param[in] maxDimension How many voxels there are on the longest side of
///                         the model's bounding box.
/// \param[in] direction Along which axis the two-dimensional slice should 
///                      sweep. 0 stands for x, 1 for y and 2 for z.
///                      Slicing along the x-axis is rather inefficient, so 
///                      consider manually rotating the model and using one of 
///                      the other directions instead.
/// \param[in] slice The coordinate of the current slice. Ranges from 0 to 
///                  however long the bounding box is along the chosen axis.
/// \param[in] voxSplitRes Determines on how large chunks the voxelization is 
///                        carried out internally. Smaller chunks cause more
///                        serialization, but may be more stable and less 
///                        likely to choke on its own workload.
/// \param[in] matSplitRes Same as \p voxSplitRes, but for the part of the 
///                        program that calculates the materials. Generally 
///                        heavier than the plain voxelization, so the numbers
///                        should be at most as large.
/// \return A \p NodePointer that contains the host pointer for the default 
///         device, as well as other useful information, such as the 
///         dimensions of the node array. The array has zero padding on 
///         all sides, resulting in a three-voxel-thick slice, where the 
///         middle slice is the actual result.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
NodePointer<Node>  Voxelizer<Node, SNode>::voxelizeSliceToRAM( uint  maxDimension
                                                      , int   direction
                                                      , uint  slice
                                                      , uint2 voxSplitRes
                                                      , uint2 matSplitRes )
{
    NodePointer<Node> result = NodePointer<Node>();

    if ( Node::isFCCNode() )
    {
        std::cout << "FCC Nodes can currently only be defined through the "
                     "length of their cubic cells";
        return result;
    }

    this->options.nodeOutput = true;
    this->options.slices = true;
    this->options.sliceOOB = false;
    this->options.slice = slice;

    this->setResolution( maxDimension );

    // If the direction changes, then everything will have to be
    // recalculated, so make slicePrepared = false.
    if ( this->options.sliceDirection != direction )
    {
        this->options.slicePrepared = false;
    }

    this->options.sliceDirection = direction;

    // Devconfig.
    const uint2 dc = make_uint2( 1 );
    // Voxel split resolution.
    const uint3 vsr = make_uint3( voxSplitRes.x
                                , voxSplitRes.y
                                , voxSplitRes.y );
    // Material split resolution.
    const uint3 msr = make_uint3( matSplitRes.x
                                , matSplitRes.y
                                , matSplitRes.y );

    this->voxelizeEntry( dc, vsr, msr, NULL );

    result = this->collectData( this->devices[0], true );

    this->options.slicePrepared = true;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Functionally identical to \p voxelizeSlice(), but returns only a single 
/// \p NodePointer and multi-device voxelization is disabled.
///
/// \warning Voxelizing slices along the x-direction is currently much less 
///          efficient than the other directions due to having to rotate the 
///          vertices before and after the voxelization. The un-rotation also 
///          includes allocation and copying into another slice. It is advised 
///          to manually rotate the model and use one of the y- or z-slicing 
///          directions and then interpret the results accordingly.
///
/// \throws Exception if CUDA encounters an error.
///
/// \param[in] cubeLength Distance between \p Nodes along any of the three main 
///                       axes.
/// \param[in] direction Along which axis the two-dimensional slice should 
///                      sweep. 0 stands for x, 1 for y and 2 for z.
///                      Slicing along the x-axis is rather inefficient, so 
///                      consider manually rotating the model and using one of 
///                      the other directions instead.
/// \param[in] slice The coordinate of the current slice. Ranges from 0 to 
///                  however long the bounding box is along the chosen axis.
/// \param[in] voxSplitRes Determines on how large chunks the voxelization is 
///                        carried out internally. Smaller chunks cause more
///                        serialization, but may be more stable and less 
///                        likely to choke on its own workload.
/// \param[in] matSplitRes Same as \p voxSplitRes, but for the part of the 
///                        program that calculates the materials. Generally 
///                        heavier than the plain voxelization, so the numbers
///                        should be at most as large.
/// \return A \p NodePointer that contains the host pointer for the default 
///         device, as well as other useful information, such as the 
///         dimensions of the node array. The array has zero padding on 
///         all sides, resulting in a three-voxel-thick slice, where the 
///         middle slice is the actual result.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
NodePointer<Node> Voxelizer<Node, SNode>::voxelizeSliceToRAM( double cubeLength
                                                     , int   direction
                                                     , uint  slice
                                                     , uint2 voxSplitRes
                                                     , uint2 matSplitRes )
{
    NodePointer<Node> result = NodePointer<Node>();

    this->options.nodeOutput = true;
    this->options.slices = true;
    this->options.sliceOOB = false;
    this->options.slice = slice;

    // If the direction changes, then everything will have to be
    // recalculated, so make slicePrepared = false.
    if ( this->options.sliceDirection != direction || 
         cubeLength != this->hostVars.voxelLength )
    {
        this->options.slicePrepared = false;
    }

    this->options.sliceDirection = direction;

    this->options.voxelDistanceGiven = true;
    this->hostVars.voxelLength = cubeLength;

    // Devconfig.
    const uint2 dc = make_uint2( 1 );
    // Voxel split resolution.
    const uint3 vsr = make_uint3( voxSplitRes.x
                                , voxSplitRes.y
                                , voxSplitRes.y );
    // Material split resolution.
    const uint3 msr = make_uint3( matSplitRes.x
                                , matSplitRes.y
                                , matSplitRes.y );

    this->voxelizeEntry( dc, vsr, msr, NULL );

    result = this->collectData( this->devices[0], true );

    this->options.slicePrepared = true;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Produces a plain, integer voxelization based on the given arguments.
///
/// \throws Exception if CUDA encounters any errors.
///
/// \param[in] maxDimension The number of voxel centers along the longest side 
///                         of the model's bounding box.
/// \param[in] voxSplitRes The max internal voxelization size.
/// \return An array of <tt>unsigned int</tt>, where each bit represents a 
///         voxel.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
NodePointer<Node> Voxelizer<Node, SNode>::voxelizeToRAM( uint  maxDimension,
                                                  uint3 voxSplitRes )
{
    NodePointer<Node> result = NodePointer<Node>();

    if ( Node::isFCCNode() )
    {
        std::cout << "Please use a standard Node-type when producing a " <<
                     "plain voxelization" << std::endl;
        return result;
    }

    this->setResolution( maxDimension );

    this->voxelizeEntry( make_uint2( 1 )
        , voxSplitRes 
        , make_uint3( 0 )
        , NULL );

    result = this->collectData( this->devices[0], true );

    this->deallocate();

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Produces a plain, integer voxelization based on the given arguments.
///
/// \throws Exception if CUDA encounters any errors.
///
/// \param[in] cubeLength Distance between \p Nodes along any of the three main
///                       axes.
/// \param[in] voxSplitRes The max internal voxelization size.
/// \return An array of <tt>unsigned int</tt>, where each bit represents a 
///         voxel.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
NodePointer<Node> Voxelizer<Node, SNode>::voxelizeToRAM( double cubeLength,
                                                  uint3 voxSplitRes )
{
    NodePointer<Node> result = NodePointer<Node>();

    if ( Node::isFCCNode() )
    {
        std::cout << "Please use a standard Node-type when producing a " <<
                     "plain voxelization" << std::endl;
        return result;
    }

    this->options.voxelDistanceGiven = true;
    this->hostVars.voxelLength = cubeLength;

    this->voxelizeEntry( make_uint2( 1 )
        , voxSplitRes 
        , make_uint3( 0 )
        , NULL );

    result = this->collectData( this->devices[0], true );

    this->deallocate();

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Iterates through each of the model's vertices and calculates the lower 
/// and upper bounds of the bounding box. They are caled \p minVertex and 
/// \p maxVertex. Then, the number of voxel centers along each dimension is 
/// determined. Since the number of vertices along the longest side is known, 
/// the rest of the sides can be determined by scaling. The x-axis is increased 
/// by 2 to make room for a zero-padding around the voxelization. The other 
/// directions are similarly increated elsewhere -- not here. The distance 
/// between voxel centers is also calculated and stored in \p voxelLength.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::determineBBAndResolution()
{
    this->hostVars.minVertex = make_double3( this->vertices[0], 
                                             this->vertices[1],
                                             this->vertices[2] );
    this->hostVars.maxVertex = make_double3( this->vertices[0], 
                                             this->vertices[1], 
                                             this->vertices[2] );

    // Calculating the upper and lower bounds of the bounding box.
    for (uint i = 1; i < this->hostVars.nrOfVertices; i++)
    {
        if (this->vertices[3*i] > this->hostVars.maxVertex.x)
            this->hostVars.maxVertex.x = this->vertices[3*i];
        if (this->vertices[3*i + 1] > this->hostVars.maxVertex.y)
            this->hostVars.maxVertex.y = this->vertices[3*i + 1];
        if (this->vertices[3*i + 2] > this->hostVars.maxVertex.z)
            this->hostVars.maxVertex.z = this->vertices[3*i + 2];

        if (this->vertices[3*i] < this->hostVars.minVertex.x)
            this->hostVars.minVertex.x = this->vertices[3*i];
        if (this->vertices[3*i + 1] < this->hostVars.minVertex.y)
            this->hostVars.minVertex.y = this->vertices[3*i + 1];
        if (this->vertices[3*i + 2] < this->hostVars.minVertex.z)
            this->hostVars.minVertex.z = this->vertices[3*i + 2];
    }

    double3 diffVertex = this->hostVars.maxVertex - this->hostVars.minVertex;

    if (diffVertex.x > diffVertex.y)
    {
        if (diffVertex.x > diffVertex.z)
        {
            // X is the longest side.
            this->hostVars.voxelLength = 
                double(diffVertex.x) / 
                double(this->hostVars.resolution.max.x - 1);
            this->hostVars.resolution.max.y = uint( 
                ceil( double(diffVertex.y) / this->hostVars.voxelLength ) ) 
                + 1;
            this->hostVars.resolution.max.z = uint( 
                ceil( double(diffVertex.z) / this->hostVars.voxelLength ) ) 
                + 1;

            // X-padding.
            this->hostVars.resolution.max.x += 2;
        }
        else
        {
            // Z is the longest side.
            this->hostVars.voxelLength = 
                double(diffVertex.z) / 
                double(this->hostVars.resolution.max.x - 1);
            this->hostVars.resolution.max.z = this->hostVars.resolution.max.x;
            this->hostVars.resolution.max.x = uint(
                ceil( double(diffVertex.x) / this->hostVars.voxelLength ) ) 
                + 1;
            this->hostVars.resolution.max.y = uint(
                ceil( double(diffVertex.y) / this->hostVars.voxelLength ) ) 
                + 1;

            // X-padding.
            this->hostVars.resolution.max.x += 2;
        }
    }
    else
    {
        if (diffVertex.y > diffVertex.z)
        {
            // Y is the longest side.
            this->hostVars.voxelLength = 
                double(diffVertex.y) / 
                double(this->hostVars.resolution.max.x - 1);
            this->hostVars.resolution.max.y = this->hostVars.resolution.max.x;
            this->hostVars.resolution.max.x = uint(
                ceil( double(diffVertex.x) / this->hostVars.voxelLength) ) + 1;
            this->hostVars.resolution.max.z = uint(
                ceil( double(diffVertex.z) / this->hostVars.voxelLength) ) + 1;

            // X-padding.
            this->hostVars.resolution.max.x += 2;
        }
        else
        {
            // Z is the longest side.
            this->hostVars.voxelLength = 
                double(diffVertex.z) / 
                double(this->hostVars.resolution.max.x - 1);
            this->hostVars.resolution.max.z = this->hostVars.resolution.max.x;
            this->hostVars.resolution.max.x = uint(
                ceil( double(diffVertex.x) / this->hostVars.voxelLength ) ) 
                + 1;
            this->hostVars.resolution.max.y = uint(
                ceil( double(diffVertex.y) / this->hostVars.voxelLength ) ) 
                + 1;

            // X-padding.
            this->hostVars.resolution.max.x += 2;
        }
    }

    if (this->options.verbose) 
        std::cout << "Initial resolution: X: " << 
                     this->hostVars.resolution.max.x << 
                     ", Y: " << this->hostVars.resolution.max.y << ", Z: " << 
                     this->hostVars.resolution.max.z << "\n";
}
///////////////////////////////////////////////////////////////////////////////
/// Iterates through each of the model's vertices and calculates the lower 
/// and upper bounds of the bounding box. They are called \p minVertex and 
/// \p maxVertex.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::calculateBoundingBox()
{
    this->hostVars.minVertex = make_double3( this->vertices[0], 
                                             this->vertices[1],
                                             this->vertices[2] );
    this->hostVars.maxVertex = make_double3( this->vertices[0], 
                                             this->vertices[1], 
                                             this->vertices[2] );

    // Calculating the upper and lower bounds of the bounding box.
    for (uint i = 1; i < this->hostVars.nrOfVertices; i++)
    {
        if (this->vertices[3*i] > this->hostVars.maxVertex.x)
            this->hostVars.maxVertex.x = this->vertices[3*i];
        if (this->vertices[3*i + 1] > this->hostVars.maxVertex.y)
            this->hostVars.maxVertex.y = this->vertices[3*i + 1];
        if (this->vertices[3*i + 2] > this->hostVars.maxVertex.z)
            this->hostVars.maxVertex.z = this->vertices[3*i + 2];

        if (this->vertices[3*i] < this->hostVars.minVertex.x)
            this->hostVars.minVertex.x = this->vertices[3*i];
        if (this->vertices[3*i + 1] < this->hostVars.minVertex.y)
            this->hostVars.minVertex.y = this->vertices[3*i + 1];
        if (this->vertices[3*i + 2] < this->hostVars.minVertex.z)
            this->hostVars.minVertex.z = this->vertices[3*i + 2];
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Since the number of vertices along the longest side is known, 
/// the rest of the sides can be determined by scaling. The x-axis is increased 
/// by 2 to make room for a zero-padding around the voxelization. The other 
/// directions are similarly increated elsewhere -- not here. The distance 
/// between voxel centers is also calculated and stored in \p voxelLength.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::determineDimensions()
{
    double3 diffVertex = this->hostVars.maxVertex - this->hostVars.minVertex;

    if (diffVertex.x > diffVertex.y)
    {
        if (diffVertex.x > diffVertex.z)
        {
            // X is the longest side.
            this->hostVars.voxelLength = 
                double(diffVertex.x) / 
                double(this->hostVars.resolution.max.x - 1);
            this->hostVars.resolution.max.y = uint( 
                ceil( double(diffVertex.y) / this->hostVars.voxelLength ) ) 
                + 1;
            this->hostVars.resolution.max.z = uint( 
                ceil( double(diffVertex.z) / this->hostVars.voxelLength ) ) 
                + 1;

            // X-padding.
            this->hostVars.resolution.max.x += 2;
        }
        else
        {
            // Z is the longest side.
            this->hostVars.voxelLength = 
                double(diffVertex.z) / 
                double(this->hostVars.resolution.max.x - 1);
            this->hostVars.resolution.max.z = this->hostVars.resolution.max.x;
            this->hostVars.resolution.max.x = uint(
                ceil( double(diffVertex.x) / this->hostVars.voxelLength ) ) 
                + 1;
            this->hostVars.resolution.max.y = uint(
                ceil( double(diffVertex.y) / this->hostVars.voxelLength ) ) 
                + 1;

            // X-padding.
            this->hostVars.resolution.max.x += 2;
        }
    }
    else
    {
        if (diffVertex.y > diffVertex.z)
        {
            // Y is the longest side.
            this->hostVars.voxelLength = 
                double(diffVertex.y) / 
                double(this->hostVars.resolution.max.x - 1);
            this->hostVars.resolution.max.y = this->hostVars.resolution.max.x;
            this->hostVars.resolution.max.x = uint(
                ceil( double(diffVertex.x) / this->hostVars.voxelLength) ) 
                + 1;
            this->hostVars.resolution.max.z = uint(
                ceil( double(diffVertex.z) / this->hostVars.voxelLength) ) 
                + 1;

            // X-padding.
            this->hostVars.resolution.max.x += 2;
        }
        else
        {
            // Z is the longest side.
            this->hostVars.voxelLength = 
                double(diffVertex.z) / 
                double(this->hostVars.resolution.max.x - 1);
            this->hostVars.resolution.max.z = this->hostVars.resolution.max.x;
            this->hostVars.resolution.max.x = uint(
                ceil( double(diffVertex.x) / this->hostVars.voxelLength ) ) 
                + 1;
            this->hostVars.resolution.max.y = uint(
                ceil( double(diffVertex.y) / this->hostVars.voxelLength ) ) 
                + 1;

            // X-padding.
            this->hostVars.resolution.max.x += 2;
        }
    }

    if (this->options.verbose) 
        std::cout << "Initial resolution: X: " << 
                     this->hostVars.resolution.max.x << ", Y: " << 
                     this->hostVars.resolution.max.y << ", Z: " << 
                     this->hostVars.resolution.max.z << "\n";
}
///////////////////////////////////////////////////////////////////////////////
/// Calculates the lengths of the sides of the models bounding box and 
/// determines the number of voxels along each side given the already known 
/// voxel length. The x-axis is increased by 2 to make room for a zero-padding 
/// around the voxelization. The other directions are similarly increated 
/// elsewhere -- not here. The distance between voxel centers is also calculated 
/// and stored in \p voxelLength.
/// 
/// \param[in] d Distance between sample points in a normal voxelization.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::determineDimensions( double d )
{
    double3 diffVertex = this->hostVars.maxVertex - this->hostVars.minVertex;
    this->hostVars.voxelLength = d;

    this->hostVars.resolution.max.x = uint( ceil( diffVertex.x / d ) ) + 1;
    this->hostVars.resolution.max.y = uint( ceil( diffVertex.y / d ) ) + 1;
    this->hostVars.resolution.max.z = uint( ceil( diffVertex.z / d ) ) + 1;

    // X-padding.
    this->hostVars.resolution.max.x += 2;

    if (this->options.verbose) 
        std::cout << "Initial resolution: X: " << 
                     this->hostVars.resolution.max.x << 
                     ", Y: " << this->hostVars.resolution.max.y << ", Z: " << 
                     this->hostVars.resolution.max.z << "\n";
}
///////////////////////////////////////////////////////////////////////////////
/// The length of the array along the x-axis needs to be divisible by 32
/// due to the use of integers to store voxel information. If the x-axis is 
/// to be split into multiple parts, then each part also has to be divisible
/// by 32.
///
/// \param[in] xSplits Into how many parts the array is split along the 
///                    x-direction.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::adjustResolution( uint xSplits )
{
    int xMod = VOX_BPI * xSplits;

    // Make the resolution.max. divisible by their respective Mod values.
    if (this->hostVars.resolution.max.x % xMod != 0)
        this->hostVars.resolution.max.x += 
            xMod - (this->hostVars.resolution.max.x % xMod);

    if (this->options.verbose) 
        std::cout << "Changed resolution to X: " << 
                this->hostVars.resolution.max.x << 
                ", Y: " << this->hostVars.resolution.max.y << ", Z: " << 
                this->hostVars.resolution.max.z << "\n";
}
///////////////////////////////////////////////////////////////////////////////
/// Calculates the size of the <em>work queue</em> and the contents of the 
/// <em>offset buffer</em> based on how many \a tiles overlap each triangle. 
/// The size of the <em>work queue</em> is essentially the sum of the number of 
/// <em>tile overlaps</em> for each triangle, and the <em>offset buffer</em> 
/// contains the partial sums of the summing.
/// 
/// \param[in] device Which device's variables are being accessed.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::prepareForConstructWorkQueue( DevContext<Node,SNode> & device )
{
    if (this->options.verbose) 
        std::cout << (clock() - this->startTime) << 
                     ": Calling prepareForConstructWorkQueue.\n";

    // Set up the offset buffer. Also mark the index of the first triangle 
    // with tile overlaps.
    for ( uint i = 0; i < this->hostVars.nrOfTriangles; i++ )
    {
        if (device.tileOverlaps[i] == 0)
        {
            device.offsetBuffer[i] = 0;
        }
        else
        {
            if (device.data.firstTriangleWithTiles < 0)
                device.data.firstTriangleWithTiles = i;

            device.offsetBuffer[i] = device.data.workQueueSize;
            device.data.workQueueSize += device.tileOverlaps[i];
        }
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Opens an output \p filestream on \p log, and writes the time and date
/// into it. After this function call, the \p log is ready to accept input.
///
/// \param[in] filename The name of the logfile.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::openLog( char const * filename )
{
    this->log.open( filename, std::ios::out | std::ios::trunc );
    std::time_t rawtime;
    std::time( &rawtime );
    struct std::tm * timeinfo = std::localtime( &rawtime );
    char * timeAndDate = std::asctime(timeinfo);
    this->log << "Voxelizer logfile: " << timeAndDate;
    free( timeAndDate );
}
///////////////////////////////////////////////////////////////////////////////
/// Automatically opens the logfile and writes general information about the 
/// voxelization, such as the resolution, how many integers the x-axis needs to
/// represent its voxels and all triangle data including normals.
///
/// \param[in] device Which device's information to write.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::printGeneralInfo( DevContext<Node,SNode> & device )
{
    this->openLog("Voxelizer.log");

    uint intsPerX = device.data.resolution.max.x >> VOX_DIV;

    this->log << "Resolution: (" << device.data.resolution.max.x << ", " << 
                 device.data.resolution.max.y << ", " << 
                 device.data.resolution.max.z << "), Integers on the " << 
                 "x-axis: " << intsPerX << ".\n\n";

    this->log << "Printing triangle data:\n";

    double3 v0, v1, v2, n;
    for (uint i = 0; i < this->hostVars.nrOfTriangles; i++)
    {
        v0 = make_double3(
            this->vertices[3*this->indices[3*i + 0] + 0], 
            this->vertices[3*this->indices[3*i + 0] + 1], 
            this->vertices[3*this->indices[3*i + 0] + 2] );
        v1 = make_double3(
            this->vertices[3*this->indices[3*i + 1] + 0], 
            this->vertices[3*this->indices[3*i + 1] + 1], 
            this->vertices[3*this->indices[3*i + 1] + 2] );
        v2 = make_double3(
            this->vertices[3*this->indices[3*i + 2] + 0], 
            this->vertices[3*this->indices[3*i + 2] + 1], 
            this->vertices[3*this->indices[3*i + 2] + 2] );
        n = normalize(cross(v0 - v2, v1 - v0));

        this->log << "TID: " << i << "\n";
        this->log << " V1: (" << v0.x << ", " << v0.y << ", " << v0.z << ")\n";
        this->log << " V2: (" << v1.x << ", " << v1.y << ", " << v1.z << ")\n";
        this->log << " V3: (" << v2.x << ", " << v2.y << ", " << v2.z << ")\n";
        this->log << "  N: (" << n.x << ", " << n.y << ", " << n.z << ")\n\n";
    }
    
}
///////////////////////////////////////////////////////////////////////////////
/// Writes to the \p log how many tiles overlap each triangle. Must be called 
/// after \p calcTileOverlap() has been called.
///
/// \todo Remove \p direction as it is no longer used.
///
/// \param[in] device Which device's tile overlaps should be written.
/// \param[in] direction Deprecated.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::printTileOverlaps( DevContext<Node,SNode> & device
                                            , MainAxis direction )
{
    this->log << "Printing tile overlaps for the " << 
                 (direction == xAxis ? "x" : (direction == yAxis ? "y" : "z"))
                 << " axis.\n";
    for (uint i = 0; i < this->hostVars.nrOfTriangles; i++)
    {
        this->log << "[" << i << " : " << 
                     device.tileOverlaps[i] << "] ";
        if ((i + 1) % 10 == 0)
            this->log << "\n";
    }
    this->log << "\n";
}
///////////////////////////////////////////////////////////////////////////////
/// Writes to the \p log the offsets to where the data for each \a tile begins 
/// in the <em>work queue</em>.
///
/// \todo Remove \p direction as it is no longer used.
///
/// \param[in] device Which device is being used.
/// \param[in] direction Deprecated.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::printOffsetBuffer( DevContext<Node,SNode> & device
                                            , MainAxis direction )
{
    this->log << "Printing the offset buffer for the " << 
        (direction == xAxis ? "x" : (direction == yAxis ? "y" : "z")) << 
        " axis.\n";
    for (uint i = 0; i < this->hostVars.nrOfTriangles; i++)
    {
        this->log << "[" << i << " : " << 
                     device.offsetBuffer[i] << "] ";

        if ((i + 1) % 10 == 0)
            this->log << "\n";
    }
    this->log << "\n";
}
///////////////////////////////////////////////////////////////////////////////
/// Writes to the \p log the contents of the <em>work queue</em>. This involves 
/// printing all triangle-tile pairs.
///
/// \todo Remove \p direction as it is no longer used.
///
/// \param[in] device Which device is being used.
/// \param[in] direction Deprecated.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
void Voxelizer<Node, SNode>::printWorkQueue( 
    DevContext<Node,SNode> & device, 
    MainAxis direction )
{
    this->log << "Printing the work queue for the " << 
        (direction == xAxis ? "x" : (direction == yAxis ? "y" : "z")) << 
        " axis.\n";
    uint i_max = device.data.workQueueSize;

    for (uint i = 0; i < i_max; i++)
    {
        this->log << "[" << device.workQueueTriangles[i] << 
                     ", " << device.workQueueTiles[i] << "] ";
        if ((i + 1) % 10 == 0)
            this->log << "\n";
    }
    this->log << "\n";
}
///////////////////////////////////////////////////////////////////////////////
/// Writes to the \p log the contents of the <em>sorted work queue</em>. This 
/// involves printing all tile-triangle pairs.
///
/// \todo Remove \p direction as it is no longer used.
///
/// \param[in] device Which device is being used.
/// \param[in] direction Deprecated.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
void Voxelizer<Node, SNode>::printSortedWorkQueue( DevContext<Node,SNode> & device
                                               , MainAxis direction )
{
    this->log << "Printing the sorted work queue for the " << 
        (direction == xAxis ? "x" : (direction == yAxis ? "y" : "z")) << 
        " axis.\n";
    uint i_max = device.data.workQueueSize;

    for (uint i = 0; i < i_max; i++)
    {
        this->log << "[" << device.workQueueTiles[i] << ", " << 
                     device.workQueueTriangles[i] << "] ";
        if ((i + 1) % 10 == 0)
            this->log << "\n";
    }
    this->log << "\n";
}
///////////////////////////////////////////////////////////////////////////////
/// Writes to the \p log the contents of the <em>compacted tile list</em>. For 
/// each \a tile, the offset to the <em>work queue</em> where its tile-triangle 
/// pairs begin is printed.
///
/// \todo Remove \p direction as it is no longer used.
///
/// \param[in] device Which device is being used.
/// \param[in] direction Deprecated.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
void Voxelizer<Node, SNode>::printCompactedList( DevContext<Node,SNode> & device
                                             , MainAxis direction )
{
    this->log << "Printing the compacted tile list for the " << 
        (direction == xAxis ? "x" : (direction == yAxis ? "y" : "z")) <<
        " axis.\n";
    for (uint i = 0; i < device.data.nrValidElements; i++)
    {
        this->log << "[" << device.tileList[i] << ", " << 
                     device.tileOffsets[i] << "] ";
        if ((i + 1) % 10 == 0)
            this->log << "\n";
    }
    this->log << "\n";
}

template <class Node, class SNode> 
void Voxelizer<Node, SNode>::closeLog()
{
    this->log.close();
}
///////////////////////////////////////////////////////////////////////////////
/// Returns the number of cores per SM for a variety of GPU architectures.
/// This function was directly copied from a header file in the CUDA SDK.
///
/// \param[in] major Major compute capability.
/// \param[in] minor Minor compute capability.
///
/// \return Number of cores per SM.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
inline int Voxelizer<Node, SNode>::convertSMVer2Cores(
    int major, 
    int minor ) const
{
    // Defines for GPU Architecture types (using the SM version to determine 
    // the # of cores per SM
    typedef struct
    {
        // 0xMm (hexidecimal notation), M = SM Major version, and m = 
        // SM minor version
        int SM;
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }
    
    // If we don't find the values, we default use the previous one to run properly
    printf( "MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores \
            /SM\n", 
            major, 
            minor, 
            nGpuArchCoresPerSM[7].Cores );
    return nGpuArchCoresPerSM[7].Cores;
}
///////////////////////////////////////////////////////////////////////////////
/// Splits a given space (in voxel coordinates) into a number of \a subspaces
/// by limiting the maximum length of each \a subspace in each direction. For
/// example, if the x-direction of the given space has length 1222, and we set
/// the maximum length of the x-direction to 512, then the space needs to be 
/// split into 3 parts in order to make each part at most 512 voxels in length.
/// The splitting tries to keep each length equally long, when possible.
///
/// \param[in] maxDimensions The maximum lengths of the \a subspaces along each 
///                          direction.
/// \param[in] resolution The initial space to be subdivided.
///
/// \return A pair that contains both the number of splits along each dimension 
///         as well as a unique_ptr to an array of \p Bounds objects.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
SplitData<uint3> Voxelizer<Node, SNode>::splitResolutionByMaxDim
    ( uint3		    const & maxDimensions
    , Bounds<uint3> const & resolution 
    )
{
    // Calculate the required number of partitions and forward calculations to
    // splitResolutionByNrOfParts.
    uint3 const res = resolution.max - resolution.min;
    uint3 const nrOfPartitions = make_uint3(
        ( res.x + maxDimensions.x - 1 ) / maxDimensions.x,
        ( res.y + maxDimensions.y - 1 ) / maxDimensions.y,
        ( res.z + maxDimensions.z - 1 ) / maxDimensions.z );

    SplitData<uint3> result = { 
        nrOfPartitions,
        this->splitResolutionByNrOfParts( nrOfPartitions, resolution ) 
    };

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Splits a given space (in voxel coordinates) into a number of \a subspaces by 
/// limiting the maximum length of each \a subspace in each direction. This is a 
/// two-dimensional variant of this function, and it functions by ignoring the 
/// x-axis. Only the y- and z-axes can be split along.
///
/// \param[in] maxDimensions The maximum lengths of the \a subspaces along the 
///                          y- and z-directions.
/// \param[in] resolution The initial space to be subdivided.
///
/// \return A pair that contains both the number of splits along each dimension 
///         as well as a unique_ptr to an array of \p Bounds objects.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
SplitData<uint2> Voxelizer<Node, SNode>::splitResolutionByMaxDim
    ( uint2		    const & maxDimensions
    , Bounds<uint3> const & resolution 
    )
{
    // Calculate the required number of partitions and forward calculations to
    // splitResolutionByNrOfParts.
    uint2 const res = make_uint2(
        resolution.max.y - resolution.min.y,
        resolution.max.z - resolution.min.z );
    uint2 const nrOfPartitions = make_uint2(
        ( res.x + maxDimensions.x - 1 ) / maxDimensions.x,
        ( res.y + maxDimensions.y - 1 ) / maxDimensions.y );

    SplitData<uint2> result = { 
        nrOfPartitions,
        this->splitResolutionByNrOfParts( nrOfPartitions, resolution ) 
    };

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Splits a given space (in voxel coordinates) into a number of \a subspaces by 
/// limiting the maximum length of the \a subspace along the x-direction. This 
/// is a one-dimensional variant of this function, and it only splits spaces 
/// along the x-axis. 
///
/// \param[in] maxDimension The maximum length of the \a subspaces along the 
///                         x-direction.
/// \param[in] resolution The initial space to be subdivided.
///
/// \return A pair that contains both the number of splits along the x-axis 
///         as well as a unique_ptr to an array of \p Bounds objects.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode> 
SplitData<uint> Voxelizer<Node, SNode>::splitResolutionByMaxDim
    ( uint				  maxDimension
    , Bounds<uint3> const & resolution 
    )
{
    // Calculate the required number of partitions and forward calculations to
    // splitResolutionByNrOfParts.
    uint const res = resolution.max.x - resolution.min.x;

    uint const nrOfPartitions = ( res + maxDimension - 1 ) / maxDimension;

    SplitData<uint> result = { 
        nrOfPartitions,
        this->splitResolutionByNrOfParts( nrOfPartitions, resolution ) 
    };

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Splits a given space (in voxel coordinates) into a number of \a subspaces
/// by directly specifying how many divisions there should be along each 
/// direction. For example, if the initial space has a length of 1222 along the 
/// x-direction, and the number of x-splits is set to 3, then the lengths of
/// the \a subspaces along the x-direction would be 407, 407 and 408.
/// The splitting tries to keep each length equally long, when possible.
///
/// \param[in] nrOfPartitions The desired number of subdivisions along each 
///                           direction.
/// \param[in] resolution The initial space to be subdivided.
///
/// \return A unique_ptr that contains an array of \p Bounds objects.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
boost::shared_array<Bounds<uint3> > 
Voxelizer<Node, SNode>::splitResolutionByNrOfParts
    ( uint3		    const & nrOfPartitions
    , Bounds<uint3> const & resolution 
    )
{
    // Number of subspaces.
    uint const resultSize = 
        nrOfPartitions.x * nrOfPartitions.y * nrOfPartitions.z;

    boost::shared_array<Bounds<uint3> > result( 
        new Bounds<uint3>[resultSize] );

    // The algorithm essentially divides a length into multiple sublengths by
    // dividing the initial length with the number of splits and rounding down.
    // This is the first sublength. Then, the sublength is subtracted from 
    // the total length and this new partial length is divided by the number of
    // splits minus one, and the resulting sublength is again rounded down.
    // This continues until the final length is divided by one, at which point
    // the algorithm is considered finished.

    uint3 prev_min = resolution.max - resolution.min;

    Bounds<uint> xBound, yBound, zBound;

    boost::scoped_array<Bounds<uint> > xBounds( 
        new Bounds<uint>[nrOfPartitions.x] );

    boost::scoped_array<Bounds<uint> > yBounds(
        new Bounds<uint>[nrOfPartitions.y] );

    boost::scoped_array<Bounds<uint> > zBounds(
        new Bounds<uint>[nrOfPartitions.z] );

    uint const maxNrOfPartitions = 
        max( max( nrOfPartitions.x, nrOfPartitions.y ), nrOfPartitions.z );

    for ( uint i = maxNrOfPartitions; i > 0; --i ) {

        if ( i <= nrOfPartitions.x ) {
            xBound.max = prev_min.x;
            xBound.min = xBound.max - ( xBound.max / i );
            prev_min.x = xBound.min;

            xBounds[i - 1] = xBound;
        }
        if ( i <= nrOfPartitions.y ) {
            yBound.max = prev_min.y;
            yBound.min = yBound.max - ( yBound.max / i );
            prev_min.y = yBound.min;

            yBounds[i - 1] = yBound;
        }
        if ( i <= nrOfPartitions.z ) {
            zBound.max = prev_min.z;
            zBound.min = zBound.max - ( zBound.max / i );
            prev_min.z = zBound.min;

            zBounds[i - 1] = zBound;
        }
    }
    // All permutations of sublengths are combined together to create the full 
    // set of subspaces.
    int currentPartition = 0;
    for ( uint z = 0; z < nrOfPartitions.z; ++z ) {
        for ( uint y = 0; y < nrOfPartitions.y; ++y ) {
            for ( uint x = 0; x < nrOfPartitions.x; ++x ) {
                result[currentPartition].min = 
                    make_uint3( resolution.min.x + xBounds[x].min, 
                                resolution.min.y + yBounds[y].min, 
                                resolution.min.z + zBounds[z].min );
                result[currentPartition].max = 
                    make_uint3( resolution.min.x + xBounds[x].max, 
                                resolution.min.y + yBounds[y].max, 
                                resolution.min.z + zBounds[z].max );
                currentPartition++;
            }
        }
    }

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Splits a given space (in voxel coordinates) into a number of \a subspaces
/// by directly specifying how many divisions there should be along each 
/// direction. This is the two-dimensional version of this function. It only 
/// accepts splitting along the y- and z-direction, and leaves the x-direction
/// intact.
///
/// \param[in] nrOfPartitions The desired number of subdivisions along the y- 
///                           and z-directions.
/// \param[in] resolution The initial space to be subdivided.
///
/// \return A unique_ptr that contains an array of \p Bounds objects.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
boost::shared_array<Bounds<uint2> >  Voxelizer<Node, SNode>::splitResolutionByNrOfParts
    ( uint2		    const & nrOfPartitions
    , Bounds<uint3> const & resolution
    )
{
    uint const resultSize = nrOfPartitions.x * nrOfPartitions.y;

    boost::shared_array<Bounds<uint2> > result( 
        new Bounds<uint2>[resultSize] );

    // More or less identical to the three-dimensional version.

    uint2 prev_min = make_uint2(
        resolution.max.y - resolution.min.y,
        resolution.max.z - resolution.min.z );

    Bounds<uint> yBound, zBound;

    boost::scoped_array<Bounds<uint> > yBounds(
        new Bounds<uint>[nrOfPartitions.x] );
    boost::scoped_array<Bounds<uint> > zBounds(
        new Bounds<uint>[nrOfPartitions.y] );

    uint const maxNrOfPartitions = max( nrOfPartitions.x, nrOfPartitions.y );

    for ( uint i = maxNrOfPartitions; i > 0; --i ) {

        if ( i <= nrOfPartitions.x ) {
            yBound.max = prev_min.x;
            yBound.min = yBound.max - ( yBound.max / i );
            prev_min.x = yBound.min;

            yBounds[i - 1] = yBound;
        }
        if ( i <= nrOfPartitions.y ) {
            zBound.max = prev_min.y;
            zBound.min = zBound.max - ( zBound.max / i );
            prev_min.y = zBound.min;

            zBounds[i - 1] = zBound;
        }
    }

    int currentPartition = 0;
    for ( uint z = 0; z < nrOfPartitions.y; ++z ) {
        for ( uint y = 0; y < nrOfPartitions.x; ++y ) {
            result[currentPartition].min = 
                make_uint2( resolution.min.y + yBounds[y].min, 
                            resolution.min.z + zBounds[z].min );
            result[currentPartition].max = 
                make_uint2( resolution.min.y + yBounds[y].max, 
                            resolution.min.z + zBounds[z].max );
            currentPartition++;
        }
    }

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Splits a given space (in voxel coordinates) into a number of \a subspaces
/// by directly specifying how many divisions there should be along the 
/// x-direction. This is the one-dimensional version of this function. It only 
/// accepts splitting along the x-direction, and leaves the other directions
/// intact.
///
/// \param[in] nrOfPartitions The desired number of \a subdivisions along the 
///                           x-direction.
/// \param[in] resolution The initial space to be subdivided.
///
/// \return A unique_ptr that contains an array of \p Bounds objects.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
boost::shared_array<Bounds<uint> > Voxelizer<Node, SNode>::splitResolutionByNrOfParts
    ( uint				  nrOfPartitions 
    , Bounds<uint3> const & resolution
    )
{
    boost::shared_array<Bounds<uint> > result( 
        new Bounds<uint>[nrOfPartitions] );

    // Identical to the two- and three-dimensional versions, except that it is
    // much simpler due to there only being one dimension to worry about.

    uint prev_min = resolution.max.x - resolution.min.x;

    Bounds<uint> xBound;

    for ( uint i = nrOfPartitions; i > 0; --i ) {

        xBound.max = prev_min;
        xBound.min = xBound.max - ( xBound.max / i );
        prev_min = xBound.min;

        result[i - 1].min = resolution.min.x + xBound.min;
        result[i - 1].max = resolution.min.x + xBound.max;
    }

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Enables or disables verbose output during the execution of the voxelizer.
/// Verbose output is set to \p false by default.
/// 
/// \param[in] verbose \p true to enable, \p false to disable verbose output.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
void Voxelizer<Node, SNode>::verboseOutput(bool verbose) throw()
{
    this->options.verbose = verbose;
}
///////////////////////////////////////////////////////////////////////////////
/// Rotates all vertices of the model 90 degrees counter clockwise around the 
/// z-axis. This is used to simulate slicing along the x-axis by rotating the 
/// model and instead slicing along the y-axis. The new vertex after the 
/// rotation is: \f[ \left[ \begin{array}{c} x' \\ y' \\ z' \end{array} 
/// \right] = \left[ \begin{array}{c} a - y \\ b + x \\ z \end{array} \right], 
/// \f] where \f$ a = c_{y} + c_{x} \f$, \f$ b = c_{y} - c_{x} \f$ and \f$ c 
/// \f$ is the \a centroid of the model's bounding box.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
void Voxelizer<Node, SNode>::rotateVertices()
{
    double3 minC = this->hostVars.minVertex;
    double3 maxC = this->hostVars.maxVertex;

    double2 centroid = {
        ( maxC.x - minC.x ) / 2.0,
        ( maxC.y - minC.y ) / 2.0
    };

    double a = centroid.y + centroid.x;
    double b = centroid.y - centroid.x;

    for ( uint i = 0; i < this->hostVars.nrOfVertices; ++i ) {
        double tempX = this->vertices[3*i];
        this->vertices[3*i] = 
            float(a - double(this->vertices[3*i + 1]));
        this->vertices[3*i + 1] = float(b + tempX);
    }
}
///////////////////////////////////////////////////////////////////////////////
/// Performs the opposite rotation of \p rotateVertices(), effectively undoing 
/// the rotation. This function only works on voxel / \p Node coordinates. Can 
/// rotate both normal and FCC \p Nodes.
///
/// \param[in] vec The voxel coordinates to be transformed.
/// \param[in] xDim The size of the voxel array in the x-direction before the 
///                 tranformation.
///
/// \return The tranformed coordinates.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
uint3 Voxelizer<Node, SNode>::unRotateCoords( uint3 vec, uint xDim )
{
    uint3 result = vec;

    if ( Node::isFCCNode() )
    {
        result.x = 2*vec.y + (vec.x + vec.z + 1) % 2;
        result.y = (xDim / 2) - 1 - (vec.x / 2);
    }
    else
    {
        result.x = vec.y;
        result.y = xDim - 1 - vec.x;
    }

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Calculates the dimensions of the array of Nodes that the slice voxelizer 
/// will produce. It should work on other forms of voxelization, as well, but 
/// most of the relevant information should be available in the \p NodePointer 
/// struct. Used mostly during the testing of the slice voxelizer, but maybe it
/// has other uses, as well. Works for both normal and FCC \p Nodes.
/// 
/// \param[in] longestSizeInVoxels How many voxel centers there are along the 
///                                length of the longest size of the model's 
///                                bounding box.
/// \param[in] maxInternalXSize How many voxels along the x-axis the voxelizer 
///                             should voxelize. Can safely be set to 1024.
/// \param[in] sliceAlongX \p true if the slicing happens along the x-axis, \p 
///                        false otherwise.
///
/// \return The dimensions of the voxel array should it be constructed with the
///         given parameters.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
uint3 Voxelizer<Node, SNode>::getArrayDimensions( uint longestSizeInVoxels
                                              , uint maxInternalXSize
                                              , bool sliceAlongX )
{
    if ( sliceAlongX ) {
        std::vector<float> vertsCopy( this->vertices );

        this->setResolution( longestSizeInVoxels );
        this->calculateBoundingBox();
        this->rotateVertices();
        this->calculateBoundingBox();
        this->determineDimensions();

        SplitData<uint> xSplits = 
            splitResolutionByMaxDim( maxInternalXSize
                                   , this->hostVars.resolution );

        this->adjustResolution( xSplits.counts );

        this->hostVars.resolution.max.y += 2;
        this->hostVars.resolution.max.z += 2;
        uint3 result = 
            this->hostVars.resolution.max - this->hostVars.resolution.min;

        Bounds<uint3> temp = {
            this->unRotateCoords( this->hostVars.resolution.min, result.x ),
            this->unRotateCoords( this->hostVars.resolution.max - 1, result.x )
        };

        this->hostVars.resolution.min = min( temp.min, temp.max );
        this->hostVars.resolution.max = max( temp.min, temp.max ) + 1;

        result = this->hostVars.resolution.max - this->hostVars.resolution.min;

        std::swap( this->vertices, vertsCopy );

        this->calculateBoundingBox();

        if ( Node::isFCCNode() )
        {
            result.x *= 2;
            result.z *= 2;
        }

        return result;
    }

    uint3 result;

    this->setResolution( longestSizeInVoxels );
    this->determineDimensions();

    SplitData<uint> xSplits = 
        splitResolutionByMaxDim( maxInternalXSize
                               , this->hostVars.resolution );

    this->adjustResolution( xSplits.counts );
    
    result = this->hostVars.resolution.max - this->hostVars.resolution.min;
    result.y += 2;
    result.z += 2;

    if ( Node::isFCCNode() )
    {
        result.x *= 2;
        result.z *= 2;
    }

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Calculates the dimensions of the array of Nodes that the slice voxelizer 
/// will produce. It should work on other forms of voxelization, as well, but 
/// most of the relevant information should be available in the \p NodePointer 
/// struct. Used mostly during the testing of the slice voxelizer, but maybe it
/// has other uses, as well. Works for both normal and FCC \p Nodes.
/// 
/// \param[in] cubeLength Distance between voxel centers along any of the three 
///                       main axes.
/// \param[in] maxInternalXSize How many voxels along the x-axis the voxelizer 
///                             should voxelize. Can safely be set to 1024.
/// \param[in] sliceAlongX \p true if the slicing happens along the x-axis, \p 
///                        false otherwise.
///
/// \return The dimensions of the voxel array should it be constructed with the
///         given parameters.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
uint3 Voxelizer<Node, SNode>::getArrayDimensions( double cubeLength
                                         , uint maxInternalXSize
                                         , bool sliceAlongX )
{
    if ( sliceAlongX ) {
        std::vector<float> vertsCopy( this->vertices );

        uint3 oldMaxDim = this->hostVars.resolution.max;
        uint3 oldMinDim = this->hostVars.resolution.min;

        this->rotateVertices();
        this->calculateBoundingBox();
        this->determineDimensions( cubeLength );

        SplitData<uint> xSplits = 
            splitResolutionByMaxDim( maxInternalXSize
                                   , this->hostVars.resolution );
        this->adjustResolution( xSplits.counts );

        this->hostVars.resolution.max.y += 2;
        this->hostVars.resolution.max.z += 2;

        if ( Node::isFCCNode() )
        {
            this->hostVars.resolution.min.x *= 2;
            this->hostVars.resolution.min.z *= 2;
            this->hostVars.resolution.max.x *= 2;
            this->hostVars.resolution.max.z *= 2;
        }

        uint3 result = 
            this->hostVars.resolution.max - this->hostVars.resolution.min;

        Bounds<uint3> temp = {
            this->unRotateCoords( this->hostVars.resolution.min, result.x ),
            this->unRotateCoords( this->hostVars.resolution.max - 1, result.x )
        };

        if ( Node::isFCCNode() )
        {
            this->hostVars.resolution.min = make_uint3( temp.min.x - 1
                                                      , temp.max.y
                                                      , temp.min.z );
            this->hostVars.resolution.max = make_uint3( temp.max.x + 1
                                                      , temp.min.y + 1
                                                      , temp.max.z + 1 );
        }
        else
        {
            this->hostVars.resolution.min = min( temp.min, temp.max );
            this->hostVars.resolution.max = max( temp.min, temp.max ) + 1;
        }

        result = this->hostVars.resolution.max - this->hostVars.resolution.min;

        this->hostVars.resolution.max = oldMaxDim;
        this->hostVars.resolution.min = oldMinDim;

        std::swap( this->vertices, vertsCopy );

        this->calculateBoundingBox();

        return result;
    }

    uint3 result;
    uint3 oldMaxDim = this->hostVars.resolution.max;
    uint3 oldMinDim = this->hostVars.resolution.min;

    this->determineDimensions( cubeLength );

    SplitData<uint> xSplits = 
        splitResolutionByMaxDim( maxInternalXSize
                               , this->hostVars.resolution );

    this->adjustResolution( xSplits.counts );
    
    result = this->hostVars.resolution.max - this->hostVars.resolution.min;
    result.y += 2;
    result.z += 2;

    this->hostVars.resolution.max = oldMaxDim;
    this->hostVars.resolution.min = oldMinDim;

    if ( Node::isFCCNode() )
    {
        result.x *= 2;
        result.z *= 2;
    }

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Allows for conflict-free reuse of already allocated memory by the plain 
/// voxelization algorithm. The work queue memsets could potentially be 
/// redundant.
///
/// \throws Exception if CUDA reports an error.
///
/// \param[in] device Which device should have its memory reset.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
void Voxelizer<Node, SNode>::resetDataStructures( DevContext<Node,SNode> & device )
{
    device.voxels_gpu.zero();
    device.workQueueTriangles_gpu.zero();
    device.workQueueTiles_gpu.zero();
}
///////////////////////////////////////////////////////////////////////////////
/// Calls collectData(NodePointer) for each device used during the 
/// voxelization. It returns NodePointer objects filled with the data of one 
/// device, which are then added to the vector object.
///
/// \return vector of \p NodePointer that contains data from all devices.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
std::vector<NodePointer<Node> > Voxelizer<Node, SNode>::collectData()
{
    std::vector<NodePointer<Node> > result;

    const bool uploadToHost = this->options.simulateMultidevice;

    for ( int i = 0; i < this->nrOfDevicesInUse; ++i )
        result.push_back( this->collectData( this->devices[i]
                                           , uploadToHost
                                           ) );

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Calls collectData(NodePointer) for each device used during the 
/// voxelization. It returns NodePointer objects filled with the data of one 
/// device, which are then added to the vector object. This function is meant 
/// to be used with the voxelizations involving two arrays.
///
/// \return vector of \p Node2APointer that contains data from all devices.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
std::vector<Node2APointer<Node, SNode> > 
    Voxelizer<Node, SNode>::collectSurfData()
{
    std::vector<Node2APointer<Node, SNode> > result;

    const bool uploadToHost = this->options.simulateMultidevice;

    for ( int i = 0; i < this->nrOfDevicesInUse; ++i )
        result.push_back( this->collectSurfData( this->devices[i]
                                               , uploadToHost
                                               ) );

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Takes into account the unique characteristics of every different output 
/// type. Usually called by collectData(vector<NodePointer>) as it cycles 
/// through the devices used during the voxelization. When voxelizing directly 
/// to device pointers, the vector argumented version should be used. When 
/// voxelizing directly to host pointers, this version should be used.
/// 
/// \throws Exception if CUDA reports an error.
///
/// \param[in,out] device Which device is being used.
/// \param[in] hostPointers \p true if a host pointer should be produced.
///
/// \return \p NodePointer that contains the data of a particular device.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
NodePointer<Node> Voxelizer<Node, SNode>::collectData
    ( DevContext<Node,SNode> & device
    , const bool         hostPointers
    )
{
    NodePointer<Node> result = NodePointer<Node>();

    // Dimensions of the output grid.
    const uint3 res = device.data.allocResolution.max - 
                      device.data.allocResolution.min;

    result.dev = device.data.dev;
    result.dim = res;
    result.loc = device.data.location;

    if ( this->options.nodeOutput )
    {   // Outputting an array of Nodes.

        if ( hostPointers )
        {   // Ouputting host pointers instead of device pointers.
            result.ptr = new Node[ device.nodes_gpu.size() ];

            cudaSetDevice( result.dev );

            if ( this->options.slices && this->options.sliceDirection == 0 )
            {   // Slicing along the x-axis: Read from the Node array copy.
                device.nodesCopy_gpu.copyTo( result.ptr );
            }
            else
            {   // Usually just read from the standard location.
                device.nodes_gpu.copyTo( result.ptr );
            }

        }
        else
        {   // Outputting device pointers instead of host pointers.
            if ( this->options.slices && this->options.sliceDirection == 0 )
            {   // Slicing along the x-axis: Read from the Node array copy.
                result.ptr = device.nodesCopy_gpu.release();
            }
            else
            {   // Usually just read from the standard location.
                result.ptr = device.nodes_gpu.release();
            }
        }
    }
    else
    {   // Outputting an array of unsigned integers.

        if ( hostPointers )
        {   // Ouputting host pointers instead of device pointers.
            result.vptr = new VoxInt[ device.voxels_gpu.size() ];

            cudaSetDevice( result.dev );

            device.voxels_gpu.copyTo( result.vptr );
        }
        else
        {   // Ouputting device pointers instead of host pointers.
            result.vptr = device.voxels_gpu.release();
        }
    }

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// Takes into account the unique characteristics of every different output 
/// type. Usually called by collectData(vector<NodePointer>) as it cycles 
/// through the devices used during the voxelization. When voxelizing directly 
/// to device pointers, the vector argumented version should be used. When 
/// voxelizing directly to host pointers, this version should be used. This 
/// function is meant to be used when voxelizing to two arrays.
/// 
/// \throws Exception if CUDA reports an error.
///
/// \param[in,out] device Which device is being used.
/// \param[in] hostPointers \p true if a host pointer should be produced.
///
/// \return \p Node2APointer that contains the data of a particular device.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
Node2APointer<Node, SNode> Voxelizer<Node, SNode>::collectSurfData
    ( DevContext<Node,SNode> & device
    , const bool         hostPointers
    )
{
    Node2APointer<Node, SNode> result = Node2APointer<Node, SNode>();

    // Dimensions of the output grid.
    const uint3 res = device.data.allocResolution.max - 
                      device.data.allocResolution.min;

    result.dev = device.data.dev;
    result.dim = res;
    result.loc = device.data.location;
    result.nrOfSurfNodes = device.data.nrOfSurfaceNodes;
    result.indices = device.data.hashMap;

    if ( hostPointers )
    {   // Ouputting host pointers instead of device pointers.
        result.nodes = new Node[ device.nodes_gpu.size() ];
        result.surfNodes = new SNode[ device.surfNodes_gpu.size() ];

        cudaSetDevice( result.dev );

        if ( this->options.slices && this->options.sliceDirection == 0 )
        {   // Slicing along the x-axis: Read from the Node array copy.
            device.nodesCopy_gpu.copyTo( result.nodes );
        }
        else
        {   // Usually just read from the standard location.
            device.nodes_gpu.copyTo( result.nodes );
            device.surfNodes_gpu.copyTo( result.surfNodes );
            result.indices.convertToHostMemory();
        }

    }
    else
    {   // Outputting device pointers instead of host pointers.
        if ( this->options.slices && this->options.sliceDirection == 0 )
        {   // Slicing along the x-axis: Read from the Node array copy.
            result.nodes = device.nodesCopy_gpu.release();
        }
        else
        {   // Usually just read from the standard location.
            result.nodes = device.nodes_gpu.release();
            result.surfNodes = device.surfNodes_gpu.release();
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// Sets the simulateMultiDevice variable to \p true before calling the 
/// provided function and returning its result. This function is really only 
/// intended to be used for testing purposes. The simplest way to use it is to 
/// give it a lambda function, like so:
///
/// \code
/// Voxelizer voxelizer( ... );
/// auto result = simulateMultiDevice(
///     [&] () { return voxelizer.voxelizeToNodes( ... ); } );
/// \endcode
///
/// The function that is called needs to return a vector of \p NodePointer, so 
/// it needs to be called with a function that returns device pointers. The 
/// results in the vector will be host pointers, however.
///
/// \throws Exception if CUDA encounters an error.
///
/// \param[in] func Function object that is called. Should take no parameters 
///                 and return a vector of \p NodePointer.
///
/// \return The return value of the provided function object, except it has 
///         host pointers instead of device pointers.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
std::vector<NodePointer<Node> > Voxelizer<Node, SNode>::simulateMultidevice
    ( 
        boost::function< std::vector< NodePointer<Node> >() > func
    )
{
    this->options.simulateMultidevice = true;
    return func();
}
///////////////////////////////////////////////////////////////////////////////
/// The voxelization produced by this function is a standard voxelization 
/// (ie. not using an FCC grid) that in addition includes a lot of data about 
/// the intersections between the triangles and the surface voxels. When a 
/// triangle intersects a voxel, the areas and volumes of the voxel are cut 
/// and they can be classified as being inside or outside of the model. These 
/// inner volumes and areas of the voxel's sides are available in each surface 
/// node. In order to handle this large amount of data, the surface voxels had 
/// to be split into separate voxel types. There are, as a result, now two 
/// arrays of nodes: one for the solid voxelization, and one for the surface 
/// voxels. The surface voxels are accessed by retrieving their index from a 
/// hash map, by using the index of a solid node as a key.
///
/// \throws Exception if CUDA reports an error.
///
/// \param[in] cubeLength Distance between voxel centers, or the length of 
///                       each side of a voxel.
/// \param[in] devConfig Device configuration for multi-device processing.
/// \param[in] voxSplitRes Maximum processable volume for the calculation of 
///                        the plain voxelization.
/// \param[in] matSplitRes Maximum processable volume for the materials
///                        calculations.
///
/// \return A vector of \p Node2APointer that contains both node arrays, the 
///         hash map and additional information needed to use them.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
std::vector<Node2APointer<Node, SNode> >
    Voxelizer<Node, SNode>::voxelizeToSurfaceNodes( 
        double cubeLength, 
        uint2 devConfig,
        uint3 voxSplitRes, 
        uint3 matSplitRes
    )
{
    std::vector<Node2APointer<Node, SNode> > result;

    this->options.nodeOutput = true;
    this->options.voxelDistanceGiven = true;
    this->hostVars.voxelLength = cubeLength;

    this->voxelizeEntry( devConfig, voxSplitRes, matSplitRes, NULL );

    result = this->collectSurfData();

    this->deallocate();

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// The voxelization produced by this function is a standard voxelization 
/// (ie. not using an FCC grid) that in addition includes a lot of data about 
/// the intersections between the triangles and the surface voxels. When a 
/// triangle intersects a voxel, the areas and volumes of the voxel are cut 
/// and they can be classified as being inside or outside of the model. These 
/// inner volumes and areas of the voxel's sides are available in each surface 
/// node. In order to handle this large amount of data, the surface voxels had 
/// to be split into separate voxel types. There are, as a result, now two 
/// arrays of nodes: one for the solid voxelization, and one for the surface 
/// voxels. The surface voxels are accessed by retrieving their index from a 
/// hash map, by using the index of a solid node as a key.
///
/// \throws Exception if CUDA reports an error.
///
/// \param[in] maxDimension How many voxels should there be along the longest
///                         side of the model.
/// \param[in] devConfig Device configuration for multi-device processing.
/// \param[in] voxSplitRes Maximum processable volume for the calculation of 
///                        the plain voxelization.
/// \param[in] matSplitRes Maximum processable volume for the materials
///                        calculations.
///
/// \return A vector of \p Node2APointer that contains both node arrays, the 
///         hash map and additional information needed to use them.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
std::vector<Node2APointer<Node, SNode> >
    Voxelizer<Node, SNode>::voxelizeToSurfaceNodes( 
        uint maxDimension, 
        uint2 devConfig,
        uint3 voxSplitRes, 
        uint3 matSplitRes
    )
{
    std::vector<Node2APointer<Node, SNode> > result;

    this->options.nodeOutput = true;

    this->setResolution( maxDimension );

    this->voxelizeEntry( devConfig, voxSplitRes, matSplitRes, NULL );

    result = this->collectSurfData();

    this->deallocate();

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// The voxelization produced by this function is a standard voxelization 
/// (ie. not using an FCC grid) that in addition includes a lot of data about 
/// the intersections between the triangles and the surface voxels. When a 
/// triangle intersects a voxel, the areas and volumes of the voxel are cut 
/// and they can be classified as being inside or outside of the model. These 
/// inner volumes and areas of the voxel's sides are available in each surface 
/// node. In order to handle this large amount of data, the surface voxels had 
/// to be split into separate voxel types. There are, as a result, now two 
/// arrays of nodes: one for the solid voxelization, and one for the surface 
/// voxels. The surface voxels are accessed by retrieving their index from a 
/// hash map, by using the index of a solid node as a key. This function 
/// returns pointers to host memory.
///
/// \throws Exception if CUDA reports an error.
///
/// \param[in] cubeLength Distance between voxel centers, or the length of 
///                       each side of a voxel.
/// \param[in] devConfig Device configuration for multi-device processing.
/// \param[in] voxSplitRes Maximum processable volume for the calculation of 
///                        the plain voxelization.
/// \param[in] matSplitRes Maximum processable volume for the materials
///                        calculations.
///
/// \return A \p Node2APointer that contains both node arrays, the hash map and 
///         additional information needed to use them.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
Node2APointer<Node, SNode>
    Voxelizer<Node, SNode>::voxelizeToSurfaceNodesToRAM( 
        double cubeLength, 
        uint3 voxSplitRes, 
        uint3 matSplitRes
    )
{
    Node2APointer<Node, SNode> result = Node2APointer<Node, SNode>();

    this->options.nodeOutput = true;
    this->options.voxelDistanceGiven = true;
    this->hostVars.voxelLength = cubeLength;

    this->voxelizeEntry( make_uint2( 1, 1 ), voxSplitRes, matSplitRes, NULL );

    result = this->collectSurfData( this->devices[0], true );

    this->deallocate();

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// The voxelization produced by this function is a standard voxelization 
/// (ie. not using an FCC grid) that in addition includes a lot of data about 
/// the intersections between the triangles and the surface voxels. When a 
/// triangle intersects a voxel, the areas and volumes of the voxel are cut 
/// and they can be classified as being inside or outside of the model. These 
/// inner volumes and areas of the voxel's sides are available in each surface 
/// node. In order to handle this large amount of data, the surface voxels had 
/// to be split into separate voxel types. There are, as a result, now two 
/// arrays of nodes: one for the solid voxelization, and one for the surface 
/// voxels. The surface voxels are accessed by retrieving their index from a 
/// hash map, by using the index of a solid node as a key. This function 
/// returns pointers to host memory.
///
/// \throws Exception if CUDA reports an error.
///
/// \param[in] maxDimension How many voxels should there be along the longest
///                         side of the model.
/// \param[in] devConfig Device configuration for multi-device processing.
/// \param[in] voxSplitRes Maximum processable volume for the calculation of 
///                        the plain voxelization.
/// \param[in] matSplitRes Maximum processable volume for the materials
///                        calculations.
///
/// \return A \p Node2APointer that contains both node arrays, the hash map and 
///         additional information needed to use them.
///////////////////////////////////////////////////////////////////////////////
template <class Node, class SNode>
Node2APointer<Node, SNode>
    Voxelizer<Node, SNode>::voxelizeToSurfaceNodesToRAM( 
        uint maxDimension, 
        uint3 voxSplitRes, 
        uint3 matSplitRes
    )
{
    Node2APointer<Node, SNode> result = Node2APointer<Node, SNode>();

    this->options.nodeOutput = true;
    this->setResolution( maxDimension );

    this->voxelizeEntry( make_uint2( 1, 1 ), voxSplitRes, matSplitRes, NULL );

    result = this->collectSurfData( this->devices[0], true );

    this->deallocate();

    return result;
}

// Template instantiations to force the compiler to compile them.
template class Voxelizer<ShortNode>;
template class Voxelizer<LongNode>;
template class Voxelizer<PartialNode>;
template class Voxelizer<ShortFCCNode>;
template class Voxelizer<LongFCCNode>;
template class Voxelizer<VolumeNode, SurfaceNode>;
template class Voxelizer<VolumeMapNode, SurfaceNode>;

} // End namespace vox