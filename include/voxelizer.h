#ifdef _WIN32
#pragma once
#endif

#ifndef VOXELIZER_H
#define VOXELIZER_H

///////////////////////////////////////////////////////////////////////////////
/// \mainpage Voxelizer documentation
///
/// \author Henrik Karlsson
/// \version 0.9
/// \date 28.8.2013
/// \copyright Copyrights here.
///
/// \tableofcontents
/// 
/// \section mainpage_intro Introduction
///
/// The program described in this document is a solid voxelizer that contains 
/// functionality specializing it to be used in acoustic modelling 
/// applications. 
///
/// The voxelization produced by the various modes of the program revolves 
/// around the concept of a voxel center. As long as the center of the voxel 
/// is inside of the volume defined by a thre dimensional mesh, the voxel is 
/// considered to be \a solid. This is the case for all voxelizations without 
/// material data attached to them. If material data is enabled, then the 
/// surface voxelizer will generate a shell of voxels that abide by a different 
/// rule: As long as a triangle even touches the volume of the voxel, the 
/// voxel is considered to be \a solid.
///
/// The output of most functions is a grid of \p Nodes, where each \p Node 
/// represents a single voxel along with whatever additional information the 
/// particular \p Node type happens to contain. Of particular interest to 
/// acoustics applications is the <em>boundary id</em> that can be found in all 
/// \p Node types. It encodes both the \a solid / \a non-solid status of the 
/// \p Node, as well as the neighborhood configuration, i.e. which neighbors 
/// are \a solid and which are \a non-solid. The configurations assume a 
/// right-handed coordinate system where the x-axis is pointing to the right, 
/// the y-axis is pointing towards the screen and the z-axis is pointing up.
/// \a Left and \a right refer then to the negative and positive directions 
/// along the x-axis, \a in and \a out refer to the negative and positive 
/// directions along the y-axis, and \a down and \a up likewise for the z-axis. 
/// The configurations are as follows: 
/// 
/// - The \p Node is non-solid.
///   - 00 = \p Node is non-solid and neighbors can be whatever.
/// 
/// - The \p Node and three neighboring \p Nodes are solid.
///   - 01 = Down, Left, In 
///   - 02 = Down, Right, In 
///   - 03 = Down, Left, Out 
///   - 04 = Down, Right, Out 
///   - 05 = Up, Left, In 
///   - 06 = Up, Right, In 
///   - 07 = Up, Left, Out 
///   - 08 = Up, Right, Out 
/// 
/// - The \p Node and four neighboring \p Nodes are solid.
///   - 09 = Down, Left, Right, In 
///   - 10 = Down, Left, Right, Out 
///   - 11 = Down, Left, In, Out 
///   - 12 = Down, Right, In, Out 
///   - 13 = Up, Left, Right, In 
///   - 14 = Up, Left, Right, Out 
///   - 15 = Up, Left, In, Out 
///   - 16 = Up, Right, In, Out 
///   - 17 = Up, Down, Left, In 
///   - 18 = Up, Down, Right, In 
///   - 19 = Up, Down, Left, Out 
///   - 20 = Up, Down, Right, Out 
/// 
/// - The \p Node and five neighboring \p Nodes are solid.
///   - 21 = Left, Right, In, Out, Down 
///   - 22 = Left, Right, Out, Down, Up 
///   - 23 = Left, Right, In, Down, Up 
///   - 24 = Right, In, Out, Down, Up 
///   - 25 = Left, In, Out, Down, Up 
///   - 26 = Left, Right, In, Out, Up 
/// 
/// - The \p Node and all six neighboring \p Nodes are solid.
///   - 27 = Left, Right, In, Out, Down, Up
///
/// \section mainpage_functionality Features and how to use them
/// 
/// The voxelizer can produce various kinds of output, each of which have their 
/// own special properties. This section is meant to serve as a guide to the 
/// successful use of the program, and will introduce its features and how to 
/// use them.
/// 
/// The entry point to the voxelizer is the \p Voxelizer class, defined in the 
/// vox namespace and found in the file voxelizer.h. It defines public member 
/// functions that entirely define the public interface of this library. It 
/// has to be instantiated before use, and if one plans to use any \p Node type 
/// other the plain voxelization or the \p ShortNode type, then the wanted \p 
/// Node type should be provided as a template parameter, like so:
///
/// \code
/// float * vertices = NULL;
/// unsigned int * indices = NULL;
/// unsigned int nrOfVertices = 0, nrOfTriangles = 0;
///
/// loadModel( "model.obj"
///          , &vertices
///          , &indices
///          , &nrOfVertices
///          , &nrOfIndices );
/// 
/// vox::Voxelizer<vox::LongNode> voxelizer( vertices
///                                        , indices
///                                        , nrOfVertices
///                                        , nrOfTriangles );
/// \endcode
///
/// A loadModel() function is not provided by the library -- it was only used 
/// to demonstrate how the various pieces of data tie in with the 
/// initialization of the voxelizer. As can be seen from the example, the 
/// model data is supplied upon creation of the voxelizer, and it cannot be 
/// changed later. Also, if the \p LongNode is omitted, then a \p ShortNode is 
/// assumed. It is the \p Node type with the smallest memory requirements.
///
/// Only one model format is supported: A combination of a vertex and index 
/// array. The single vertex consists of three consecutive floats: One for 
/// each spatial coordinate. An index in the index array consists of three 
/// consecutive unsigned integers: One for the each vertex of a triangle. The 
/// index corresponds to the vertice's index as they are ordered in the 
/// vertex array, but not directly to the data in the array, as there are 
/// three elements per one vertex there.
/// 
/// The voxelizer also supports material data in such a way that, if each 
/// triangle of the model has an associated material index with it, then this 
/// material index can be included in the \p Nodes of the voxelization. 
/// Materials can be supplied at any time and the inclusion of material 
/// information can also be turned on and off at any time after the voxelizer 
/// has been initialized. The appropriate functions are 
/// \p Voxelizer::setMaterials() and \p Voxelizer::setMaterialOutput(). The 
/// former uploads the material data and the latter enables or disables the 
/// material calculations.
///
/// The model and material data is copied upon construction of the voxelizer, 
/// so it can be safely deleted after the class has been initialized.
///
/// The results of the voxelization are given in a \p NodePointer struct. It 
/// contains a pointer to the \p Node or voxel data and how to interpret the 
/// array, i.e. the dimensions of the array in each direction. The functions 
/// that automatically upload the data to the computer's RAM return a single 
/// \p NodePointer, while those functions that directly return device pointers 
/// always return a \p std::vector of \p NodePointer. The \p vector is because 
/// of the possibility of voxelizing using multiple devices.
///
/// The \p Nodes and voxels are arranged in a specific way in their respective 
/// arrays. The first \p Node or voxel of an array corresponds to coordinates 
/// (0, 0, 0). The following ones will have increasing x-coordinates. Once 
/// the maximum x-coordinate has been reached, the following \p Node or voxel 
/// will have its y-coordinate increased by one, and so on. Here are some 
/// equations that can be used to convert to and from array indices and 
/// coordinates: \f{align*}{ i_{n} &= \text{dim}_{x} \cdot \text{dim}_{y} 
/// \cdot z_{n} + \text{dim}_{x} \cdot y_{n} + x_{n}, \\ i_{v} &= 
/// \frac{\text{dim}_{x}}{32} \cdot \text{dim}_{y} \cdot z_{v} + 
/// \frac{\text{dim}_{x}}{32} \cdot y_{v} + x_{v}, \\ x_{n} &= i_{n} 
/// \pmod{\text{dim}_{x}}, \\ y_{n} &= \frac{i_{n} \pmod{\text{dim}_{x} \cdot 
/// \text{dim}_{y}}}{\text{dim}_{x}}, \\ z_{n} &= \frac{i_{n}}{\text{dim}_{x} 
/// \cdot \text{dim}_{y}}, \\ x_{v} &= i_{v} \pmod{\frac{\text{dim}_{x}}{32}}, 
/// \\ y_{v} &= \frac{i_{v} \pmod{\frac{\text{dim}_{x}}{32} \cdot 
/// \text{dim}_{y}}}{\frac{\text{dim}_{x}}{32}}, \\ z_{v} &= 
/// \frac{i_{v}}{\frac{\text{dim}_{x}}{32} \cdot \text{dim}_{y}}. \f}
/// 
/// Each of the voxelizing functions allows the user to specify a number of 
/// parameters: The \a resolution, the <em>device configuration</em> and some 
/// restraints on how much of the voxelization is calculated at once. The 
/// resolution refers to how many voxels there should be along the longest side 
/// of the input model's bounding box. The number of voxels along the other 
/// sides are then calculated from that. The voxels are distributed so that 
/// the very ends of the longest side have one voxel each, and then the rest 
/// are distributed evenly in between. Each function also has the option to 
/// directly specify the distance between voxel centers along any of the main 
/// axes.
///
/// The device configuration is given with at most two values: The number of 
/// splits along the y- and z-axes. The idea is to divide the voxelization 
/// between the available devices by subdividing the volume and letting each 
/// device handle the voxelization within its own volume. If the y-axis is 
/// split into two, then that creates two volumes, where half of the y-axis is 
/// handled by one device, and the other half by another device. By combining 
/// multiple axes the number of subvolumes, or \a subspaces as they are called 
/// in the documentation, quickly increases. The only rule is that when 
/// multiplying the number of splits along each axis with each other, the 
/// result will equal the number of devices required for the voxelization. It 
/// is up to the user to find a proper configuration that works with the 
/// number of devices available.
///
/// The limitations on how the size of the volumes that the voxelizer processes 
/// parallelly can be changed through the \p voxSplitRes and \p voxMatRes 
/// arguments. Really large volumes allow for faster voxelization, but may 
/// cause the program to crash if the graphics card cannot handle the workload. 
/// The defaults are good for most purposes. If your graphics card crashes 
/// with the default values, you could try using a smaller space, such as 
/// (1024, 256, 256) for the \p voxSplitRes and (128, 128, 128) for the 
/// \p matSplitRes. The x-size of \p voxSplitRes should not make much of a 
/// difference on performance, so keeping it at 1024 is a safe choice.
///
/// There are currently five kinds of \p Nodes available: \p ShortNode, \p 
/// LongNode, \p PartialNode, \p ShortFCCNode and \p LongFCCNode. \p ShortNode 
/// stores its boundary and material information in a single \p char, 
/// restricting the number of different materials to 8. \p LongNode allocates 
/// two \p char values for its data, increasing the number of available 
/// materials to 256. \p PartialNode is a \p LongNode with an additional float 
/// value, called the \a ratio, that stores information about how large a 
/// fraction of the volume of the voxel is \a solid. The \p Short- and \p 
/// LongFCCNodes mirror their counterparts in the sense that the short version 
/// is a more compact and restricted version of the long version. The \p FCC 
/// \p Nodes voxelize on a different kind of lattice than the other \p Node 
/// types do. This lattice is called a \a Face \a Centered \a Cubice \a lattice
/// and more information about it can be found in node_types.h.
///
/// \p Voxelizer::voxelize() and \p Voxelizer::voxelizeToRAM() both produce a 
/// plain voxelization, and is the fastest voxelization to calculate. The 
/// former returns device pointers and the latter returns a host pointer.
///
/// \p Voxelizer::voxelizeToNodes() and \p Voxelizer::voxelizeToNodesToRAM() 
/// both produce an array of \p Nodes (The type of which is whatever the 
/// voxelizer was instantiated with) as their output. Material information can 
/// optionally be included in the \p Nodes if the material information has been 
/// made available through \p Voxelizer::setMaterials() and material output 
/// been enabled through \p Voxelizer::setMaterialOutput(). If the \p Node type 
/// is set to \p PartialNode, then the material calculations will also produce 
/// the partial volume information that is characteristic of that particular \p 
/// Node type. 
///
/// \p Voxelizer::voxelizeSlice() and \p Voxelizer::voxelizeSliceToRAM() both 
/// voxelize a thin slice of the total voxelization at a time. There are two 
/// more arguments that can be given: The \a direction along which the slicing 
/// will take place and \a which slice should be voxelized. The direction is 
/// given as an integer between 0 and 2. 0 represents the x-axis, 1 the y-axis 
/// and 2 the z-axis. The direction is essentially the normal of the slice, 
/// and successive slices will move in the given direction one voxel at a time. 
/// It is worth noting that slicing along the x-direction is somewhat more 
/// expensive both in terms of performance and memory. This is due to a 
/// limitation in the way the solid voxelization is calculated: The model has 
/// to be rotated and values copied from one array to another in order to be 
/// able to simulate voxelizing along the x-axis using one of the other 
/// directions. It is preferred that the user rotates the model manually and 
/// uses either the y- or z-directions for slicing, and then interprets the 
/// results accordingly. The slice to be calculated represents the coordinate 
/// on whatever axis was chosen as the direction. 
/// 
/// \section mainpage_implementation Implementation details
/// 
/// This section describes some technical details regarding the implementation 
/// of the voxelizer and should not be required reading if the goal is to 
/// simply use the program. For those who may want to modify the source code 
/// this section will give a general overview of how the different parts work 
/// both individually and together to produce the various available 
/// voxelizations.
///
/// The voxelization can be divided into two parts: The \a functions that are 
/// executed on the host side, that set up data structures and call the 
/// appropriate kernels, and the \a kernels that are executed on the graphics 
/// card.
///
/// One of the more confusing things is the division of the voxelization space 
/// into smaller spaces, both with and without multiple devices enabled. Before 
/// any divisions, there is the \a proper voxelization space, the dimensions of 
/// which are usually kept in \p HostContext.resolution. This space can be 
/// divided into one or more smaller spaces, depending on how many graphics 
/// cards are used. These spaces could be called \a device spaces, and their 
/// dimensions are kept in the appropriate \p DevContext.resolution, there 
/// being as many DevContext objects as there are graphics cards in use.
/// Finally, each of the device spaces can be further divided into multiple \a 
/// subspaces, to meet the restrictions on the size of the space that is 
/// voxelized at once, as given by \p voxSplitRes and \p matSplitRes when 
/// calling voxelizing functions. These subspaces are voxelized consecutively 
/// on the same device.
///
/// The part of the voxelization that calculates the <em>boundary ids</em> of a 
/// \p Node requires knowledge of its surrounding \p Nodes in order to know 
/// which of its neighbors are solid. This is easy to accomplish when the 
/// device space is divided into subspaces, as each subspace has access to the 
/// whole device space. When the proper space is divided into multiple device 
/// spaces, however, there is a problem accessing neighboring nodes along the 
/// seams of the splits, since the desired nodes reside on separate devices.
/// In order to gain access to \p Nodes across devices, the device spaces are 
/// extended along certain directions to make adjacent device spaces overlap 
/// each other by one voxel. Special \p bool values are used to determine which 
/// directions need overlap: \a Left, \a right, \a up and \a down. The 
/// directions refer to directions on the yz-plane, where left and right are 
/// the negative and positive directions, respectively, along the y-axis. Since 
/// the x-axis is never divided across devices, there is no need to consider 
/// it.
///
/// Something that ties into the concept of overlapping device spaces is the 
/// zero padding around each device space. The borders around each device 
/// space is automatically padded with zeroes before the pointers are returned 
/// to the user. In order to accommodate for the padding, the allocated array 
/// of voxels and nodes takes the padding into account from the very beginning.
/// With careful defining of where the voxelization begins and ends and where 
/// to write and not to write into the array one can re-use the space reserved 
/// for the padding for the overlapping voxels between device spaces. This 
/// involves adjusting coordinates by adding one to them depending on what 
/// adjacent device spaces there are. In the case of a one-device voxelization, 
/// the proper space is the same as the one device space, and thus all adjacent 
/// device spaces are false, and the only thing that needs to be done is to 
/// adjust all coordinates by one to "overcome" the initial padding. When 
/// voxelizing \p FCC \p Nodes, the padding becomes two voxels thick along the 
/// x-axis due to the way the voxelization is contructed.
///
/// The slicing algorithm uses the adjacent device space flags in order to 
/// force voxelization of adjacent slices, so that the \p Nodes of the relevant 
/// slice get proper adjacency information. This happens even when there is 
/// only one device in use.
///
/// \subsection mainpage_plain_vox Plain voxelization
/// 
/// The solid voxelization algorithm presented in this section is an 
/// implementation that closely follows the algorithm \cite schwarz-seidel-2010 
/// presented by Schwarz and Seidel.
/// 
/// \a Tiles are groups of 4 x 4 voxel centers projected onto the yz-plane. 
/// Most of the calculations during plain voxelization happen in the yz-plane, 
/// and it is not until the very end where things are extended into the full 
/// volume. <em>Edge functions</em> are functions that take a point in a plane 
/// as a parameter and return a value depending on which side of the edge the 
/// point lies on. In order to determine if a point is inside a triangle, when 
/// both lie in the same plane, the edge functions of each edge of the triangle 
/// are calculated and then evaluated against the point. The right combination 
/// of return values (all positive, for example, provided proper ordering) 
/// implies that the point has to reside within the triangle.
///
/// The very first thing that needs to be calculated during the plain 
/// voxelization is how many tiles each triangle overlaps with when projected 
/// onto the yz-plane. This is done by the \p calculateTileOverlap() function.
/// It uses the same technique for determining point/triangle overlap mentioned 
/// earlier, but with the points being the voxel centers of the tile. 
/// Parallellism is achieved through assigning one thread per triangle.
///
/// One goal is to produce a <em>work queue</em>, which lists all triangles 
/// that some tile overlaps with. Once we know how many tiles each triangle 
/// overlaps with, we can calculate both the size of the work queue and 
/// something that's called the <em>offset buffer</em>. The offset buffer 
/// basically contains offsets into the work queue for each triangle, i.e. it 
/// tells where the data for each triangle begins in the work queue. These two 
/// things are calculated by \p Voxelizer::prepareForConstructWorkQueue().
///
/// Once the size of the work queue is known, it can be allocated and filled.
/// \p constructWorkQueue() performs largely the same steps as \p 
/// calculateTileOverlap(), but fills the work queue with the tile ids of the 
/// tiles that intersect with the triangles.
/// 
/// Once the work queue is filled, it needs to be sorted by tile id, instead 
/// of by triangle id as it currently is. The \p sortWorkQueue() function 
/// handles this. It uses thrust to perform the sorting.
///
/// In order to be able to quickly distribute tile ids to different threads 
/// the work queue needs to be compacted. The idea is to remove duplicate 
/// tile ids and instead of referring to triangle ids, it could contain offsets 
/// to the work queue where that tile's data begins. \p compactWorkQueue()
/// calculates the compacted tile list using thrust.
///
/// Finally, the actual filling of the voxel grid (grid of integers) happens 
/// in \p generateVoxelization(). The parallellization is done by assigning a 
/// tile per warp and thus processing each voxel center in parallel. The steps 
/// taken are largely the same as in \p calculateTileOverlap() and 
/// \p constructWorkQueue(), but when overlap is confirmed, its exact location 
/// along the x-axis is calculated and the voxel it corresponds to, along with 
/// every voxel after that, has its bit flipped by a \p XOR operation.
/// 
/// \subsection mainpage_nodes Converting to Nodes.
/// 
/// There are two functions that deal with producing the \p Node 
/// representation: \p constructNodeList2() and \p fillNodeList2(). As one 
/// might expect from the naming, the former initializes the array of \p Nodes 
/// by translating each bit of the integer representation to a \p Node and the 
/// latter performs the boundary id calculations. Parallellism is achieved by 
/// assigning one thread per bit in \p constructNodeList2() and one thread per 
/// \p Node in \p fillNodeList2().
///
/// The boundary ids are calculated by encoding the state of a \p Nodes 
/// neighbors as bits in an <tt>unsigned int</tt>. The neighbor along the -x 
/// direction is 1, the one along +x is 2, the one along -y is 4, etc. There is 
/// then a function that translates this integer to a boundary id, which 
/// represents either a non-solid \p Node if the value is 0, or a valid, solid 
/// \p Node if the value is between 1 and 27. Many of the 64 possible 
/// permutations are invalid \p Nodes, that when encountered signify that 
/// the voxelization isn't quite well-formed. Invalid \p Nodes have their 
/// boundary ids set to 0. Due to this, the \p fillNodeList2() function needs 
/// to be run again to account for the changes in the \p Node array. A \p bool 
/// called the error bool will be set to \p true if the function needs to be 
/// run again.
///
/// \subsection mainpage_surface_vox Surface voxelization and materials.
///
/// The surface voxelizers presented in this section are implementations based 
/// on the algorithms \cite schwarz-seidel-2010 by Schwarz and Seidel.
///
/// The materials are calculated with a surface voxelizer, since it is 
/// difficult to accurately map triangles to individual voxels with a solid 
/// voxelizing algorithm. Unlike the solid voxelizer, the overlap testing takes 
/// into consideration the full volume of the voxel. This leads to a larger 
/// amount of processing per voxel, and explains why the surface voxelizer is 
/// heavier on the graphics card than the solid version. It is also the reason 
/// why \p voxSplitRes and \p matSplitRes are separate arguments. The surface 
/// voxelizer performs an especially large amount of work if it needs to 
/// calculate the solid volume of a voxel when using the \p PartialNode type.
///
/// It is worth noting that the surface voxelizer should be called after 
/// \p constructNodeList2() and before \p fillNodeList2(), due to the way they 
/// modify the \p Node array. constructNodeList2() sets sets all values of 
/// a \p Node to their defaults through the default constructor, and then sets 
/// the <em>boundary id</em> to either 0 or 1, depending on the state of the 
/// voxel it represents. Since the surface voxelizer may add additional solid 
/// \p Nodes due to the nature of its conservative voxelization, it should be 
/// called before \p fillNodeList2() calculates the boundary ids for each 
/// solid \p Node.
/// 
/// There are two versions of the surface voxelizer available: A simpler 
/// version and a more complicated version that has been augmented with various 
/// optimizations. The simpler version resides in a single kernel, 
/// \p SimpleSurfaceVoxelizer(). It is not meant to be used, since it doesn't 
/// contain many of the extra features, such as subspaces and device spaces, 
/// but it is a rather illustrative implementation that shows the basic 
/// concepts of the surface voxelizer pretty well. The more complicated version 
/// is divided into the following kernels: \p classifyTriangles(), 
/// \p process1DTriangles(), \p process2DTriangles() and \p 
/// process3DTriangles().
///
/// The basic idea of the surface voxelizer is to utilize a full box/triangle 
/// overlap test, which states that a triangle overlaps with an axis-aligned
/// box if:
/// - The triangle's plane intersects with the box.
/// - The triangle overlaps the box when projected into the xy-plane.
/// - The triangle overlaps the box when projected into the yz-plane.
/// - The triangle overlaps the box when projected into the zx-plane.
/// If all of the above conditions hold, then the triangle overlaps the box.
/// The simple surface voxelizer applies the above test to each and every 
/// voxel within a triangle's bounding box, which is rather inefficient.
///
/// The more complicated surface voxelizer first classifies each triangle 
/// according to three criteria: Type of bounding box, dominant axis of the 
/// normal and the number of voxel columns along the dominant axis the triangle 
/// spans. The type of bounding box is determined by the thickness of the 
/// bounding box in terms of how many voxels it overlaps -- A one-dimensional 
/// bounding box only covers one voxel or a number of voxels along one of the 
/// main axes -- a two-dimensional bounding box is one-voxel thick along one 
/// of the main axes, and a three-dimensional bounding box fits neither 
/// category. The dominant axis of the normal is simply the direction along 
/// which the normal's component has the largest magnitude. Finally, the 
/// number of voxel columns refers to the area in voxels of the side of the 
/// bounding box whose normal is parallel to the dominant axis. It basically 
/// expresses the upper limit on the number of voxel columns (that are 
/// parallel to the dominant axis) need to be processed when performing an 
/// overlap test in the plane, whose normal is also parallel to the dominant 
/// axis. The classification is performed by \p classifyTriangles(), which 
/// produces an array of <tt>unsigned int</tt>s, where the array index 
/// corresponds to a triangle id. The integers are aquired by encoding a 
/// \p TriData struct with the \p encodeTriangleType() function.
///
/// Once the classification array has been computed, it is sorted. This makes 
/// the 1D triangles appear first, the 2D triangles appear after that, etc. 
/// Thrust is used to both sort the array and to calculate information about 
/// the new triangle order, such as number of triangles of each type and at 
/// which indices each of the triangle types begin and end.
///
/// Once the boundaries between triangle types have been determined, a kernel 
/// that specializes in each triangle type is launched. These are 
/// \p process1DTriangles(), \p process2DTriangles() and \p 
/// process3DTriangles(). The optimizations in the specialized versions 
/// generally involve skipping overlap testing in one or more planes depending 
/// on the bounding box and dominant axis.
///
/// Once overlap between a voxel and a triangle has been found, the 
/// \p processVoxel() function is called. This function reads the appropriate 
/// material index of the triangle causing the overlap, and writes it to the 
/// \p Node. The \p Nodes boundary id is also set to 1.
///
/// \subsection mainpage_partial PartialNode and partial voxels.
///
/// If the \p Node type is set to \p PartialNode, then the processing that 
/// determines how much of the voxel is solid happens in the surface voxelizer.
/// More specifically, the \p processVoxel<PartialNode>() function is 
/// specialized for \p PartialNode, and it calls \p 
/// calculateVoxelPlaneIntersectionVolume() for the calculations.
///
/// First of all, the function does not actually calculate the exact quotient 
/// between \a solid and \a full volumes, but instead approximates the triangle 
/// to a plane, and only really takes into consideration one randomly chosen 
/// triangle of the potentially many triangles that may be intersecting with 
/// the voxel. The actual algorithm \cite timmes-2013 that the function relies 
/// on has been developed by Francis Xavier Timmes, whose effort is an 
/// extension of an algorithm \cite salama-kolb-2005 published by Salama and 
/// Kolb.
///
/// Conceptually, the algorithm calculates the intersection points between the 
/// plane and the voxel and then constructs the faces of the irregular 
/// polyhedron, the volume of which equals the volume that is solid. The volume 
/// is calculated by choosing a fixed point as the tip of each pyramid formed 
/// by each of the faces of the polyhedron, and then calculating and summing 
/// up the volumes of the pyramids using the surveyor's formula, also known 
/// as the shoelace formula \cite shoelace-2013 and Gauss' area formula, to 
/// calculate the area of the bases of the pyramids.
///
/// The algorithm involves defining paths from the voxel's corner vertex that 
/// is closest to the camera to the corner vertex that is furthest from the 
/// camera, when the cutting plane is parallel to the view plane. The paths 
/// follow the edges of the voxel. Three paths in total are defined, and these 
/// paths are constructed in such a way that they 1) don't share any edges and 
/// 2) the direction of the paths follow the basis vectors of a specific 
/// right-handed coordinate system. In order to find the intersection points 
/// between a plane and a box, it suffices to find the intersections between 
/// the plane and the edges of the box. By testing for intersections with the 
/// edges (of the paths) in a specific order, all intersection points of the 
/// polyhedron can be determined with minimal effort in addition to aquiring 
/// the intersection points in the correct order to ensure consistent winding 
/// of the faces.
///
/// \subsection mainpage_slicing Slicing
///
/// There are no slicing-exclusive functions that participate in the slicing.
/// Instead, already existing facilities are used wherever possible, with some 
/// code thrown in here and there to manage the unique properties of the 
/// slicing. The slices themselves are easily constructed by setting one of 
/// the dimensions to being one voxel thick. Now, add the padding and the slice 
/// becomes three voxels thick. This is all possible by letting the proper 
/// space be what you would normally have but defining a custom device space 
/// that is one voxel thick. If we set the \a left and \a right \p bool values 
/// to \p true when we're voxelizing a slice that moves along the y-direction, 
/// then the voxelizer thinks that there are device spaces to either side of 
/// the slice and will voxelize over the padding. This allows the \p 
/// fillNodeList2() kernel access to the relevant neighboring \p Nodes of the 
/// slice and produces a correct \p Node output.
///
/// Most of the additional codepaths when performing slicing have to do with 
/// either re-using allocated memory on the graphics card, or with the special 
/// arrangements that had to be made to make slicing along the x-axis work as 
/// the user would expect. The non-changing memory is allocated only once, as 
/// long as the direction doesn't change, and the reallocatable memory is kept 
/// as is between calls to the slicing function.
///
/// Slicing along the x-axis is pointless to implement directly by making the 
/// device space one voxel thick along the x-direction, because the whole 
/// voxelization process happens in the yz-plane, and so calculating one 
/// x-slice is essentially the same as calculating the whole voxelization -- 
/// minus the overhead of writing to memory. There is practically no 
/// performance improvement between slicing and doing the full voxelization.
/// Instead, the vertices are rotated 90 degrees around the z-axis, using the 
/// \p Voxelizer::rotateVertices() function, and the model is sliced along the 
/// y-direction instead. After the materials have been calculated, the \p Nodes 
/// are rotated in the opposite direction and copied into another array, using 
/// the \p unRotateNodes() kernel. Once the \p Nodes are in the same 
/// configuration they would have been in if the voxelization had truly been 
/// performed along the x-axis, the \p fillNodeList2() kernel is called to 
/// calculate the boundary ids.
///
/// \subsection mainpage_fcc FCC Nodes
///
/// If the \p Node type is chosen to be either \p ShortFCCNode or \p 
/// LongFCCNode, then the voxelization is going to be performed on a \p FCC 
/// (Face Centered Cubic) lattice. The (x, y, z)-coordinates of an individual 
/// \p Node can be determined identically to the ordinary voxelization, but 
/// the interpretation of these coordinates differs somewhat. node_types.h 
/// contains information about how the coordinates map to actual real world 
/// coordinates.
///
/// The FCC voxelization is produced by the same methods as the ordinary 
/// voxelization. There are no option booleans that need to be set to diverge 
/// execution to FCC-specific code -- This is handled by the static function 
/// \p isFCCNode() that each \p Node class has. \p Voxelizer::voxelizeEntry() 
/// is mostly unchanged, excepting certain allocations due to the differing 
/// sizes between normal grids and FCC grids. The FCC implementation has its 
/// own worker function, however, and it is called \p Voxelizer::fccWorker().
///
/// The FCC grid is constructed by merging four regular, rectilinear grids that 
/// are all slightly shifted in relation to each other. This translates quite 
/// easily to four separate voxelizations that are smaller in size than what 
/// the resulting FCC grid is going to be. No large changes need to be made 
/// to the program flow other than setting up the voxelizations, performing 
/// them and translating their data into the array of \p Nodes. Analogously, 
/// the materials are also calculated by performing four surface voxelizations 
/// which are then combined into the \p Node grid.
///
/// The function that translates a voxel grid to a \p Node grid is called 
/// \p convertToFCCGrid() and works more or less in the same way as \p
/// constructNodeLst2(), the only difference being that it has different cases 
/// for each of the four voxelizations.
///
/// The materials are handled the same way as with other \p Node types, but 
/// instead of always passing 1 as the voxel grid type by default, the 
/// voxel grids are assigned grid types from 1 to 4 and then the processVoxel() 
/// function assignes the data in the grids to the appropriate places in the 
/// \p Node grid, pretty much identically to how it's done in 
/// convertToFCCGrid().
///
/// The actual voxel grids are setup in the following way:
///
/// - Grid 1: \f$ \mathbf{vec{bbox}}_{min}^{new} = 
///   \mathbf{vec{bbox}}_{min}^{old} + \left( 0, 0, 0 \right) \f$.
/// - Grid 2: \f$ \mathbf{vec{bbox}}_{min}^{new} = 
///   \mathbf{vec{bbox}}_{min}^{old} + \left( \frac{d}{2}, \frac{d}{2}, 0 
///   \right) \f$.
/// - Grid 3: \f$ \mathbf{vec{bbox}}_{min}^{new} = 
///   \mathbf{vec{bbox}}_{min}^{old} + \left( 0, \frac{d}{2}, \frac{d}{2} 
///   \right) \f$.
/// - Grid 4: \f$ \mathbf{vec{bbox}}_{min}^{new} = 
///   \mathbf{vec{bbox}}_{min}^{old} + \left( \frac{d}{2}, 0, \frac{d}{2} 
///   \right) \f$.
///
/// The \f$ d \f$ refers to the distance between voxel centers along the main 
/// axes.
///
/// The translation from voxel coordinates to \p Node indices happens in 
/// the following way:
///
/// - Grid 1: \f$ A \cdot 2z + L \cdot y + 2x \f$.
/// - Grid 2: \f$ A \cdot 2z + L \cdot y + 2x + 1 \f$.
/// - Grid 3: \f$ A \cdot \left( 2z + 1 \right) + L \cdot y + 2x \f$.
/// - Grid 4: \f$ A \cdot \left( 2z + 1 \right) + L \cdot y + 2x + 1 \f$.
///
/// Where \f$ L = 2 \cdot dim_{x} \text{and} A = L \cdot dim_{y} \f$.
///
/// The function that calculates the boundary ids of the \p Nodes is called 
/// calculateFCCBoundaries(), and works in pretty much the same way as 
/// fillNodeList2(), except that it doesn't currently perform \p Node pruning 
/// of bad \p Nodes.
///
/// The zeroPadding() function has also been updated to take into account the 
/// fact that FCC Nodes have two coordinates worth of padding on boths ends 
/// of the grid along the x- and z-axes. The extra padding comes from the fact 
/// that the four voxelizations also have padding around them, and this carries 
/// over to the \p Node grid during the combining.
///////////////////////////////////////////////////////////////////////////////

#include "node_types.h"
#include "host_support.h"
#include "helper_math.h"

#include <iostream>
#include <ctime>
#include <fstream>
#include <vector>

#include <boost/function.hpp>

#include <boost/scoped_array.hpp>
#include <boost/shared_array.hpp>

///////////////////////////////////////////////////////////////////////////////
/// \brief Root namespace of the Voxelizer library.
///
/// Encapsulates everything that the Voxelizer library contains. For a starting 
/// point, refer to the Voxelizer class -- it contains the public interface 
/// for the librarys functions.
///////////////////////////////////////////////////////////////////////////////
namespace vox {

///////////////////////////////////////////////////////////////////////////////
/// \brief Result type of most voxelization operations. 
/// 
/// Functions that advertise to return device pointers return a vector of \p 
/// NodePointer structs, one for each device used in the voxelization. 
/// Functions that say that they return host pointers only return a single \p 
/// NodePointer, due to multidevice voxelization being disabled in that case.
/// The \p ptr and \p vptr variables both point to the same memory location.
/// The correct interpretation of the data depends on where it came from.
/// If multiple devices are in use, then the entire voxelization space is 
/// subdivided into multiple smaller spaces which are voxelized by each device.
/// The "coordinates" of each \a device \a space is given by the \p loc 
/// variable, and simply correspond to the x-, y- and z-coordinates of the \a 
/// device \a space in relation to the whole space.
/// 
/// \tparam Node Type of \p Node used in the result if it is interpreted as an 
///              array of \p Nodes. It is the same as the \p Node type the \p 
///              Voxelizer is initialized with.
///////////////////////////////////////////////////////////////////////////////
template <class Node>
struct NodePointer {
    /// Union that contains a pointer to either \p Node or voxel data.
    union {
        Node* ptr;      ///< Pointer to the \p Node data.
        VoxInt* vptr;   ///< Pointer to the \p voxel data.
    };
    int dev;            ///< Device id the data resides on, if applicable.
    uint3 dim;          ///< Dimensions of the \p Node grid.
    uint3 loc;          ///< Location of the \p Node grid.
};
///////////////////////////////////////////////////////////////////////////////
/// \brief Result type of voxelization operations with two arrays.
///
/// Analogous to \p NodePointer. This one contains two arrays of different 
/// \p Node types -- one for solid nodes and another for surface nodes. The 
/// surface nodes are accessed through a hash map, which maps solid node 
/// indices to surface node indices.
///
/// \tparam VolumeNode Type of node to use for the vast majority of nodes, ie. 
///                    the solid nodes.
/// \tparam SurfaceNode Type of node to use for the surface voxels.
///////////////////////////////////////////////////////////////////////////////
template <class VolumeNode, class SurfaceNode>
struct Node2APointer {
    VolumeNode * nodes;      ///< Array of all nodes in the voxelization.
    SurfaceNode * surfNodes; ///< Array of (large sized) surface nodes.
    HashMap indices;         ///< Maps coordinates to indices into surfNodes.

    int dev;            ///< Device id the data resides on, if applicable.
    uint3 dim;          ///< Dimensions of the \p Node grid.
    uint3 loc;          ///< Location of the \p Node grid.
    uint nrOfSurfNodes; ///< Size of the surfNodes array.
};

///////////////////////////////////////////////////////////////////////////////
/// \brief Class that encapsulates the voxelizer. 
/// 
/// It is initialized with the model it is supposed to voxelize, the type of \p 
/// Node to be used in the voxelization, and optionally with material 
/// information associated with the triangles in the model.
/// The supported \p Node types can be found in node_types.h. If performing a 
/// plain voxelization only, then the type of \p Node chosen does not matter.
/// Every voxelizing function has two versions that differ in the way the 
/// density of the voxelization is defined. The first way to define the density 
/// is to supply a number of voxels that the longest side of the models 
/// bounding box should have, starting and ending at the edges. The other way 
/// is to define a distance between voxels along one of the main axes. The 
/// number of voxels along each dimension is then automatically determined.
/// 
/// \tparam Node Type of \p Node to be used in the result. Applies to all 
///              functions that claim to return \p Nodes.
///////////////////////////////////////////////////////////////////////////////
template <class Node = ShortNode, class SNode = SurfaceNode>
class Voxelizer
{
public:
    /// Constructor without material information.
    Voxelizer( float const * _vertices
             , uint const * _indices
             , uint _nrOfVertices
             , uint _nrOfTriangles );
    /// Contructor with material information.
    Voxelizer( float const * _vertices
             , uint const * _indices
             , uchar const * _materials
             , uint _nrOfVertices
             , uint _nrOfTriangles
             , uint _nrOfUniqueMaterials );
    /// Default destructor.
    ~Voxelizer() {};

    /// Voxelize into an integer representation, returning device pointer(s).
    std::vector<NodePointer<Node> > voxelize( 
        uint maxDimension = 256, 
        uint2 devConfig = make_uint2( 1, 1 ), 
        uint3 voxSplitRes = make_uint3( 1024, 512, 512 ) );
    /// Voxelize into an integer representation, returning device pointer(s).
    std::vector<NodePointer<Node> > voxelize( 
        double cubeLength, 
        uint2 devConfig = make_uint2( 1, 1 ), 
        uint3 voxSplitRes = make_uint3( 1024, 512, 512 ) );
    /// Voxelize into an integer representation, returning a host pointer.
    NodePointer<Node> voxelizeToRAM( 
        uint  maxDimension = 256, 
        uint3 voxSplitRes = make_uint3( 1024, 512, 512 ) );
    /// Voxelize into an integer representation, returning a host pointer.
    NodePointer<Node> voxelizeToRAM( 
        double cubeLength, 
        uint3 voxSplitRes = make_uint3( 1024, 512, 512 ) );
    /// Voxelizes into a \p Node representation, returning device pointer(s).
    std::vector<NodePointer<Node> > 
        voxelizeToNodes( uint maxDimension = 256
                       , uint2 devConfig = make_uint2( 1, 1 )
                       , uint3 voxSplitRes = make_uint3( 1024, 512, 512 )
                       , uint3 matSplitRes = make_uint3( 512, 512, 512 ) );
    /// Voxelizes into a \p Node representation, returning device pointer(s).
    std::vector<NodePointer<Node> > 
        voxelizeToNodes( double cubeLength
                       , uint2 devConfig = make_uint2( 1, 1 )
                       , uint3 voxSplitRes = make_uint3( 1024, 512, 512 )
                       , uint3 matSplitRes = make_uint3( 512, 512, 512 ) );
    /// Voxelizes into a \p Node representation, returning a host pointer.
    NodePointer<Node> 
        voxelizeToNodesToRAM( 
            uint  maxDimension = 256,
            uint3 voxSplitRes = make_uint3( 1024, 512, 512 ),
            uint3 matSplitRes = make_uint3( 512, 512, 512 ) );
    /// Voxelizes into a \p Node representation, returning a host pointer.
    NodePointer<Node> 
        voxelizeToNodesToRAM( 
            double cubeLength,
            uint3 voxSplitRes = make_uint3( 1024, 512, 512 ),
            uint3 matSplitRes = make_uint3( 512, 512, 512 ) );
    /// Voxelize one 2D slice at a time, returning device pointers.
    std::vector<NodePointer<Node> > 
        voxelizeSlice( uint  maxDimension
                     , int   direction
                     , uint  slice
                     , uint  devConfig = 1
                     , uint2 voxSplitRes = make_uint2( 1024, 512 )
                     , uint2 matSplitRes = make_uint2( 512, 512 ) );
    /// Voxelize one 2D slice at a time, returning device pointers.
    std::vector<NodePointer<Node> > 
        voxelizeSlice( double cubeLength
                     , int   direction
                     , uint  slice
                     , uint  devConfig = 1
                     , uint2 voxSplitRes = make_uint2( 1024, 512 )
                     , uint2 matSplitRes = make_uint2( 512, 512 ) );
    /// Voxelize one 2D slice at a time, returning a host pointer.
    NodePointer<Node> 
        voxelizeSliceToRAM( uint  maxDimension
                          , int   direction
                          , uint  slice
                          , uint2 voxSplitRes = make_uint2( 1024, 512 )
                          , uint2 matSplitRes = make_uint2( 512, 512 ) );
    /// Voxelize one 2D slice at a time, returning a host pointer.
    NodePointer<Node> 
        voxelizeSliceToRAM( double cubeLength
                          , int   direction
                          , uint  slice
                          , uint2 voxSplitRes = make_uint2( 1024, 512 )
                          , uint2 matSplitRes = make_uint2( 512, 512 ) );
    /// Voxelizes with large surface nodes.
    std::vector<Node2APointer<Node, SNode> >
        voxelizeToSurfaceNodes( 
            double cubeLength, 
            uint2 devConfig = make_uint2( 1, 1 ),
            uint3 voxSplitRes = make_uint3( 1024, 512, 512 ), 
            uint3 matSplitRes = make_uint3( 512, 512, 512 )
        );
    /// Voxelizes with large surface nodes.
    std::vector<Node2APointer<Node, SNode> >
        voxelizeToSurfaceNodes( 
            uint maxDimension, 
            uint2 devConfig = make_uint2( 1, 1 ),
            uint3 voxSplitRes = make_uint3( 1024, 512, 512 ), 
            uint3 matSplitRes = make_uint3( 512, 512, 512 )
        );
    /// Voxelizes with large surface nodes, returning host pointers.
    Node2APointer<Node, SNode>
        voxelizeToSurfaceNodesToRAM( 
            double cubeLength, 
            uint3 voxSplitRes = make_uint3( 1024, 512, 512 ), 
            uint3 matSplitRes = make_uint3( 512, 512, 512 )
        );
    /// Voxelizes with large surface nodes, returning host pointers.
    Node2APointer<Node, SNode>
        voxelizeToSurfaceNodesToRAM( 
            uint maxDimension, 
            uint3 voxSplitRes = make_uint3( 1024, 512, 512 ), 
            uint3 matSplitRes = make_uint3( 512, 512, 512 )
        );

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the distance between voxel centers.
    ///
    /// \return Distance between voxel centers.
    ///////////////////////////////////////////////////////////////////////////
    double getVoxelLength() const throw() 
    { 
        return this->hostVars.voxelLength; 
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the minimum corner of the model's bounding box.
    ///
    /// \return Minimum corner of the model's bounding box.
    ///////////////////////////////////////////////////////////////////////////
    double3 getMinVertex() const throw() { return this->hostVars.minVertex; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the maximum corner of the model's bounding box.
    ///
    /// \return Maximum corner of the model's bounding box.
    ///////////////////////////////////////////////////////////////////////////
    double3 getMaxVertex() const throw() { return this->hostVars.maxVertex; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the dimensions of the allocated voxels or \p Nodes.
    ///
    /// \return Dimensions of the allocated voxel or \p Node array.
    ///////////////////////////////////////////////////////////////////////////
    uint3 getResolution() const throw() 
    { 
        return this->hostVars.resolution.max; 
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the vertices of the model.
    ///
    /// \return Vertices of the model.
    ///////////////////////////////////////////////////////////////////////////
    std::vector<float> getVertices() const throw() 
    { 
        return this->vertices; 
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the indices of the model.
    ///
    /// \return Indices of the model.
    ///////////////////////////////////////////////////////////////////////////
    std::vector<uint> getIndices() const throw() 
    { 
        return this->indices; 
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the number of triangles in the model.
    ///
    /// \return Number of triangles in the model.
    ///////////////////////////////////////////////////////////////////////////
    uint getNrOfTriangles() const throw() 
    { 
        return this->hostVars.nrOfTriangles; 
    }
    /// Define materials for the model.
    void setMaterials( uchar const * _materials, uint _nrOfUniqueMaterials );
    /// Enables or disables material output.
    void setMaterialOutput( bool _materials ) throw();
	/// Enables or disables orientations output.
    void setOrientationsOutput( bool _orientations  ) throw();
	/// Enables or disables center of voxel for comparison.
    void setDisplace_VoxSpace_dX_2( bool _displace_VoxSpace_dX_2  ) throw();
	/// Was the domain already displaced by dX_2...
    void setIs_displaced( bool _is_displaced  ) throw();
    /// Enables or disables messages to the standard output stream.
    void verboseOutput( bool verbose ) throw();
    /// Calculates how large the \p Node array is going to be.
    uint3 getArrayDimensions( uint longestSizeInVoxels
                            , uint maxInternalXSize
                            , bool sliceAlongX );
    /// Calculates how large the \p Node array is going to be.
    uint3 getArrayDimensions( double cubeLength
                            , uint maxInternalXSize
                            , bool sliceAlongX );
    /// Simulates multiple devices on the CPU and returns host pointers.
    std::vector<NodePointer<Node> > simulateMultidevice( 
        boost::function<std::vector<NodePointer<Node> >()> func );

private:
    /// Array of DevContexts.
    typedef boost::scoped_array<DevContext<Node,SNode> > Devices;

    // Variable declarations.

    const int defaultDevice;       ///< Default device id. Should be 0.
    int nrOfDevices;               ///< Number of devices on the system.
    int nrOfDevicesInUse;          ///< How many devices are being used.

    CommonHostData hostVars;       ///< Host variables.
    Devices devices;               ///< Device-specific vars.

    Options options;               ///< \p Options for the voxelization.

    std::ofstream log;             ///< Writes to file if printing is enabled.

    bool fatalError;               ///< Allows immediate termination.
    clock_t startTime;             ///< Time taken at start of voxelization.

    std::vector<float> vertices;   ///< Vertices of the model.
    std::vector<uint> indices;     ///< Triangle indices for the model.
    std::vector<uchar> materials;  ///< Triangle-material mappings.

    // Function declarations.

    /// Initializes variables upon contruction.
    void initVariables();
    /// Allocates device contexts and initializes variables for each device.
    void initDevices( uint nrOfUsedDevices );
    /// Clears all dynamically allocated memory related to the voxelization.
    void deallocate();
    /// Clears the dynamically allocated memory related to plain voxelization.
    void deallocateVoxelizationData( DevContext<Node,SNode> & device );
    /// Calculates the bounding box of the \p Node / voxel array.
    void determineBBAndResolution();
    /// Calculates the bounding box of the \p Node or voxel array.
    void calculateBoundingBox();
    /// Determines the size of the voxelization.
    void determineDimensions();
    /// Determines the size of the voxelization from a distance between voxels.
    void determineDimensions( double d );
    /// Makes the x-length divisible by 32 times the number of xSplits.
    void adjustResolution( uint xSplits );
    /// Calculates the <em>offset buffer</em>
    void prepareForConstructWorkQueue( DevContext<Node,SNode> & device );
    /// Splits a space into subspaces by constraining each direction.
    SplitData<uint3>
        splitResolutionByMaxDim( uint3 const & maxDimensions
                               , Bounds<uint3> const & resolution );
    /// Splits an area into subareas by constraining each direction.
    SplitData<uint2>
        splitResolutionByMaxDim( uint2 const & maxDimensions
                               , Bounds<uint3> const & resolution );
    /// Splits a line into sublines by constraining the lengths of each line.
    SplitData<uint>
        splitResolutionByMaxDim( uint maxDimension
                               , Bounds<uint3> const & resolution );
    /// Splits a space into similarly sized subspaces.
    boost::shared_array<Bounds<uint3> >
        splitResolutionByNrOfParts( uint3 const & nrOfPartitions
                                  , Bounds<uint3> const & resolution );
    /// Splits an area into similarly sized subareas.
    boost::shared_array<Bounds<uint2> >
        splitResolutionByNrOfParts( uint2 const & nrOfPartitions
                                  , Bounds<uint3> const & resolution );
    /// Splits a line into a given number of similarly sized lines.
    boost::shared_array<Bounds<uint> >
        splitResolutionByNrOfParts( uint nrOfPartitions
                                  , Bounds<uint3> const & resolution );
    /// Performs a plain voxelization with the given settings.
    void performVoxelization( Bounds<uint2> yzSubSpace
                            , uint			     xRes
                            , uint			     nrOfXSlices
                            , DevContext<Node,SNode> & device );
    /// Opens an output \p filestream.
    void openLog( char const * filename );
    /// Writes verious information about the voxelization settings to file.
    void printGeneralInfo( DevContext<Node,SNode> & device );
    /// Writes the <em>tile overlap array</em> contents to file.
    void printTileOverlaps( DevContext<Node,SNode> & device, MainAxis direction );
    /// Writes the <em>offset buffer</em> contents to file.
    void printOffsetBuffer( DevContext<Node,SNode> & device, MainAxis direction );
    /// Prints the entire <em>work queue</em> contents to file.
    void printWorkQueue( DevContext<Node,SNode> & device, MainAxis direction );
    /// Prints the entire <em>sorted work queue</em> contents to file.
    void printSortedWorkQueue( DevContext<Node,SNode> & device, MainAxis direction );
    /// Prints the <em>compacted tile list</em> contents to file.
    void printCompactedList( DevContext<Node,SNode> & device, MainAxis direction );
    /// Closes the output \p filestream.
    void closeLog();
    /// Helper function to determine some stats about the graphics card.
    inline int convertSMVer2Cores(int major, int minor) const;
    /// Allocates an always fixed amount of memory.
    void allocStaticMem( DevContext<Node,SNode> & device );
    /// Reallocates memory when more is needed.
    void reAllocDynamicMem( DevContext<Node,SNode> & device, float multiplier);
    /// Entry function for every kind of voxelization.
    void voxelizeEntry( uint2 deviceConfig = make_uint2( 1, 1 ), 
                        uint3 voxSplitRes = make_uint3( 1024, 512, 512 ), 
                        uint3 matSplitRes = make_uint3( 512, 512, 512 ),
                        char const * filename = NULL );
    /// The actual voxelization happens here.
    void voxelizeWorker( uint  xRes, 
                         uint  xSplits, 
                         uint3 voxSplitRes,
                         uint3 matSplitRes,
                         DevContext<Node,SNode> & device );
    /// Specialized worker function for voxelizations with two arrays.
    void twoNodeArraysWorker( uint  xRes
                            , uint  xSplits
                            , uint3 voxSplitRes
                            , uint3 matSplitRes
                            , DevContext<Node,SNode> & device );
    /// Specialized worker function for FCC-related \p Nodes.
    void fccWorker( uint  xRes
                  , uint  xSplits
                  , uint3 voxSplitRes
                  , uint3 matSplitRes
                  , DevContext<Node,SNode> & device );
    /// Prepares the voxelizer with a new max array length.
    void setResolution( uint resolution ) { 
        this->hostVars.resolution.min = make_uint3( 0, 0, 0 ); 
        this->hostVars.resolution.max = make_uint3(resolution, 0, 0); 
    }
    /// Rotates the vertices 90 degrees around the z-axis.
    void rotateVertices();
    /// Restores the coordinates of a vector to their un-rotated form.
    uint3 unRotateCoords( uint3 vec, uint xDim );
    /// Resets certain data structures to their default values.
    void resetDataStructures( DevContext<Node,SNode> & device );
    /// Constructs the <tt>vector<NodePointer></tt> returnable.
    std::vector<NodePointer<Node> > collectData();
    /// Constructs the \p NodePointer returnable.
    NodePointer<Node> collectData( DevContext<Node,SNode> & device
                                 , const bool hostPointers );
    /// Constructs the <tt>vector<Node2APointer></tt> returnable.
    std::vector<Node2APointer<Node, SNode> > collectSurfData();
    /// Constructs the \p Node2APointer returnable.
    Node2APointer<Node, SNode> collectSurfData( DevContext<Node,SNode> & device
                                              , const bool         hostPointers
                                              );
};

///////////////////////////////////////////////////////////////////////////////
/// Traverses each vertex of the triangle and finds the minimum and maximum
/// coordinate components and uses them to construct the minimum and maximum
/// corners of the bounding box.
/// This version uses \p float3.
///////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ void getTriangleBounds
    ( float3 const * vertices ///< [in] Vertices of the triangle.
    , Bounds<float3> & bounds ///< [out] Bounding box of the triangle.
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
/// Expects the triangle's bounding box to be made of \p float3 and returns a
/// bounding box made of \p uint3.
///////////////////////////////////////////////////////////////////////////////

// 		!! This one is inside device_code.h !!
inline __host__ __device__ void getVoxelBounds
    ( Bounds<float3> const & triBB   ///< [in] Triangle's bounding box.
    , float3 const & modelBBMin      /**< [in] Minimum corner of the device's
                                               voxelization space. */
    , Bounds<uint3> & voxBB          /**< [out] Triangle's bounding
                                                box in voxel coordinates. */
    , float d                        ///< [in] Distance between voxel centers.
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

inline __host__ __device__ void getVoxelBounds
    ( Bounds<double3> const & triBB   ///< [in] Triangle's bounding box.
    , double3 const & modelBBMin      /**< [in] Minimum corner of the device's
                                               voxelization space. */
    , Bounds<uint3> & voxBB          /**< [out] Triangle's bounding
                                                box in voxel coordinates. */
    , float d                        ///< [in] Distance between voxel centers.
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

} // End namespace vox

#endif
