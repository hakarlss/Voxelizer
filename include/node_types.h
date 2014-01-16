#ifdef _WIN32
#pragma once
#endif

#ifndef VOX_NODE_TYPES_H
#define VOX_NODE_TYPES_H

#include "common.h"

namespace vox {

///////////////////////////////////////////////////////////////////////////////
/// \brief A \p Node type that only takes 1 byte of memory. 
///
/// Stores the <em>boundary id</em> in the first 5 bits, and the 
/// <em>material id</em> in the final 3 bits. This restricts the number of 
/// maximum different materials to 8. 
///////////////////////////////////////////////////////////////////////////////
class ShortNode
{
public:
    /// Default constructor.
    __host__ __device__ ShortNode(): bim(0) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructs \p ShortNode given a \p bim.
    /// 
    /// \param[in] _bim Combined <em>boundary id</em> and <em>material id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ ShortNode(uchar _bim): bim(_bim) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructs \p ShortNode given both a <em>material id</em> and 
    /// <em>boundary id</em>.
    /// 
    /// \param[in] _mat <em>Material id</em>.
    /// \param[in] _bid <em>Boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ 
    ShortNode( uchar _bid, uchar _mat ) { set( _mat, _bid ); }

    __host__ __device__
    ShortNode( uchar _bid, uchar _mat, float _r ) { set( _mat, _bid); }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Extracts the <em>material id</em> from the \p bim.
    /// 
    /// \return The <em>material id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ uchar mat() const { return bim >> ShortNode::SHIFT; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Extracts the <em>boundary id</em> from the \p bim.
    /// 
    /// \return The <em>boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ uchar bid() const { return bim & ShortNode::MASK; } 
    ///////////////////////////////////////////////////////////////////////////
    /// \brief \p ShortNode has no \p r value.
    /// 
    /// \return 0.0f
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ float r() const { return 0.0f; } 
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Sets the <em>material id</em>. 
    ///
    /// <em>Material id</em>s exceeding the allowed limit produces undefined 
    /// behavior.
    /// 
    /// \param[in] mat The <em>material id</em> to be inserted into the \p bim.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void mat(uchar mat)
    {
        bim = (mat << ShortNode::SHIFT) | (bim & ShortNode::MASK);
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Sets the <em>boundary id</em>. 
    ///
    /// <em>Boundary id</em>s exceeding the allowed limit produces undefined 
    /// behavior.
    /// 
    /// \param[in] bid The <em>boundary id</em> to be inserted into the \p bid.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void bid(uchar bid)
    {
        bim = (bid & ShortNode::MASK) | (bim & ShortNode::MASK_INV); 
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Setter method for the \a ratio. 
    ///
    /// \p ShortNode has no \a ratio so does nothing.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void r(float ratio) { } // Do nothing.
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Sets both the <em>boundary id</em> and the <em>material id</em> 
    /// at the same time. 
    ///
    /// Values exceeding their specified limit produce undefined behavior. 
    /// Setting both values at once is potentially more efficient than setting 
    /// them independently.
    /// 
    /// \param[in] mat <em>Material id</em>.
    /// \param[in] bid <em>Boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void set(uchar mat, uchar bid)
    { 
        bim = (mat << ShortNode::SHIFT) | (bid & ShortNode::MASK); 
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the maximum supported <em>material index</em> for this 
    ///        \p Node type.
    /// 
    /// \return 7.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ uchar maxMat() { return 7; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the maximum supported <em>boundary index</em> for this 
    ///        \p Node type.
    /// 
    /// \return 31.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ uchar maxBid() { return 31; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns existence of a fractional value. Only \p PartialNode has 
    ///        it.
    /// 
    /// \return false.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool hasRatio() { return false; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns \p true if the \p Node is meant to be regarded as a 
    ///        point on an FCC lattice.
    /// 
    /// \return \p false.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool isFCCNode() { return false; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns \p true if the \p Node is meant to be used in 
    ///        conjunction with another \p Node type for surface nodes.
    ///
    /// \return \p false.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool usesTwoArrays() { return false; }
private:
    uchar bim; ///< Combined \a boundary and <em>material ids</em>.

    
    /// \brief Bitmask that represents the <em>boundary id</em>. 
    ///
    /// It masks out the <em>material id</em>, so <tt>OR</tt>'ing the \p bim 
    /// with this mask yields the <em>boundary id</em>.
    ///
    static const uchar MASK = 31;
    /// Inverse of the \p MASK. Represents the <em>material id</em> instead.
    static const uchar MASK_INV = 224;
    /// \brief The number of right shifts required in order to place the 
    ///        <em>material id</em> at the beginning of the bit field.
    ///
    static const uchar SHIFT = 5;
};
///////////////////////////////////////////////////////////////////////////////
/// \brief A \p Node type that takes 2 bytes of memory. 
///
/// In this node, the <em>material id</em> and the <em>boundary id</em> are in 
/// separate chars, allowing for at most 256 unique values per id type.
///////////////////////////////////////////////////////////////////////////////
class LongNode
{
public:
    /// Default constructor.
    __host__ __device__ LongNode(): _bid(0), _mat(0) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructor that takes a <em>boundary id</em>. 
    ///
    /// Sets the <em>material id</em> to zero.
    /// 
    /// \param[in] bid <em>Boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ LongNode(uchar bid): _bid(bid), _mat(0) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructor that takes both a <em>boundary id</em> and a 
    /// <em>material id</em>.
    /// 
    /// \param[in] bid <em>Boundary id</em>.
    /// \param[in] mat <em>Material id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ LongNode(uchar bid, uchar mat): _bid(bid), _mat(mat) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructor that takes three parameters.
    ///
    /// \param[in] bid Boundary id.
    /// \param[in] mat Material id.
    /// \param[in] r Ratio. Has no effect.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ 
    LongNode(uchar bid, uchar mat, float r): _bid(bid), _mat(mat) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Getter method for the <em>material id</em>.
    /// 
    /// \return <em>Material id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ uchar mat() const { return _mat; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Getter method for the <em>boundary id</em>.
    /// 
    /// \return <em>Boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ uchar bid() const { return _bid; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Getter method for the \a ratio. 
    ///
    /// \p LongNode has no \a ratio so always return 0.
    /// 
    /// \return Always returns 0.0f.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ float r() const { return 0.0f; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Setter method for the <em>material id</em>. 
    /// 
    /// \param[in] mat New material id.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void mat(uchar mat) { _mat = mat; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Setter method for the <em>boundary id</em>.
    /// 
    /// \param[in] bid New boundary id.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void bid(uchar bid) { _bid = bid; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Setter method for the \a ratio. 
    /// Does nothing since \p LongNode has no \a ratio.
    /// 
    /// \param[in] ratio New \a ratio.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void r(float ratio) { }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Sets both the \a material and <em>boundary ids</em> at once.
    /// 
    /// \param[in] mat <em>New material id</em>.
    /// \param[in] bid <em>New boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ 
    void set(uchar mat, uchar bid) { _mat = mat; _bid = bid; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Gets the maximum supported <em>material id</em>.
    /// 
    /// \return 255.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ uchar maxMat() { return UCHAR_MAX; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Gets the maximum supported <em>boundary id</em>.
    /// 
    /// \return 255.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ uchar maxBid() { return UCHAR_MAX; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Asks if \p LongNode has a \a ratio. 
    /// 
    /// Since it doesn't, always returns \p false.
    /// 
    /// \return Always returns \p false.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool hasRatio() { return false; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns \p true if the \p Node is meant to be regarded as a 
    ///        point on an FCC lattice.
    /// 
    /// \return \p false.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool isFCCNode() { return false; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns \p true if the \p Node is meant to be used in 
    ///        conjunction with another \p Node type for surface nodes.
    ///
    /// \return \p false.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool usesTwoArrays() { return false; }
private:
    uchar _bid;     ///< <em>Boundary id</em>.
    uchar _mat;     ///< <em>Material id</em>.
};
///////////////////////////////////////////////////////////////////////////////
/// \brief A \p Node type that, in addition to the standard \a boundary and 
///        <em>material ids</em>, stores information about what fraction of it 
///        is solid, called its \a ratio. 
/// 
/// Takes 8 bytes of memory due to alignment issues with the two \p chars and 
/// the \p float. The \a ratio is defined as 
/// \f$r = \frac{V_{solid}}{V_{total}}\f$. The calculation of the fraction is 
/// performed as part of the material calculations, and is very demanding, so 
/// it is recommended to set the \p matSplitRes to something low like 
/// <tt>(256, 256, 256)</tt> so that the GPU doesn't choke on the workload.
///////////////////////////////////////////////////////////////////////////////
class PartialNode
{
public:
    /// Default constructor. Initializes all member variables to 0.
    __host__ __device__ PartialNode(): _bid(0), _mat(0), _r(0.0f) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Contructor that takes a <em>boundary id</em>. 
    ///
    /// Initializes the rest of the member variables to 0.
    /// 
    /// \param[in] bid <em>Boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ PartialNode(uchar bid): _bid(bid), _mat(0), _r(0.0f) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructor that takes both a \a boundary and <em>material 
    /// id</em>. 
    ///
    /// Sets the \a ratio to 0.
    /// 
    /// \param[in] bid <em>Boundary id</em>.
    /// \param[in] mat <em>Material id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ 
    PartialNode(uchar bid, uchar mat): _bid(bid), _mat(mat) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructor that takes a \a boundary and <em>material id</em>, 
    ///        as well as a \a ratio.
    /// 
    /// \param[in] bid <em>Boundary id</em>.
    /// \param[in] mat <em>Material id</em>.
    /// \param[in] ratio \a Ratio.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ PartialNode( uchar bid
                                   , uchar mat
                                   , float ratio ): _bid( bid )
                                                  , _mat( mat )
                                                  , _r( ratio ) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Getter method for the <em>material id</em>.
    /// 
    /// \return <em>Material id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ uchar mat() const { return _mat; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Getter method for the <em>Boundary id</em>.
    /// 
    /// \return <em>Boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ uchar bid() const { return _bid; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Getter method for the \a Ratio.
    /// 
    /// \return \a Ratio.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ float r() const { return _r; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Setter method for the <em>material id</em>.
    /// 
    /// \param[in] mat New <em>material id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void mat(uchar mat) { _mat = mat; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Setter method for the <em>boundary id</em>.
    /// 
    /// \param[in] bid New <em>boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void bid(uchar bid) { _bid = bid; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Setter method for the \a ratio.
    /// 
    /// \param[in] ratio New \a ratio.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void r(float ratio) { _r = ratio; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Setter method for both the \a material and <em>boundary ids</em>.
    /// 
    /// \param[in] mat New <em>material id</em>.
    /// \param[in] bid New <em>boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ 
    void set(uchar mat, uchar bid) { _mat = mat; _bid = bid; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Setter method for the \a material and <em>boundary id</em>s and 
    ///        the \a ratio.
    /// 
    /// \param[in] mat New <em>material id</em>.
    /// \param[in] bid New <em>boundary id</em>.
    /// \param[in] ratio New \a ratio.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void set( uchar mat
                                , uchar bid
                                , float ratio )
    { 
        _mat = mat; 
        _bid = bid; 
        _r = ratio; 
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Gets the largest allowable <em>material id</em>. 
    /// 
    /// Defined as the maximum representable value of an <tt>unsigned char</tt>.
    /// 
    /// \return 255.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ uchar maxMat() { return UCHAR_MAX; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Gets the largest allowable <em>boundary id</em>. 
    ///
    /// Defined as the maximum representable value of an <tt>unsigned 
    /// char</tt>.
    /// 
    /// \return 255.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ uchar maxBid() { return UCHAR_MAX; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns whether or not \p PartialNode has a \a ratio. 
    /// 
    /// Always returns \p true.
    /// 
    /// \return Always returns \p true.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool hasRatio() { return true; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns \p true if the \p Node is meant to be regarded as a 
    ///        point on an FCC lattice.
    /// 
    /// \return \p false.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool isFCCNode() { return false; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns \p true if the \p Node is meant to be used in 
    ///        conjunction with another \p Node type for surface nodes.
    ///
    /// \return \p false.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool usesTwoArrays() { return false; }
private:
    uchar _bid;     ///< <em>Boundary id</em>.
    uchar _mat;     ///< <em>Material id</em>.
    float _r;       ///< \a Ratio of how much of the voxel is solid.
};
///////////////////////////////////////////////////////////////////////////////
/// \brief A \p Node type that takes 2 bytes of memory and represents a \p Node 
///        in a FCC (face centered cubic) lattice.
///
/// Stores the <em>boundary id</em> in the first 12 bits, and the 
/// <em>material id</em> in the final 4 bits. This restricts the number of 
/// maximum different materials to 16. The <em>boundary id</em> differs from 
/// the <em>boundary id</em> in other \p Nodes by directly encoding the solid 
/// status of its neighbors instead of filtering out bad combinations.
///
/// The bits correspond to neighboring \p Nodes in the following way:
/// 
/// \code
/// 15 14 13 12 | 11 10 09 08 07 06 05 04 03 02 01 00
/// -------------------------------------------------
/// Materials   | Neighbors
///
///      (11)
/// (09)      (10) Upper layer
///      (08)
/// ---------------
/// (06)      (07)
///      (__)      Middle layer, (__) is the current Node.
/// (04)      (05)
/// ---------------
///      (03)
/// (01)      (02) Lower layer
///      (00)
/// \endcode
///
/// Using FCC nodes produces an array that has a slightly different indexing 
/// setup than normal nodes have. This indexing setup produces the following 
/// lattice:
///
/// \f[ G_{n} = \left\{ \mathbf{\vec{x}_{m,a}} = \left[ \mathbf{\vec{m}} + 
/// m_{y}\mathbf{\hat{e}_{y}} + \left( \left( m_{x} + m_{z} \pmod 2 \right)
/// \mathbf{\hat{e}_{y}} \right) \right] \cdot \frac{a}{2} \right\}, \f]
///
/// where \f$ \mathbf{\vec{x}_{m,a}} \f$ is the position in space of a point on 
/// the fcc lattice, \f$ \mathbf{\vec{m}} = m_{x}\mathbf{\hat{e}_{x}} + 
/// m_{y}\mathbf{\hat{e}_{y}} + m_{z}\mathbf{\hat{e}_{z}} \f$ are the lattice 
/// coordinates (used by the array) and \f$ a \f$ is the side length of the 
/// cubic cell, as well as the distance between two sample points in the 
/// individual voxelizations.
///////////////////////////////////////////////////////////////////////////////
class ShortFCCNode
{
public:
    /// Default constructor.
    __host__ __device__ ShortFCCNode(): bim(0) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructs \p ShortFCCNode given a \p bid.
    /// 
    /// \param[in] bim Combined <em>boundary id</em> and <em>material id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ ShortFCCNode(ushort bim): bim(bim) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructs \p ShortFCCNode given both a <em>material id</em> and 
    /// <em>boundary id</em>.
    /// 
    /// \param[in] mat <em>Material id</em>.
    /// \param[in] bid <em>Boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ 
    ShortFCCNode( ushort bid, ushort mat ) { set( mat, bid ); }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructs \p ShortFCCNode given both a <em>material id</em> and 
    /// <em>boundary id</em> and ratio.
    /// 
    /// \param[in] mat <em>Material id</em>.
    /// \param[in] bid <em>Boundary id</em>.
    /// \param[in] r Ratio. Has no effect.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ 
    ShortFCCNode( ushort bid, ushort mat, float r ) { set( mat, bid ); }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Extracts the <em>material id</em> from the \p bim.
    /// 
    /// \return The <em>material id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ ushort mat() const 
    {
        return bim >> ShortFCCNode::SHIFT; 
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Extracts the <em>boundary id</em> from the \p bim.
    /// 
    /// \return The <em>boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ ushort bid() const { return bim & ShortFCCNode::MASK; } 
    ///////////////////////////////////////////////////////////////////////////
    /// \brief \p ShortFCCNode has no \p r value.
    /// 
    /// \return 0.0f
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ float r() const { return 0.0f; } 
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Sets the <em>material id</em>. 
    ///
    /// <em>Material id</em>s exceeding the allowed limit produces undefined 
    /// behavior.
    /// 
    /// \param[in] mat The <em>material id</em> to be inserted into the \p bim.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void mat(ushort mat)
    {
        bim = (mat << ShortFCCNode::SHIFT) | (bim & ShortFCCNode::MASK);
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Sets the <em>boundary id</em>. 
    ///
    /// <em>Boundary id</em>s exceeding the allowed limit produces undefined 
    /// behavior.
    /// 
    /// \param[in] bid The <em>boundary id</em> to be inserted into the \p bim.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void bid(ushort bid)
    {
        bim = (bid & ShortFCCNode::MASK) | (bim & ShortFCCNode::MASK_INV); 
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Setter method for the \a ratio. 
    ///
    /// \p ShortFCCNode has no \a ratio so does nothing.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void r(float ratio) { } // Do nothing.
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Sets both the <em>boundary id</em> and the <em>material id</em> 
    /// at the same time. 
    ///
    /// Values exceeding their specified limit produce undefined behavior. 
    /// Setting both values at once is potentially more efficient than setting 
    /// them independently.
    /// 
    /// \param[in] mat <em>Material id</em>.
    /// \param[in] bid <em>Boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void set(ushort mat, ushort bid)
    { 
        bim = (mat << ShortFCCNode::SHIFT) | (bid & ShortFCCNode::MASK); 
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the maximum supported <em>material index</em> for this 
    ///        \p Node type.
    /// 
    /// \return 15.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ ushort maxMat() { return 15; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns existence of a fractional value. Only \p PartialNode has 
    /// it.
    /// 
    /// \return false.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool hasRatio() { return false; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns \p true if the \p Node is meant to be regarded as a 
    ///        point on an FCC lattice.
    /// 
    /// \return \p true.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool isFCCNode() { return true; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns \p true if the \p Node is meant to be used in 
    ///        conjunction with another \p Node type for surface nodes.
    ///
    /// \return \p false.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool usesTwoArrays() { return false; }
private:
    ushort bim;     ///< Combined boundary and material id.

    static const ushort MASK = 4095;        ///< Bitmask that gives the bid.
    static const ushort MASK_INV = 61440;   ///< Bitmask that gives the mat.
    static const ushort SHIFT = 12;         ///< Steps to rshift for the mat.
};

///////////////////////////////////////////////////////////////////////////////
/// \brief A \p Node type that takes 4 bytes of memory and represents a \p Node 
///        in an FCC (face centered cubic) lattice.
///
/// Stores the <em>boundary id</em> in the first 16 bits, and the 
/// <em>material id</em> in the final 16 bits. The <em>boundary id</em> differs 
/// from the <em>boundary id</em> in other \p Nodes by directly encoding the 
/// solid status of its neighbors instead of filtering out bad combinations.
///
/// The bits correspond to neighboring \p Nodes in the following way:
/// 
/// \code
///  .. 12 | 11 10 09 08 07 06 05 04 03 02 01 00
/// -------------------------------------------------
/// Unused | Neighbors
///
///      (11)
/// (09)      (10) Upper layer
///      (08)
/// ---------------
/// (06)      (07)
///      (__)      Middle layer, (__) is the current Node.
/// (04)      (05)
/// ---------------
///      (03)
/// (01)      (02) Lower layer
///      (00)
/// \endcode
///
/// Using FCC nodes produces an array that has a slightly different indexing 
/// setup than normal nodes have. This indexing setup produces the following 
/// lattice:
///
/// \f[ G_{n} = \left\{ \mathbf{\vec{x}_{m,a}} = \left[ \mathbf{\vec{m}} + 
/// m_{y}\mathbf{\hat{e}_{y}} + \left( \left( m_{x} + m_{z} \pmod 2 \right)
/// \mathbf{\hat{e}_{y}} \right) \right] \cdot \frac{a}{2} \right\}, \f]
///
/// where \f$ \mathbf{\vec{x}_{m,a}} \f$ is the position in space of a point on 
/// the fcc lattice, \f$ \mathbf{\vec{m}} = m_{x}\mathbf{\hat{e}_{x}} + 
/// m_{y}\mathbf{\hat{e}_{y}} + m_{z}\mathbf{\hat{e}_{z}} \f$ are the lattice 
/// coordinates (used by the array) and \f$ a \f$ is the side length of the 
/// cubic cell, as well as the distance between two sample points in the 
/// individual voxelizations.
///////////////////////////////////////////////////////////////////////////////
class LongFCCNode
{
public:
    /// Default constructor.
    __host__ __device__ LongFCCNode(): _bid(0), _mat(0) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructs \p LongFCCNode given a \p bid.
    /// 
    /// \param[in] bid <em>Boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ LongFCCNode(ushort bid): _bid(bid), _mat(0) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructs \p LongFCCNode given both a <em>material id</em> and 
    /// <em>boundary id</em>.
    /// 
    /// \param[in] mat <em>Material id</em>.
    /// \param[in] bid <em>Boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ 
    LongFCCNode( ushort bid, uchar mat ): _bid(bid), _mat(mat) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Constructs \p LongFCCNode given both a <em>material id</em> and 
    /// <em>boundary id</em> and ratio.
    /// 
    /// \param[in] mat <em>Material id</em>.
    /// \param[in] bid <em>Boundary id</em>.
    /// \param[in] r Ratio. Has no effect.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ 
    LongFCCNode( ushort bid, uchar mat, float r ): _bid(bid), _mat(mat) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the <em>material id</em>.
    /// 
    /// \return The <em>material id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ uchar mat() const { return _mat; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the <em>boundary id</em>.
    /// 
    /// \return The <em>boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ ushort bid() const { return _bid; } 
    ///////////////////////////////////////////////////////////////////////////
    /// \brief \p LongFCCNode has no \p r value.
    /// 
    /// \return 0.0f
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ float r() const { return 0.0f; } 
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Sets the <em>material id</em>. 
    ///
    /// <em>Material id</em>s exceeding the allowed limit produces undefined 
    /// behavior.
    /// 
    /// \param[in] mat The <em>material id</em> to be inserted into the \p bim.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void mat(uchar mat) { _mat = mat; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Sets the <em>boundary id</em>. 
    ///
    /// <em>Boundary id</em>s exceeding the allowed limit produces undefined 
    /// behavior.
    /// 
    /// \param[in] bid The <em>boundary id</em> to be inserted into the \p bim.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void bid(ushort bid) { _bid = bid; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Setter method for the \a ratio. 
    ///
    /// \p ShortFCCNode has no \a ratio so does nothing.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void r(float ratio) { } // Do nothing.
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Sets both the <em>boundary id</em> and the <em>material id</em> 
    /// at the same time. 
    ///
    /// Values exceeding their specified limit produce undefined behavior. 
    /// Setting both values at once is potentially more efficient than setting 
    /// them independently.
    /// 
    /// \param[in] mat <em>Material id</em>.
    /// \param[in] bid <em>Boundary id</em>.
    ///////////////////////////////////////////////////////////////////////////
    __host__ __device__ void set(uchar mat, ushort bid)
    { 
        _bid = bid;
        _mat = mat;
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the maximum supported <em>material index</em> for this 
    ///        \p Node type.
    /// 
    /// \return UCHAR_MAX.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ ushort maxMat() { return UCHAR_MAX; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns existence of a fractional value. Only \p PartialNode has 
    /// it.
    /// 
    /// \return false.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool hasRatio() { return false; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns \p true if the \p Node is meant to be regarded as a 
    ///        point on an FCC lattice.
    /// 
    /// \return \p true.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool isFCCNode() { return true; }
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns \p true if the \p Node is meant to be used in 
    ///        conjunction with another \p Node type for surface nodes.
    ///
    /// \return \p false.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool usesTwoArrays() { return false; }
private:
    ushort _bid;    ///< Boundary id.
    uchar _mat;     ///< Material id.
};

///////////////////////////////////////////////////////////////////////////////
/// \brief A \p Node type used as a solid node when producing a twin array 
///        voxelization.
///
/// Practically identical to \p ShortNode. Could probably be replaced by it 
/// without any big changes.
///////////////////////////////////////////////////////////////////////////////
class VolumeNode
{
public:
    /// Default constructor.
    __host__ __device__ VolumeNode(): _bid(0) {}
    /// Constructor that takes bid.
    __host__ __device__ VolumeNode( uchar type ): _bid(type) {}
    /// Constructor that takes bid, mat - ignores mat.
    __host__ __device__ VolumeNode( uchar type, uchar mat ): _bid(type) {}
    /// Constructor that takes bid, mat, r - ignores mat, r.
    __host__ __device__ VolumeNode( uchar type, uchar mat, float r ): _bid(type) {}
    /// Default destructor - does nothing.
    __host__ __device__ ~VolumeNode() {}

    uchar _bid; ///< Boundary id.

    /// Getter for material id, always returns 0.
    __host__ __device__ uchar mat() const { return 0; }
    /// Getter for boundary id.
    __host__ __device__ uchar bid() const { return _bid; }
    /// Getter for ratio, always return 0.
    __host__ __device__ float r() const { return 0.0f; }
    /// Setter for material id, does nothing.
    __host__ __device__ void mat( uchar m ) {}
    /// Setter for boundary id.
    __host__ __device__ void bid( uchar b ) { _bid = b; }
    /// Setter for ratio, does nothing.
    __host__ __device__ void r( float r ) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns \p true if the \p Node is meant to be used in 
    ///        conjunction with another \p Node type for surface nodes.
    ///
    /// \return \p true.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool usesTwoArrays() { return true; }
    /// This is not a FCC node so returns false.
    static __host__ __device__ bool isFCCNode() { return false; }
    /// This node leads to a node with ratio, so returns true.
    static __host__ __device__ bool hasRatio() { return true; }
    /// The maximum material is set to 255 for convenience.
    static __host__ __device__ uchar maxMat() { return 255; }
};

///////////////////////////////////////////////////////////////////////////////
/// \brief A \p Node type used as a solid node when producing a twin array 
///        voxelization.
///
/// Meant to directly contain an index into the surface node array. Not 
/// currently used anywhere.
///////////////////////////////////////////////////////////////////////////////
class VolumeMapNode
{
public:
    /// Default constructor.
    __host__ __device__ VolumeMapNode(): _bid(0) {}
    /// Constructor with bid.
    __host__ __device__ VolumeMapNode( uint32_t i ): _bid(i) {}
    /// Constructor with bid, mat - ignores mat.
    __host__ __device__ VolumeMapNode( uint32_t type, char mat ): _bid(type) {}
    /// Constructor with bid, mat, r - ignores mat, r.
    __host__ __device__ VolumeMapNode( uint32_t type, char mat, float r ): _bid(type) {}
    /// Default destructor - does nothing.
    __host__ __device__ ~VolumeMapNode() {}

    uint32_t _bid;  ///< Index into a surface node array.

    /// Getter for mat - always returns 0.
    __host__ __device__ uint32_t mat() const { return 0; }
    /// Getter for bid.
    __host__ __device__ uint32_t bid() const { return _bid; }
    /// Getter for r - always returns 0.
    __host__ __device__ float r() const { return 0.0f; }
    /// Setter for material id - does nothing.
    __host__ __device__ void mat( uint32_t m ) {}
    /// Setter for boundary id.
    __host__ __device__ void bid( uint32_t b ) { _bid = b; }
    /// Setter for ratio - does nothing.
    __host__ __device__ void r( float r ) {}
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns \p true if the \p Node is meant to be used in 
    ///        conjunction with another \p Node type for surface nodes.
    ///
    /// \return \p true.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool usesTwoArrays() { return true; }
    /// Returns false as this is not a FCC node.
    static __host__ __device__ bool isFCCNode() { return false; }
    /// Returns true as this node points to a node with a ratio.
    static __host__ __device__ bool hasRatio() { return true; }
    /// Returns a max material id of 255 for convenience.
    static __host__ __device__ uchar maxMat() { return 255; }
};

///////////////////////////////////////////////////////////////////////////////
/// \brief 
///////////////////////////////////////////////////////////////////////////////
class SurfaceNode
{
public:
    /// Default constructor.
    __host__ __device__ SurfaceNode(): orientation(0), material(0), 
        volume(0.0f), cutNormal(make_float3(0.0f, 0.0f, 0.0f)), cutArea(0.0f), 
        xPosArea(0.0f), xNegArea(0.0f), yPosArea(0.0f), yNegArea(0.0f), 
        zPosArea(0.0f), zNegArea(0.0f)/*, mergedNodes(UINT_MAX)*/ {}
    /// Constructor with bid, mat.
    __host__ __device__ SurfaceNode( uint32_t type, uchar mat ): 
        orientation(type), material(mat), volume(0.0f), 
        cutNormal(make_float3(0.0f, 0.0f, 0.0f)), cutArea(0.0f), 
        xPosArea(0.0f), xNegArea(0.0f), yPosArea(0.0f), yNegArea(0.0f), 
        zPosArea(0.0f), zNegArea(0.0f)/*, mergedNodes(UINT_MAX)*/ {}
    /// Constructor with bid, mat, r - ignores r.
    __host__ __device__ SurfaceNode( uint32_t type, uchar mat, float r ): 
        orientation(type), material(mat), volume(0.0f), 
        cutNormal(make_float3(0.0f, 0.0f, 0.0f)), cutArea(0.0f), 
        xPosArea(0.0f), xNegArea(0.0f), yPosArea(0.0f), yNegArea(0.0f), 
        zPosArea(0.0f), zNegArea(0.0f)/*, mergedNodes(UINT_MAX)*/ {}
    /// Default destructor.
    __host__ __device__ ~SurfaceNode() {}

    uchar orientation;      ///< Boundary id.
    uchar material;         ///< Material id.

    float volume;           ///< Volume of voxel that is inside model.

    float3 cutNormal;       ///< Normal of the cut polygon.
    float cutArea;          ///< Area of the cut polygon.

    float xPosArea;         ///< Area of the cut surface along +x.
    float xNegArea;         ///< Area of the cut surface along -x.
    float yPosArea;         ///< Area of the cut surface along +y.
    float yNegArea;         ///< Area of the cut surface along -y.
    float zPosArea;         ///< Area of the cut surface along +z.
    float zNegArea;         ///< Area of the cut surface along -z.

    //uint mergedNodes;

    /// Getter for material id, always returns 0.
    __host__ __device__ uchar mat() const { return 0; }
    /// Getter for boundary id.
    __host__ __device__ uchar bid() const { return orientation; }
    /// Getter for ratio, always returns 0.
    __host__ __device__ float r() const { return 0.0f; }
    /// Setter for material id, does nothing.
    __host__ __device__ void mat( uchar m ) {}
    /// Setter for boundary id.
    __host__ __device__ void bid( uchar b ) { orientation = b; }
    /// Setter for ratio, does nothing.
    __host__ __device__ void r( float r ) {}

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns \p true if the \p Node is meant to be used in 
    ///        conjunction with another \p Node type for surface nodes.
    ///
    /// \return \p true.
    ///////////////////////////////////////////////////////////////////////////
    static __host__ __device__ bool usesTwoArrays() { return true; }
    /// Returns false as this is not a FCC node.
    static __host__ __device__ bool isFCCNode() { return false; }
    /// Returns false as this is irrelevant.
    static __host__ __device__ bool hasRatio() { return false; }
    /// Returns zero as it is irrelevant.
    static __host__ __device__ uchar maxMat() { return 0; }
};

} // End namespace vox

#endif
