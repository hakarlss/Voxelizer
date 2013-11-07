#ifdef _WIN32
#pragma once
#endif

#ifndef VOXELIZER_UTILS_H
#define VOXELIZER_UTILS_H

#include "voxelizer.h"

#include "vtkPolyData.h"
#include "vtkCellArray.h"
#include "vtkPolygon.h"

#include <math.h>

#define UTIL_RES 256

struct intList
{
	uint centre, left, right, up, down, in, out;
	uint ipx, ipz, x, y, z;
};

struct nodeList
{
	uint centre, left, right, up, down, in, out;
	uint x, y, z;
};

struct bitList
{
	bool centre, left, right, up, down, in, out;
};

void testOverlap(vtkPolyData *data, uint y, uint z, bool returnAllResults, uint resolution);

double3 normalize(double3 v);
double2 normalize(double2 v);
double dot(double3 a, double3 b);
double dot(double2 a, double2 b);
double3 cross(double3 a, double3 b);

intList make_intList(uint i, uint* voxels, uint3 res);
template <class NT>
nodeList make_nodeList(NT* nodes, uint i, uint3 resolution);
bitList make_bitList(intList ints, int bit);

void printVoxels(uint* voxels, uint3 resolution);
template <class NT>
void printNodes(NT* nodes, uint3 resolution);
uint calculateNrOfSolidVoxels(uint* voxels, uint3 resolution);
template <class NT>
uint calculateNrOfSolidNodes(NT* nodes, uint3 resolution);
void printVoxelizationIntoFile(char* filename, uint* voxels, uint3 resolution);
template <class NT>
uint* convertNodesToIntegers(NT* nodes, uint3 resolution)
{
	uint* voxels = new uint[resolution.z * resolution.y * resolution.x / 32];

	uint bitpattern = 0;
	uint x = 0;
	uint bit = 0;
	uint currentVoxel = 0;

	for (uint i = 0; i < resolution.x * resolution.y * resolution.z; i++)
	{
	    x = i % resolution.x;

		if (nodes[i].bid() > 0)
		{
			bitpattern |= 1u << (31 - bit);
		}

		bit++;

		if ((x == resolution.x - 1) || (bit == 32))
		{
			// Put the integer into voxels and reset the mechanism.
			voxels[currentVoxel] = bitpattern;
			bitpattern = 0;
			bit = 0;
			currentVoxel++;
		}
	}

	return voxels;
}

template <class NT>
void compareTwoNodeRepresentations(NT* nodes1, NT* nodes2, uint3 resolution);

#endif