#include "voxelizer_utils.h"

void testOverlap(vtkPolyData *data, uint y, uint z, bool returnAllResults, uint res)
{
    double3* vertices;
    uint3* indices;
    double3* normals;
    double3 minVertex;
    double3 maxVertex;
    double voxelDistance;
    uint3 resolution;
    bool testState;

    // Get triangles with indices (polys) and the vertices (points).
    vtkCellArray* polys = data->GetPolys();
    vtkPoints* points = data->GetPoints();

    uint numberOfVertices = (uint)points->GetNumberOfPoints();
    cout << numberOfVertices << " vertices found.\n";
    uint numberOfPolygons = (uint)polys->GetNumberOfCells();
    cout << numberOfPolygons << " polygons found.\n";

    // Allocate vertex and index arrays in main memory.
    vertices = new double3[numberOfVertices];
    indices = new uint3[numberOfPolygons];
    normals = new double3[numberOfPolygons];

    // Calculate the bounding box of the input model.
    double vert[3];
    points->GetPoint(0, vert);

    minVertex.x = vert[0];
    maxVertex.x = vert[0];
    minVertex.y = vert[1];
    maxVertex.y = vert[1];
    minVertex.z = vert[2];
    maxVertex.z = vert[2];

    for (vtkIdType i = 1; i < points->GetNumberOfPoints(); i++)
    {
        points->GetPoint(i, vert);
        // Determine minimum and maximum coordinates.

        if (vert[0] > maxVertex.x)
            maxVertex.x = vert[0];
        if (vert[1] > maxVertex.y)
            maxVertex.y = vert[1];
        if (vert[2] > maxVertex.z)
            maxVertex.z = vert[2];

        if (vert[0] < minVertex.x)
            minVertex.x = vert[0];
        if (vert[1] < minVertex.y)
            minVertex.y = vert[1];
        if (vert[2] < minVertex.z)
            minVertex.z = vert[2];
    }

    cout << "Minimum corner: " << minVertex.x << ", " << minVertex.y << ", " << minVertex.z << "\n";
    cout << "Maximum corner: " << maxVertex.x << ", " << maxVertex.y << ", " << maxVertex.z << "\n";

    // Copy vertex data to the device.
    for (vtkIdType i = 0; i < points->GetNumberOfPoints(); i++)
    {
        points->GetPoint(i, vert);

        vertices[i] = make_double3(vert[0], vert[1], vert[2]); // - minVertex;
    }

    // Copy index data to the array.
    vtkIdList* idlist = vtkIdList::New();
    uint currentPoly = 0;
    polys->InitTraversal();
    while(polys->GetNextCell(idlist))
    {
        uint index[3] = { 0, 0, 0 };
        for (vtkIdType i = 0; i < idlist->GetNumberOfIds(); i++)
        {
            if (i > 2)
            {
                cout << "Too many indices! Only triangle-meshes are supported.";
                return;
            }

            index[i] = idlist->GetId(i);
        }

        indices[currentPoly] = make_uint3(index[0], index[1], index[2]);

        // Calculate the surface normal of the current polygon.
        double3 U = make_double3(vertices[indices[currentPoly].x].x - vertices[indices[currentPoly].z].x,
                               vertices[indices[currentPoly].x].y - vertices[indices[currentPoly].z].y, 
                               vertices[indices[currentPoly].x].z - vertices[indices[currentPoly].z].z);
        double3 V = make_double3(vertices[indices[currentPoly].y].x - vertices[indices[currentPoly].x].x,
                               vertices[indices[currentPoly].y].y - vertices[indices[currentPoly].x].y, 
                               vertices[indices[currentPoly].y].z - vertices[indices[currentPoly].x].z);

        normals[currentPoly] = normalize(cross(U, V));

        currentPoly++;
    }

    double3 diffVertex = maxVertex - minVertex;

    if (diffVertex.x > diffVertex.y)
    {
        if (diffVertex.x > diffVertex.z)
        {
            voxelDistance = diffVertex.x / double(res - 1);
            resolution.x = res;
            resolution.y = uint(diffVertex.y / voxelDistance) + 1;
            resolution.z = uint(diffVertex.z / voxelDistance) + 1;
        }
        else
        {
            voxelDistance = diffVertex.z / double(res - 1);
            resolution.x = uint(diffVertex.x / voxelDistance) + 1;
            resolution.y = uint(diffVertex.y / voxelDistance) + 1;
            resolution.z = res;
        }
    }
    else
    {
        if (diffVertex.y > diffVertex.z)
        {
            voxelDistance = diffVertex.y / double(res - 1);
            resolution.x = uint(diffVertex.x / voxelDistance) + 1;
            resolution.y = res;
            resolution.z = uint(diffVertex.z / voxelDistance) + 1;
        }
        else
        {
            voxelDistance = diffVertex.z / double(res - 1);
            resolution.x = uint(diffVertex.x / voxelDistance) + 1;
            resolution.y = uint(diffVertex.y / voxelDistance) + 1;
            resolution.z = res;
        }
    }

    if (resolution.x % 32 != 0)
        resolution.x += 32 - (resolution.x % 32);
    if (resolution.y % 16 != 0)
        resolution.y += 16 - (resolution.y % 16);
    if (resolution.z % 16 != 0)
        resolution.z += 16 - (resolution.z % 16);

    cout << "Voxel width: " << voxelDistance << "\n";
    cout << "Resolution: " << resolution.x << " X " << resolution.y << " X " << resolution.z << "\n";
    cout << "YZ-coordinates: (" << y << ", " << z << ")\n\n";

    uint intersectionCount = 0;
    std::vector<uint> intersections = std::vector<uint>();

    // Shoot a ray from the voxel coordinates, and test for intersection against every triangle.
    for (uint i = 0; i < numberOfPolygons; i++)
    {
        double2 vertexProjections[3];
        double2 edgeNormals[3];
        double distanceToEdge[3];
        double testResult[3];

        if (normals[i].x == 0.0)
            continue;

        vertexProjections[0] = make_double2(vertices[indices[i].x].y, vertices[indices[i].x].z);
        vertexProjections[1] = make_double2(vertices[indices[i].y].y, vertices[indices[i].y].z);
        vertexProjections[2] = make_double2(vertices[indices[i].z].y, vertices[indices[i].z].z);

        double2 p = make_double2(minVertex.y + double(y) * voxelDistance, minVertex.z + double(z) * voxelDistance);

        double2 vMin = vertexProjections[0];
        double2 vMax = vertexProjections[0];

        if (vertexProjections[1].x < vMin.x)
            vMin.x = vertexProjections[1].x;
        if (vertexProjections[1].y < vMin.y)
            vMin.y = vertexProjections[1].y;
        if (vertexProjections[2].x < vMin.x)
            vMin.x = vertexProjections[2].x;
        if (vertexProjections[2].y < vMin.y)
            vMin.y = vertexProjections[2].y;

        if (vertexProjections[1].x > vMax.x)
            vMax.x = vertexProjections[1].x;
        if (vertexProjections[1].y > vMax.y)
            vMax.y = vertexProjections[1].y;
        if (vertexProjections[2].x > vMax.x)
            vMax.x = vertexProjections[2].x;
        if (vertexProjections[2].y > vMax.y)
            vMax.y = vertexProjections[2].y;

        if ((p.x < vMin.x) || (p.y < vMin.y) || (p.x > vMax.x) || (p.y > vMax.y))
            continue;

        for (uint j = 0; j < 3; j++)
        {
            double2 edge = make_double2(vertexProjections[(j+1)%3].x - vertexProjections[j].x, 
                                        vertexProjections[(j+1)%3].y - vertexProjections[j].y);
            if (normals[i].x >= 0.0)
                edgeNormals[j] = normalize(make_double2(-edge.y, edge.x));
            else
                edgeNormals[j] = normalize(make_double2(edge.y, -edge.x));

            distanceToEdge[j] = edgeNormals[j].x * vertexProjections[j].x + edgeNormals[j].y * vertexProjections[j].y;
            distanceToEdge[j] *= -1.0;

            testResult[j] = distanceToEdge[j] + edgeNormals[j].x * p.x + edgeNormals[j].y * p.y;

            if ((edgeNormals[j].x > 0.0) || ((edgeNormals[j].x == 0) && (edgeNormals[j].y < 0.0)))
                testResult[j] += DBL_MIN;
            
        }

        if ((testResult[0] > 0.0) && (testResult[1] > 0.0) && (testResult[2] > 0.0))
            testState = true;
        else
            testState = false;

        if (testState || returnAllResults)
        {
            if (testState)
            {
                intersectionCount++;
                cout << "Intersection nr. " << intersectionCount << " found with triangle " << i << "\n";
            }
            else
                cout << "No intersection found with triangle " << i << "\n";
            cout << "Vertex 1      : (" << vertexProjections[0].x << ", " << vertexProjections[0].y << ")\n";
            cout << "Vertex 2      : (" << vertexProjections[1].x << ", " << vertexProjections[1].y << ")\n";
            cout << "Vertex 3      : (" << vertexProjections[2].x << ", " << vertexProjections[2].y << ")\n";
            cout << "Normal        : (" << normals[i].x << ", " << normals[i].y << ", " << normals[i].z << ")\n";
            cout << "Edge normal 1 : (" << edgeNormals[0].x << ", " << edgeNormals[0].y << ")\n";
            cout << "Edge normal 2 : (" << edgeNormals[1].x << ", " << edgeNormals[1].y << ")\n";
            cout << "Edge normal 3 : (" << edgeNormals[2].x << ", " << edgeNormals[2].y << ")\n";
            cout << "Distances     : " << distanceToEdge[0] << ", " << distanceToEdge[1] << ", " << distanceToEdge[2] << "\n";
            cout << "Test results  : " << testResult[0] << ", " << testResult[1] << ", " << testResult[2] << "\n";
            cout << "P             : " << p.x << ", " << p.y << "\n";
            cout << (testState ? "Intersection found" : "No intersection") << " at X: ";

            double3 v = vertices[indices[i].x];
            double3 n = normals[i];

            double3 A = make_double3(v.x - minVertex.x, v.y - p.x, v.z - p.y);
            double B = dot(A, n);
            double px = B / n.x;

            uint voxelX = ceilf(px / voxelDistance);

            cout << voxelX << "\n\n";

            if (testState)
                intersections.push_back(voxelX);
        }
    }

    uint voxels[UTIL_RES / 32];

    for (uint i = 0; i < UTIL_RES / 32; i++)
        voxels[i] = 0;

    if (intersectionCount > 0)
    {
        for (std::vector<unsigned int>::iterator it = intersections.begin(); it < intersections.end(); it++)
        {
            uint intersection = *it;

            uint relevantInteger = intersection / 32;
            uint relevantBit = intersection % 32;

            uint bitmask = UINT_MAX >> relevantBit;
            voxels[relevantInteger] ^= bitmask;

            for (uint i = relevantInteger + 1; i < UTIL_RES / 32; i++)
                voxels[i] ^= UINT_MAX;
        }

        cout << "The voxelization for Y: " << y << ", Z: " << z << "\n\n";
        char hexRep[12];
        for (uint i = 0; i < UTIL_RES / 32; i++)
        {
            sprintf_s(hexRep, 12*sizeof(char), "%X", voxels[i]);
            cout << hexRep << " ";
        }

        cout << "\n\n";
    }
    else if (returnAllResults)
    {

    }
    else
        cout << "No intersections found...\n\n";

    delete vertices, indices, normals;
}

intList make_intList(uint i, uint* voxels, uint3 res)
{
    intList result = { 0, 0, 0, 0, 0, 0, 0 };
    result.ipx = res.x / 32;
    result.ipz = result.ipx * res.y;

    result.x = i % result.ipx;
    result.y = (i % result.ipz) / result.ipx;
    result.z = i / result.ipz;

    // Centre voxel is the given voxel.
    result.centre = voxels[i];

    // There are ipx integers per y-coordinate. The first and last integers in a row of same-y-integers
    // don't have a left or right integer.
    if (result.x > 0)
        result.left = voxels[i-1];
    if (result.x < result.ipx - 1)
        result.right = voxels[i+1];

    // There are ipz integers per z-coordinate. The first ipx and last ipx integers of a particular range
    // of same-y-coordinates don't have a valid in or out integer.
    if (result.y > 0)
        result.in = voxels[i - result.ipx];
    if (result.y < res.y - 1)
        result.out = voxels[i + result.ipx];

    // There are ipz integers per z-coordinate. The first ipz and last ipz integers of the entire voxelization
    // don't have valid up or down integers.
    if (result.z > 0)
        result.down = voxels[i - result.ipz];
    if (result.z < res.z - 1)
        result.up = voxels[i + result.ipz];

    return result;
}

template <class NT>
nodeList make_nodeList(NT* nodes, uint i, uint3 resolution)
{
    nodeList result = { 0, 0, 0, 0, 0, 0, 0 };

    result.x = i % resolution.x;
    result.y = (i % (resolution.x * resolution.y)) / resolution.x;
    result.z = i / (resolution.x * resolution.y);

    result.centre = nodes[i].bid();

    if (result.x > 0)
        result.left = nodes[i-1].bid();

    if (result.x < resolution.x - 1)
        result.right = nodes[i+1].bid();

    if (result.z > 0)
        result.down = nodes[i - (resolution.x * resolution.y)].bid();

    if (result.z < resolution.z - 1)
        result.up = nodes[i + (resolution.x * resolution.y)].bid();

    if (result.y > 0)
        result.in = nodes[i - resolution.x].bid();

    if (result.y < resolution.y - 1)
        result.out = nodes[i + resolution.x].bid();

    return result;
}

bitList make_bitList(intList ints, int bit)
{
    bitList result;
    const uint i = 0x80000000;

    result.centre = ((i >> bit) & ints.centre) > 0;

    if (bit == 0)
        result.left = (1 & ints.left) > 0;
    else
        result.left = ((i >> (bit - 1)) & ints.centre) > 0;

    if (bit == 31)
        result.right = (i & ints.right) > 0;
    else
        result.right = ((i >> (bit + 1)) & ints.centre) > 0;

    result.in = ((i >> bit) & ints.in) > 0;
    result.out = ((i >> bit) & ints.out) > 0;
    result.up = ((i >> bit) & ints.up) > 0;
    result.down = ((i >> bit) & ints.down) > 0;

    return result;
}

void printVoxels(uint* voxels, uint3 resolution)
{
    cout << "Printing voxelization: \n\n";
    uint intRes = resolution.x / 32;
    uint nrOfInts = resolution.z * resolution.y * intRes;
    char hexRep[8];
    for (uint i = 0; i < nrOfInts; i++)
    {
        sprintf_s(hexRep, 8*sizeof(char), "%X", voxels[i]);
        cout << hexRep << " ";
        if ((i+1)%intRes == 0)
            cout << "\n";
        if ((i+1)%(resolution.y*intRes) == 0)
            cout << "\n";
    }
}

template <class NT>
void printNodes(NT* nodes, uint3 resolution)
{
    uint nrOfNodes = resolution.x * resolution.y * resolution.z;
    for (uint i = 0; i < nrOfNodes; i++)
    {
        cout << nodes[i].bid() << "(" << nodes[i].mat() << ") ";
        if ((i+1)%10 == 0)
            cout << "\n";
        if ((i+1)%(resolution.x * resolution.y) == 0)
            cout << "\n";
    }
}

int getVoxel(uint value, int bit)
{
    return (value & (0x80000000 >> bit)) > 0 ? 1 : 0;
}

uint calculateNrOfSolidVoxels(uint* voxels, uint3 resolution)
{
    uint nrOfInts = resolution.z * resolution.y * (resolution.x / 32);
    uint nrOfSolidVoxels = 0;
    for (uint i = 0; i < nrOfInts; i++)
    {
        uint voxel = voxels[i];
        if (voxel == 0)
            continue;
        if (voxel == UINT_MAX)
        {
            nrOfSolidVoxels += 32;
            continue;
        }

        for (int b = 0; b < 32; b++)
            nrOfSolidVoxels += ((0x80000000U >> b) & voxel) > 0 ? 1 : 0;
    }
    return nrOfSolidVoxels;
}

template <class NT>
uint calculateNrOfSolidNodes(NT* nodes, uint3 resolution)
{
    uint nrOfNodes = resolution.x * resolution.y * resolution.z;
    uint nrOfSolidNodes = 0;
    for (uint i = 0; i < nrOfNodes; i++)
    {
        if (nodes[i].bid() > 0)
            nrOfSolidNodes++;
    }
    return nrOfSolidNodes;
}

void printVoxelizationIntoFile(char* filename, uint* voxels, uint3 resolution)
{
    ofstream log;
    log.open(filename, ios::out | ios::trunc);
    time_t rawtime;
    time(&rawtime);
    struct tm timeinfo; // = localtime(&rawtime);
    localtime_s(&timeinfo, &rawtime);
    char timeAndDate[50];
    asctime_s(timeAndDate, 50*sizeof(char), &timeinfo);
    log << "Voxelizer logfile: " << timeAndDate;

    uint intsPerX = resolution.x >> 5;
    uint nrOfInts = resolution.z * resolution.y * intsPerX;

    log << "Resolution: (" << resolution.x << ", " << resolution.y << ", " << resolution.z << "), Integers on the x-axis: " << intsPerX << ".\n\n";

    char hexRep[12];
    char yCounter[10];
    uint y_next, z_next;
    log << "Z: 0\n\n";
    log << "   0: ";
    for (uint i = 0; i < nrOfInts; i++)
    {
        sprintf_s(hexRep, 12*sizeof(char), "%8X", voxels[i]);
        log << hexRep << " ";

        if ((i + 1) % (resolution.y * intsPerX) == 0)
        {
            z_next = (i + 1) / (resolution.y * intsPerX);

            log << "\n\n";
            log << "Z: " << z_next << "\n";
        }

        if ((i + 1) % intsPerX == 0)
        {
            y_next = ((i + 1) % (resolution.y * intsPerX)) / intsPerX;

            log << "\n";
            sprintf_s(yCounter, 10*sizeof(char), "%4u", y_next);
            log << yCounter << ": ";
        }
    }
    log.close();
}

template <class NT>
void compareTwoNodeRepresentations(NT* nodes1, NT* nodes2, uint3 resolution)
{
    uint nrOfNodes = resolution.x * resolution.y * resolution.z;
    uint nrOfDifferingNodes = 0;
    bool differenceFound;
    NT node1, node2;

    for (uint i = 0; i < nrOfNodes; i++)
    {
        differenceFound = false;
        node1 = nodes1[i];
        node2 = nodes2[i];

        if (node1.bid() != node2.bid())
        {
            differenceFound = true;
            cout << "Difference found in BID\n";
        }
        if (node1.mat() != node2.mat())
        {
            differenceFound = true;
            cout << "Difference found in MAT\n";
        }

        if (differenceFound)
            nrOfDifferingNodes++;
    }
    cout << "Number of differing nodes: " << nrOfDifferingNodes << "\n";
}
