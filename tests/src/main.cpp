#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>
#include <vtkActor.h>
#include <vtkRenderer.h>

#include <vtkSmartPointer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolyDataReader.h>
#include <vtkPLYReader.h>
#include <vtkProperty.h>

#include <vtkQuad.h>
#include <vtkLine.h>
#include <vtkVertex.h>
#include <vtkGraph.h>
#include <vtkMutableUndirectedGraph.h>
#include <vtkGraphLayoutView.h>
#include <vtkIntArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkLookupTable.h>
#include <vtkDataSetAttributes.h>
#include <vtkViewTheme.h>
#include <vtkCamera.h>
#include <vtkPointData.h>
#include <vtkVertexGlyphFilter.h>

#include <voxelizer.h>
#include <voxelizer_utils.h>

#include <vector>
#include <iterator>
#include <memory>
#include <map>

#include <string>
#include <functional>

#include <boost/program_options.hpp>

void visualize(vtkPolyData* data, double3 minVertex, double3 maxVertex, double voxelLength, int y, int z, uint3 res, vox::MainAxis direction);
void renderVoxelization(unsigned int* voxels, bool onlySurface, uint3 res);
template <class Node>
void renderNodeOutput(Node* nodes, bool onlySurface, bool materials, uint3 res);
void addVoxel(float* voxelCenter, float voxelWidth, vtkPoints* points, vtkCellArray* quads);
uint* generateRandomVoxelization(uint3 resolution);
uint* generateSpecificVoxelization(uint3 resolution, uint minX, uint maxX, uint minY, uint maxY, uint minZ, uint maxZ);
uint compose32BitIntFromRands();
void testNodeCreation(uint* voxels, uint3 resolution);
template <class Node>
void testMaterials(Node const * nodes, float const * vertices, uint const * indices, uint nrOfTriangles, int x, int y, int z, uint3 resolution, float voxelLength);
template <class Node>
void testMaterialsLoop(Node const * nodes, float const * vertices, uint const * indices, uint nrOfTriangles, uint3 resolution, float voxelLength);
std::pair<std::unique_ptr<float[]>, std::unique_ptr<uint[]>> loadModel(vtkSmartPointer<vtkPolyData> & model, uint &nrOfVertices, uint &nrOfTriangles);
void testCrossproduct();
template <class Node>
uint numberOfNodesWithNoMaterial(Node* nodes, uint3 resolution);
void testTrianglePlaneIntersection();
template <class Node>
void analyzeRatioNodes(Node* nodes, uint3 resolution, float voxelLength, int nthErroneousRatio);
template <class Node>
void printOrientationDistribution(Node* nodes, uint3 resolution);
template <class Node>
Node* stitchNodeArrays(std::vector<vox::NodePointer<Node>> const &nps, uint2 splits);
template <class Node>
std::vector<vox::NodePointer<Node>> createTestNodes(uint2 splits);
template <class Node>
uint3 fetchResolution(std::vector<vox::NodePointer<Node>> const & nps, uint2 splits );
template <class Node>
void compareNodeArrays(Node const * nodes1, Node const * nodes2, uint3 res);
inline std::string printUint3( uint3 & vec );
template <class Node>
void applySlice( Node * slice
               , uint sliceNr
               , Node * nodes
               , uint3 res
               , uint direction );
template <class NT>
void renderFCCNodeOutput(NT* nodes, bool onlySurface, bool materials, uint3 res);
void testFunction();

template <typename T>
void validate( std::string var
             , std::string msg
             , boost::program_options::variables_map & vm
             , std::function<bool( T & )> func );

template <class Node>
void initTests
    ( const boost::program_options::variables_map & vm
    , const vtkSmartPointer<vtkPolyData> & polys 
    );

template<class Node>
void performPlainVoxelTest
    ( vox::Voxelizer<Node> & voxelizer
    , const boost::program_options::variables_map & vm
    );

template<class Node>
void performFCCNodeTest
    ( vox::Voxelizer<Node> & voxelizer
    , const boost::program_options::variables_map & vm
    );

template<class Node>
void performSliceTest
    ( vox::Voxelizer<Node> & voxelizer
    , const boost::program_options::variables_map & vm
    );

template<class Node>
void performNodeTest
    ( vox::Voxelizer<Node> & voxelizer
    , const boost::program_options::variables_map & vm
    );

template<class Node>
void performAllSliceTest
    ( vox::Voxelizer<Node> & voxelizer
    , const boost::program_options::variables_map & vm
    );

class TestClass
{
public:
    TestClass() throw() { 
        a = "-"; 
        print( "Default constructor! ( " + a + " )\n" ); 
    }
    TestClass( int i ) throw() {
        a = std::to_string((long long)i);
        print( std::string( "Constructor with argument " + a + "\n" ) );
    }
    TestClass( const TestClass & rhs ) throw() {
        auto oldA = a;
        a = rhs.a; 
        print( "Copy constructor! ( " + oldA + " to " + a + " )\n" ); 
    }
    TestClass( TestClass && rhs ) throw() { 
        auto oldA = a;
        std::swap( a, rhs.a );
        print( "Move constructor! ( " + oldA + " to " + a + " )\n" ); 
    }
    ~TestClass() { print( "Default destructor! ( " + a + " )\n" );  }

    TestClass & operator=( const TestClass & rhs ) throw() {
        auto oldA = a;
        a = rhs.a;
        print( "Copy assignment! ( " + oldA + " to " + a + " )\n" ); 
        return *this;
    }
    TestClass & operator=( TestClass && rhs ) throw() {
        auto oldA = a;
        std::swap( a, rhs.a );
        print( "Move assignment! ( " + oldA + " to " + a + " )\n" ); 
        return *this;
    }

    std::string get() const throw() { return a; }

private:
    std::string a;

    void print( std::string msg ) throw() { std::cout << msg; }
};

class Printable
{
public:
    virtual std::string print() = 0;
};

class Person : public Printable
{
public:
    Person() {}
    Person( std::string n ) { name = n; }

    std::string print() { return std::string( "Person name: " ) + name + "\n"; }
private:
    std::string name;
};

class Car : public Printable
{
public:
    Car() {}
    Car( std::string m ) { model = m; }

    std::string print() { return std::string( "Car model: " ) + model + "\n"; }
private:
    std::string model;
};

int main(int argc, char** argv)
{
    // Define options that the command line parser accepts.
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ( "help,h"
        , "Produce help message." )
        ( "resolution,r"
        , boost::program_options::value<uint>()
        , "Number of voxels along longest side." )
        ( "distance,d"
        , boost::program_options::value<double>()
        , "Distance between voxel centers." )
        ( "voxels,x"
        , "Produce voxel output instead of nodes." )
        ( "materials,m"
        , "Produce materials output with a simple mapping." )
        ( "long,L"
        , "Use the larger vox::LongNode type." )
        ( "part,P"
        , "Use the vox::PartialNode type." )
        ( "fcc,F"
        , "Use one of the FCC node types." )
        ( "slice,S"
        , boost::program_options::value<std::vector<uint>>()->multitoken()
        , "Produce a slice of the total voxelization." )
        ( "allSlice,A"
        , boost::program_options::value<int>()
        , "Voxelizes and concatenates all slices." )
        ( "multiDevice,D"
        , boost::program_options::value<std::vector<uint>>()->multitoken()
        , "Simulate multiple devices and combine the results." )
        ( "voxRes,V"
        , boost::program_options::value<std::vector<uint>>()->multitoken()
        , "Internal voxelization dimensions." )
        ( "matRes,M"
        , boost::program_options::value<std::vector<uint>>()->multitoken()
        , "Internal material voxelization dimensions." )
        ( "verbose,v"
        , "Prints detailed information about the voxelization." )
        ( "surface,u"
        , "Only renders the surface of the voxelization." )
        ( "filename"
        , boost::program_options::value<std::string>()->required()
        , "File to be voxelized." )
    ;

    // Make the filename not require a prefix, but require it to be at the end.
    boost::program_options::positional_options_description p;
    p.add( "filename", 1 );

    // Parse the arguments into a variables map.
    boost::program_options::variables_map vm;
    try {
        boost::program_options::store( 
            boost::program_options::command_line_parser( argc, argv ).
            options( desc ).positional( p ).run(), vm );
        boost::program_options::notify( vm );
    }
    catch ( std::exception & e )
    {
        std::cout << e.what() << "\n";
        return 1;
    }

    // Print help message.
    if ( vm.count( "help" ) )
    {
        std::cout << desc << "\n";
        return 1;
    }

    // Require that either the resolution or the distance be given.
    if ( vm.count( "resolution" ) == 0 && vm.count( "distance" ) == 0 )
    {
        std::cout << "No resolution or distance was given.\n";
        return 1;
    }

    if ( vm.count( "voxRes" ) )
    {
        if ( vm.count( "slice" ) )
        {
            if ( vm[ "voxRes" ].as<std::vector<uint>>().size() != 2 )
            {
                std::cout << "--voxRes requires 2 arguments with --slice.\n";
                return 1;
            }
        }
        else
        {
            if ( vm[ "voxRes" ].as<std::vector<uint>>().size() != 3 )
            {
                std::cout << "--voxRes requires 3 arguments.\n";
                return 1;
            }
        }
    }

    if ( vm.count( "matRes" ) )
    {
        if ( vm.count( "slice" ) )
        {
            if ( vm[ "matRes" ].as<std::vector<uint>>().size() != 2 )
            {
                std::cout << "--matRes requires 2 arguments with --slice.\n";
                return 1;
            }
        }
        else
        {
            if ( vm[ "matRes" ].as<std::vector<uint>>().size() != 3 )
            {
                std::cout << "--matRes requires 3 arguments.\n";
                return 1;
            }
        }
    }

    if ( vm.count( "slice" ) )
    {
        auto sliceArgs = vm[ "slice" ].as<std::vector<uint>>();

        if ( sliceArgs.size() != 2 )
        {
            std::cout << "--slice requires 2 arguments.\n";
            return 1;
        }

        if ( sliceArgs.at(0) > 2 )
        {
            std::cout << "The direction of slice is > 2.\n";
            return 1;
        }
    }

    if ( vm.count( "allSlice" ) )
    {
        int dir = vm["allSlice"].as<int>();
        if ( dir < 0 || dir > 2 )
        {
            std::cout << "The direction has to be between 0 and 2.\n";
            return 1;
        }
    }

    if ( vm.count( "multiDevice" ) )
    {
        auto devConf = vm[ "multiDevice" ].as<std::vector<uint>>();

        if ( devConf.size() != 2 )
        {
            std::cout << "--multiDevice: Device configuration needs two "
                "values.\n";
            return 1;
        }

        std::cout << "DevConf Y: " << devConf[0] << "\n";
        std::cout << "DevConf Z: " << devConf[1] << "\n";
    }


    // Parse the filename and load the model data. Two filetypes supported.
    vtkSmartPointer<vtkPolyData> polys;

    if ( vm.count( "filename" ) == 0 )
    {
        std::cout << "No filename was given" << std::endl;
        return 1;
    }
    else
    {
        std::string file = vm["filename"].as<std::string>();

        if ( file.rfind( ".vtk" ) != std::string::npos )
        {
            vtkSmartPointer<vtkPolyDataReader> reader = 
                vtkSmartPointer<vtkPolyDataReader>::New();
            reader->SetFileName( file.c_str() );
            reader->Update();
            polys = reader->GetOutput();
        }

        if ( file.rfind( ".ply" ) != std::string::npos )
        {
            vtkSmartPointer<vtkPLYReader> reader = 
                vtkSmartPointer<vtkPLYReader>::New();
            reader->SetFileName( file.c_str() );
            reader->Update();
            polys = reader->GetOutput();
        }
    }

    if ( polys.GetPointer() == NULL )
    {
        std::cout << "Invalid filename given" << std::endl;
    }

    // Fail if bad model data was received.
    if ( polys->GetNumberOfCells() == 0 )
    {
        std::cout << "Invalid model data given" << std::endl;
        return 1;
    }

    else if ( vm.count( "long" ) )
    {
        if ( vm.count( "fcc" ) )
        {
            std::cout << "Starting tests with vox::LongFCCNode @ " << 
                sizeof(vox::LongFCCNode) << " bytes per node.\n";
            initTests<vox::LongFCCNode>( vm, polys );
        }
        else
        {
            std::cout << "Starting tests with vox::LongNode @ " << 
                sizeof(vox::LongNode) << " bytes per node.\n";
            initTests<vox::LongNode>( vm, polys );
        }
    }
    else if ( vm.count( "fcc" ) )
    {
        std::cout << "Starting tests with vox::ShortFCCNode @ " << 
                sizeof(vox::ShortFCCNode) << " bytes per node.\n";
        initTests<vox::ShortFCCNode>( vm, polys );
    }
    else if ( vm.count( "part" ) )
    {
        std::cout << "Starting tests with vox::PartialNode @ " << 
                sizeof(vox::PartialNode) << " bytes per node.\n";
        initTests<vox::PartialNode>( vm, polys );
    }
    else
    {
        std::cout << "Starting tests with vox::ShortNode @ " << 
                sizeof(vox::ShortNode) << " bytes per node.\n";
        initTests<vox::ShortNode>( vm, polys );
    }

    // testFunction();

    return 0;
}

void visualize(vtkPolyData* data, double3 minVertex, double3 maxVertex, double d, int y, int z, uint3 res, vox::MainAxis direction)
{
    vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
    vtkSmartPointer<vtkPoints> points = data->GetPoints();

    if ((y > 1024) || (z > 1024))
    {
        // Do nothing.
    }
    else
    {
        double p0[3], p1[3];

        if (direction == vox::xAxis)
        {
            p0[0] = minVertex.x;
            p0[1] = minVertex.y + y*d;
            p0[2] = minVertex.z + z*d;
            p1[0] = minVertex.x + double(res.x - 1) * d;
            p1[1] = minVertex.y + y*d;
            p1[2] = minVertex.z + z*d;
        }
        else if (direction == vox::yAxis)
        {
            p0[0] = minVertex.x + z*d;
            p0[1] = minVertex.y;
            p0[2] = minVertex.z + y*d;
            p1[0] = minVertex.x + z*d;
            p1[1] = minVertex.y + double(res.y - 1) * d;
            p1[2] = minVertex.z + y*d;
        }
        else if (direction == vox::zAxis)
        {
            p0[0] = minVertex.x + y*d;
            p0[1] = minVertex.y + z*d;
            p0[2] = minVertex.z;
            p1[0] = minVertex.x + y*d;
            p1[1] = minVertex.y + z*d;
            p1[2] = minVertex.z + double(res.z - 1) * d;
        }

        vtkIdType pid0 = points->InsertNextPoint(p0);
        vtkIdType pid1 = points->InsertNextPoint(p1);

        line->GetPointIds()->SetId(0, pid0);
        line->GetPointIds()->SetId(1, pid1);

        lines->InsertNextCell(line);
    }

    data->SetPoints(points);
    data->SetLines(lines);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInput(data);
 
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    //actor->GetProperty()->SetRepresentationToWireframe();
 
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);
 
    renderer->AddActor(actor);
    renderer->SetBackground(.3, .6, .3);
 
    renderWindow->Render();
    renderWindowInteractor->Start();
}

void renderVoxelization(unsigned int* voxels, bool onlySurface, uint3 res)
{
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

    cout << "entered renderVoxelization\n";

    intList ints;
    bitList bits;

    // RESOLUTION * 8 * z + 8 * y + x
    unsigned int intRes = res.x / 32;
    for (unsigned int i = 0; i < res.z * res.y * intRes; i++)
    {
        ints = make_intList(i, voxels, res);

        float p[3] = { 0.0, ints.y, ints.z };

        for (unsigned int j = 0; j < 32; j++)
        {
            if (unsigned int((ints.centre >> (31 - j)) & (unsigned int(1))) == unsigned int(1))
            {
                if (onlySurface)
                {
                    bits = make_bitList(ints, j);

                    if (bits.left && bits.right && bits.in && bits.out && bits.down && bits.up)
                    {
                        continue;
                    }
                }

                p[0] = 32 * ints.x + j;

                vtkIdType pid = points->InsertNextPoint(p);
                cells->InsertNextCell(1, &pid);
            }
        }
    }

    vtkSmartPointer<vtkPolyData> data = vtkSmartPointer<vtkPolyData>::New();
    data->SetPoints(points);
    data->SetVerts(cells);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInput(data);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderer->AddActor(actor);

    renderWindow->Render();
    renderWindowInteractor->Start();
}

template <class NT>
void renderNodeOutput(NT* nodes, bool onlySurface, bool materials, uint3 res)
{
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

    // Colors
    unsigned char black[3] = { 0, 0, 0 };
    unsigned char red[3] = { 255, 0, 0 };
    unsigned char green[3] = { 0, 255, 0 };
    unsigned char blue[3] = { 0, 0, 255 };
    unsigned char yellow[3] = { 255, 255, 0 };
    unsigned char purple[3] = { 255, 0, 255 };
    unsigned char aquamarine[3] = { 0, 255, 255 };
    unsigned char grey[3] = { 127, 127, 127 };

    unsigned char white[3] = { 255, 255, 255 };

    unsigned char * colorList[8] = { white, red, green, blue, yellow, purple, aquamarine, grey };

    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);
    colors->SetName("Color");

    for (uint nodeIdx = 0; nodeIdx < res.x * res.y * res.z; nodeIdx++)
    {
        if (onlySurface)
        {
            // Ignore air nodes.
            if (nodes[nodeIdx].bid() == 27)
                continue;
        }

        if (materials)
        {
            // Ignore nodes with zero material.
            if (nodes[nodeIdx].mat() == 0)
                continue;
        }
        
        // Ignore non-solid nodes.
        if (nodes[nodeIdx].bid() == 0)
            continue;

        uint x = nodeIdx % res.x;
        uint y = (nodeIdx % (res.x * res.y)) / res.x;
        uint z = nodeIdx / (res.x * res.y);

        //cout << "Interesting node at X: " << x << ", Y: " << y << ", Z: " << z << ".\n";

        points->InsertNextPoint(double(x), double(y), double(z));

        if (materials)
            colors->InsertNextTupleValue(colorList[nodes[nodeIdx].mat()]);
        else
            colors->InsertNextTupleValue(white);
    }

    vtkSmartPointer<vtkPolyData> data = vtkSmartPointer<vtkPolyData>::New();
    data->SetPoints(points);

    vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    vertexFilter->SetInputConnection(data->GetProducerPort());
    vertexFilter->Update();

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->ShallowCopy(vertexFilter->GetOutput());

    polydata->GetPointData()->SetScalars(colors);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInput(polydata);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetPointSize(1);

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderer->AddActor(actor);

    renderWindow->Render();
    renderWindowInteractor->Start();
}


void addVoxel(float* voxelCenter, float voxelWidth, vtkPoints* points, vtkCellArray* quads)
{
    float w = voxelWidth / 2.0;
    float p0[3] = { *voxelCenter - w, *voxelCenter + w, *voxelCenter + w };
    float p1[3] = { *voxelCenter - w, *voxelCenter - w, *voxelCenter + w };
    float p2[3] = { *voxelCenter + w, *voxelCenter - w, *voxelCenter + w };
    float p3[3] = { *voxelCenter + w, *voxelCenter + w, *voxelCenter + w };

    float p4[3] = { *voxelCenter - w, *voxelCenter + w, *voxelCenter - w };
    float p5[3] = { *voxelCenter - w, *voxelCenter - w, *voxelCenter - w };
    float p6[3] = { *voxelCenter + w, *voxelCenter - w, *voxelCenter - w };
    float p7[3] = { *voxelCenter + w, *voxelCenter + w, *voxelCenter - w };

    float fpid0 = points->InsertNextPoint(p0);
    float fpid1 = points->InsertNextPoint(p1);
    float fpid2 = points->InsertNextPoint(p2);
    float fpid3 = points->InsertNextPoint(p3);
    float fpid4 = points->InsertNextPoint(p4);
    float fpid5 = points->InsertNextPoint(p5);
    float fpid6 = points->InsertNextPoint(p6);
    float fpid7 = points->InsertNextPoint(p7);

    vtkSmartPointer<vtkQuad> quad0 = vtkSmartPointer<vtkQuad>::New();
    vtkSmartPointer<vtkQuad> quad1 = vtkSmartPointer<vtkQuad>::New();
    vtkSmartPointer<vtkQuad> quad2 = vtkSmartPointer<vtkQuad>::New();
    vtkSmartPointer<vtkQuad> quad3 = vtkSmartPointer<vtkQuad>::New();
    vtkSmartPointer<vtkQuad> quad4 = vtkSmartPointer<vtkQuad>::New();
    vtkSmartPointer<vtkQuad> quad5 = vtkSmartPointer<vtkQuad>::New();

    // Face with normal along +Z
    quad0->GetPointIds()->SetId(0, fpid0);
    quad0->GetPointIds()->SetId(1, fpid1);
    quad0->GetPointIds()->SetId(2, fpid2);
    quad0->GetPointIds()->SetId(3, fpid3);

    // Face with normal along -Z
    quad1->GetPointIds()->SetId(0, fpid4);
    quad1->GetPointIds()->SetId(1, fpid7);
    quad1->GetPointIds()->SetId(2, fpid6);
    quad1->GetPointIds()->SetId(3, fpid5);

    // Face with normal along +X
    quad2->GetPointIds()->SetId(0, fpid2);
    quad2->GetPointIds()->SetId(1, fpid6);
    quad2->GetPointIds()->SetId(2, fpid7);
    quad2->GetPointIds()->SetId(3, fpid3);

    // Face with normal along -X
    quad3->GetPointIds()->SetId(0, fpid0);
    quad3->GetPointIds()->SetId(1, fpid4);
    quad3->GetPointIds()->SetId(2, fpid5);
    quad3->GetPointIds()->SetId(3, fpid1);

    // Face with normal along +Y
    quad4->GetPointIds()->SetId(0, fpid0);
    quad4->GetPointIds()->SetId(1, fpid3);
    quad4->GetPointIds()->SetId(2, fpid7);
    quad4->GetPointIds()->SetId(3, fpid4);

    // Face with normal along +Y
    quad5->GetPointIds()->SetId(0, fpid1);
    quad5->GetPointIds()->SetId(1, fpid5);
    quad5->GetPointIds()->SetId(2, fpid6);
    quad5->GetPointIds()->SetId(3, fpid2);

    quads->InsertNextCell(quad0);
    quads->InsertNextCell(quad1);
    quads->InsertNextCell(quad2);
    quads->InsertNextCell(quad3);
    quads->InsertNextCell(quad4);
    quads->InsertNextCell(quad5);
}

uint* generateRandomVoxelization(uint3 resolution)
{
    if ((resolution.x % 32) != 0)
    {
        cout << "The resolution must be divisible by 32!";
        return 0;
    }

    uint nrOfInts = resolution.z * resolution.y * (resolution.x / 32);
    uint* voxels = new uint[nrOfInts];
    srand(time(NULL));
    cout << "Max size of rand is " << RAND_MAX << "\n";
    cout << "Max size of uint is " << UINT_MAX << "\n\n";
    for (uint i = 0; i < nrOfInts; i++)
    {
        voxels[i] = compose32BitIntFromRands();
    }

    return voxels;
}

uint* generateSpecificVoxelization(uint3 resolution, uint minX, uint maxX, uint minY, uint maxY, uint minZ, uint maxZ)
{
    uint ipx = resolution.x / 32;
    uint ipy = resolution.y * ipx;
    uint maxIdx = resolution.z * ipy;
    uint* voxels = new uint[maxIdx];
    for (uint i = 0; i < maxIdx; i++)
    {
        uint x = i % ipx;
        uint y = (i % ipy) / ipx;
        uint z = i / ipy;

        if ((x >= uint(minX / 32)) && (x <= uint(maxX / 32)))
        {
            if ((y >= minY) && (y <= maxY))
            {
                if ((z >= minZ) && (z <= maxZ))
                {
                    uint value = UINT_MAX;
                    if (x == uint(minX / 32))
                    {
                        value ^= UINT_MAX << (32 - minX + x * 32);
                    }
                    if (x == uint(maxX / 32))
                    {
                        value ^= UINT_MAX >> (maxX + 1 - x * 32);
                    }
                    voxels[i] = value;
                }
                else
                    voxels[i] = 0;
            }
            else
                voxels[i] = 0;
        }
        else
            voxels[i] = 0;
    }
    return voxels;
}

uint compose32BitIntFromRands()
{
    uint i1 = uint(rand() % 0x100);
    uint i2 = uint(rand() % 0x100) << 8;
    uint i3 = uint(rand() % 0x100) << 16;
    uint i4 = uint(rand() % 0x100) << 24;

    return i1 + i2 + i3 + i4;
}

template <class Node>
void testMaterials(Node const * nodes, float const * vertices, uint const * indices, uint nrOfTriangles, int x, int y, int z, uint3 resolution, float voxelLength)
{
    uint nodeIdx = resolution.x * resolution.y * z + resolution.x * y + x;

    Node node = nodes[nodeIdx];

    cout << "Node at (" << x << ", " << y << ", " << z << ") has bid " << uint(node.bid()) << " and mat " << uint(node.mat());
    if (Node::hasRatio())
        cout << " and ratio " << node.r() << "\n";
    else
        cout << "\n";

    for (uint i = 0; i < nrOfTriangles; i++)
    {
        if (1 + (i % 255) == node.mat())
        {
            float3 tri_v1 = make_float3(vertices[3*indices[3*i + 0] + 0], vertices[3*indices[3*i + 0] + 1], vertices[3*indices[3*i + 0] + 2]);
            float3 tri_v2 = make_float3(vertices[3*indices[3*i + 1] + 0], vertices[3*indices[3*i + 1] + 1], vertices[3*indices[3*i + 1] + 2]);
            float3 tri_v3 = make_float3(vertices[3*indices[3*i + 2] + 0], vertices[3*indices[3*i + 2] + 1], vertices[3*indices[3*i + 2] + 2]);

            cout << "Triangle " << i << " vertex 1: (" << tri_v1.x << ", " << tri_v1.y << ", " << tri_v1.z << ")\n";
            cout << "Triangle " << i << " vertex 2: (" << tri_v2.x << ", " << tri_v2.y << ", " << tri_v2.z << ")\n";
            cout << "Triangle " << i << " vertex 3: (" << tri_v3.x << ", " << tri_v3.y << ", " << tri_v3.z << ")\n";
        }
    }
}

template <class Node>
void testMaterialsLoop(Node const * nodes, float const * vertices, uint const * indices, uint nrOfTriangles, uint3 resolution, float voxelLength)
{
    bool looping = true;
    int x = -1, y = -1, z = -1;

    while (looping)
    {
        cout << "Type in three coordinates sequentially, or a negative value to quit:\n";
        cout << "X: ";
        cin >> x;
        if (x < 0)
        {
            looping = false;
            continue;
        }
        cout << "Y: ";
        cin >> y;
        if (y < 0)
        {
            looping = false;
            continue;
        }
        cout << "Z: ";
        cin >> z;
        if (z < 0)
        {
            looping = false;
            continue;
        }

        testMaterials<Node>(nodes, vertices, indices, nrOfTriangles, x, y, z, resolution, voxelLength);
        x = -1;
        y = -1;
        z = -1;
    }
}

template <typename T>
struct ArrayDestroyer
{
    void operator()( T * obj )
    {
        std::cout << "DELETE\n";
        delete[] obj;
    }
};

template <class T>
class SmartArray : public std::unique_ptr<T, decltype(ArrayDestroyer<T>())> {};


auto loadModel
    ( const vtkSmartPointer<vtkPolyData> & model
    , uint & nrOfVertices
    , uint & nrOfTriangles
    ) 
    -> std::pair<std::unique_ptr<float[]>, std::unique_ptr<uint[]>>
{
    // Get triangles with indices (polys) and the vertices (points).
    vtkCellArray* polys = model->GetPolys();
    vtkPoints* points = model->GetPoints();

    nrOfVertices = points->GetNumberOfPoints();
    nrOfTriangles = polys->GetNumberOfCells();

    // Allocate vertex and index arrays in main memory.

    std::unique_ptr<float[]> vertices( new float[3 * nrOfVertices] );
    std::unique_ptr<uint[]> indices( new uint[3 * nrOfTriangles] );

    double vert[3];

    // Copy vertex data to the array.
    for (vtkIdType i = 0; i < points->GetNumberOfPoints(); i++)
    {
        points->GetPoint(i, vert);

        for (int j = 0; j < 3; j++)
            vertices[3*i + j] = (float)vert[j];
    }

    // Copy index data to the array.
    vtkIdList* idlist = vtkIdList::New();
    uint currentPoly = 0;
    polys->InitTraversal();
    while(polys->GetNextCell(idlist))
    {
        for (vtkIdType i = 0; i < idlist->GetNumberOfIds(); i++)
        {
            indices[3*currentPoly + i] = idlist->GetId(i);
        }
        currentPoly++;
    }

    return std::make_pair( std::move( vertices ), std::move( indices ) );
}

void testCrossproduct()
{
    float3 v0 = make_float3( 0.0, 1.0, 2.0);
    float3 v1 = make_float3( 2.0, 3.0, 4.0);
    float3 v2 = make_float3( 4.0, 2.0, 1.0);

    float3 n0, n1, n2, n3, n4, n5;

    cout << "Calculating the cross products in various ways:\n";

    n0 = cross(v0 - v2, v1 - v0);
    n1 = cross(v1 - v0, v2 - v1);
    n2 = cross(v2 - v1, v0 - v2);

    n3 = cross(v1 - v0, v2 - v1);
    n4 = cross(v2 - v1, v0 - v2);
    n5 = cross(v0 - v2, v1 - v0);

    cout << "X: " << n0.x << ", Y: " << n0.y << ", Z: " << n0.z << ".\n";
    cout << "X: " << n1.x << ", Y: " << n1.y << ", Z: " << n1.z << ".\n";
    cout << "X: " << n2.x << ", Y: " << n2.y << ", Z: " << n2.z << ".\n";
    cout << "X: " << n3.x << ", Y: " << n3.y << ", Z: " << n3.z << ".\n";
    cout << "X: " << n4.x << ", Y: " << n4.y << ", Z: " << n4.z << ".\n";
    cout << "X: " << n5.x << ", Y: " << n5.y << ", Z: " << n5.z << ".\n";
}

template <class Node>
uint numberOfNodesWithNoMaterial(Node* nodes, uint3 resolution)
{
    uint count = 0;
    Node node;
    for ( uint i = 0; i < resolution.x * resolution.y * resolution.z; i++ )
    {
        node = nodes[i];

        if ( Node::isFCCNode() )
        {
            if ( node.bid() == 0 || node.bid() == 4095 )
                continue;
        }
        else
        {
            if ( node.bid() == 0 || node.bid() == 27 )
                continue;
        }

        if ( node.mat() == 0 )
            count++;
    }
    return count;
}

void testTrianglePlaneIntersection()
{
    float3 triangle[3], triNormal, modelBBMin;
    float d;
    int3 voxel;
    
    voxel = make_int3(1, 1, 1);
    d = 10.0f;

    triangle[0] = make_float3(7.0f, 15.0f, 5.0f);
    triangle[2] = make_float3(17.0f, 15.0f, 15.0f);
    triangle[1] = make_float3(7.0f, 5.0f, 5.0f);

    modelBBMin = make_float3(0.0f, 0.0f, 0.0f);

    triNormal = normalize(cross(triangle[0] - triangle[2], triangle[1] - triangle[0]));

    //testTriangleIntersections(triangle, voxel, triNormal, modelBBMin, d);
}

template <class Node>
void printOrientationDistribution(Node* nodes, uint3 resolution)
{
    Node node;
    uint endIdx = resolution.x * resolution.y * resolution.z;
    uint orientations[28];
    uint bid;

    for (int i = 0; i < 28; ++i)
        orientations[i] = 0;

    for (uint n = 0; n < endIdx; ++n)
    {
        node = nodes[n];
        bid = node.bid();
        if (bid > 27)
            continue;
        orientations[bid] += 1;
    }

    for (int i = 0; i < 28; ++i)
        cout << "Orientation " << i << ": " << orientations[i] << " nodes.\n";
}

template <class Node>
void analyzeRatioNodes(Node* nodes, uint3 resolution, float voxelLength, int nthErroneousRatio)
{
    if (!Node::hasRatio())
        return;

    Node node;

    uint nrOfFullVoxels = 0;
    uint nrOfEmptyVoxels = 0;
    uint nrOfPartialVoxels = 0;
    uint nrOfFullBoundaryVoxels = 0;
    uint nrOfEmptyBoundaryVoxels = 0;
    uint nrOfPartialBoundaryVoxels = 0;
    uint nrOfBoundaryVoxels = 0;
    uint nrOfFullAirVoxels = 0;
    uint nrOfEmptyAirVoxels = 0;
    uint nrOfPartialAirVoxels = 0;
    uint nrOfAirVoxels = 0;
    uint nrOfFullNonSolidVoxels = 0;
    uint nrOfEmptyNonSolidVoxels = 0;
    uint nrOfPartialNonSolidVoxels = 0;
    uint nrOfNonSolidVoxels = 0;
    uint nrOfFullMaterialVoxels = 0;
    uint nrOfEmptyMaterialVoxels = 0;
    uint nrOfPartialMaterialVoxels = 0;
    uint nrOfMaterialVoxels = 0;
    uint nrOfFullNonMaterialVoxels = 0;
    uint nrOfEmptyNonMaterialVoxels = 0;
    uint nrOfPartialNonMaterialVoxels = 0;
    uint nrOfNonMaterialVoxels = 0;
    uint nrOfErroneousRatios = 0;
    

    for (uint i = 0; i < resolution.x * resolution.y * resolution.z; i++)
    {
        node = nodes[i];

        if (node.r() == 0.0f)
        {
            nrOfEmptyVoxels++;

            if (node.bid() == 0)
            {
                nrOfNonSolidVoxels++;
                nrOfEmptyNonSolidVoxels++;
            }
            else if (node.bid() == 27)
            {
                nrOfAirVoxels++;
                nrOfEmptyAirVoxels++;
            }
            else
            {
                nrOfBoundaryVoxels++;
                nrOfEmptyBoundaryVoxels++;
            }

            if (node.mat() != 0)
            {
                nrOfMaterialVoxels++;
                nrOfEmptyMaterialVoxels++;
            }
            else
            {
                nrOfNonMaterialVoxels++;
                nrOfEmptyNonMaterialVoxels++;
            }
        }
        else if (node.r() == 1.0f)
        {
            nrOfFullVoxels++;

            if (node.bid() == 0)
            {
                nrOfNonSolidVoxels++;
                nrOfFullNonSolidVoxels++;
            }
            else if (node.bid() == 27)
            {
                nrOfAirVoxels++;
                nrOfFullAirVoxels++;
            }
            else
            {
                nrOfBoundaryVoxels++;
                nrOfFullBoundaryVoxels++;
            }

            if (node.mat() != 0)
            {
                nrOfMaterialVoxels++;
                nrOfFullMaterialVoxels++;
            }
            else
            {
                nrOfNonMaterialVoxels++;
                nrOfFullNonMaterialVoxels++;
            }
        }
        else if ((node.r() > 0.0f) && (node.r() < 1.0f))
        {
            nrOfPartialVoxels++;

            if (node.bid() == 0)
            {
                nrOfNonSolidVoxels++;
                nrOfPartialNonSolidVoxels++;
            }
            else if (node.bid() == 27)
            {
                nrOfAirVoxels++;
                nrOfPartialAirVoxels++;
            }
            else
            {
                nrOfBoundaryVoxels++;
                nrOfPartialBoundaryVoxels++;
            }

            if (node.mat() != 0)
            {
                nrOfMaterialVoxels++;
                nrOfPartialMaterialVoxels++;
            }
            else
            {
                nrOfNonMaterialVoxels++;
                nrOfPartialNonMaterialVoxels++;
            }
        }
        else
        {
            if (nrOfErroneousRatios == nthErroneousRatio)
            {
                uint x = i % resolution.x;
                uint y = (i % (resolution.x * resolution.y)) / resolution.x;
                uint z = i / (resolution.x * resolution.y);

                cout << "Found erroneous ratio at index: (" << x << ", " << y << ", " << z << ")\n";
            }
            nrOfErroneousRatios++;
        }
    }

    cout << "nrOfFullVoxels: " << nrOfFullVoxels << "\n";
    cout << "nrOfEmptyVoxels: " << nrOfEmptyVoxels << "\n";
    cout << "nrOfPartialVoxels: " << nrOfPartialVoxels << "\n";
    cout << "nrOfFullBoundaryVoxels: " << nrOfFullBoundaryVoxels << "\n";
    cout << "nrOfEmptyBoundaryVoxels: " << nrOfEmptyBoundaryVoxels << "\n";
    cout << "nrOfPartialBoundaryVoxels: " << nrOfPartialBoundaryVoxels << "\n";
    cout << "nrOfBoundaryVoxels: " << nrOfBoundaryVoxels << "\n";
    cout << "nrOfFullAirVoxels: " << nrOfFullAirVoxels << "\n";
    cout << "nrOfEmptyAirVoxels: " << nrOfEmptyAirVoxels << "\n";
    cout << "nrOfPartialAirVoxels: " << nrOfPartialAirVoxels << "\n";
    cout << "nrOfAirVoxels: " << nrOfAirVoxels << "\n";
    cout << "nrOfFullNonSolidVoxels: " << nrOfFullNonSolidVoxels << "\n";
    cout << "nrOfEmptyNonSolidVoxels: " << nrOfEmptyNonSolidVoxels << "\n";
    cout << "nrOfPartialNonSolidVoxels: " << nrOfPartialNonSolidVoxels << "\n";
    cout << "nrOfNonSolidVoxels: " << nrOfNonSolidVoxels << "\n";
    cout << "nrOfFullMaterialVoxels: " << nrOfFullMaterialVoxels << "\n";
    cout << "nrOfEmptyMaterialVoxels: " << nrOfEmptyMaterialVoxels << "\n";
    cout << "nrOfPartialMaterialVoxels: " << nrOfPartialMaterialVoxels << "\n";
    cout << "nrOfMaterialVoxels: " << nrOfMaterialVoxels << "\n";

    cout << "nrOfFullNonMaterialVoxels: " << nrOfFullNonMaterialVoxels << "\n";
    cout << "nrOfEmptyNonMaterialVoxels: " << nrOfEmptyNonMaterialVoxels << "\n";
    cout << "nrOfPartialNonMaterialVoxels: " << nrOfPartialNonMaterialVoxels << "\n";
    cout << "nrOfNonMaterialVoxels: " << nrOfNonMaterialVoxels << "\n";

    cout << "nrOfErroneousRatios: " << nrOfErroneousRatios << "\n";
}

template <class Node>
Node* stitchNodeArrays
    ( std::vector<vox::NodePointer<Node>> const &nps
    , uint2 splits
    )
{
    size_t partitions = nps.size();

    if (partitions <= 0)
        return NULL;
    if (partitions == 1)
        return nps.at(0).ptr;

    uint3 res = fetchResolution( nps, splits );

    Node* result = new Node[res.x * res.y * res.z];

    for ( auto it = nps.begin(); it < nps.end(); ++it ) {
        uint3 subRes = it->dim;

        for ( uint i = 0; i < subRes.x * subRes.y * subRes.z; ++i ) {
            uint x = i % subRes.x;
            uint y = (i % (subRes.x * subRes.y)) / subRes.x;
            uint z = i / (subRes.x * subRes.y);

            if (x == 0 || x == subRes.x - 1)
                continue;
            if (y == 0 || y == subRes.y - 1)
                continue;
            if (z == 0 || z == subRes.z - 1)
                continue;

            uint xp = 1 + x;
            uint yp = 1 + it->loc.y * ( it->dim.y - 2 ) + y;
            uint zp = 1 + it->loc.z * ( it->dim.z - 2 ) + z;

            uint n = res.x * res.y * zp + res.x * yp + xp;
            result[n] = it->ptr[i];
        }
    }

    return result;
}

template <class Node>
std::vector<vox::NodePointer<Node>> createTestNodes(uint2 splits)
{
    std::vector<vox::NodePointer<Node>> result;

    vox::NodePointer<Node> nptr;
    Node* nodes;
    for (uint j = 0; j < splits.x; ++j) {
        for (uint k = 0; k < splits.y; ++k) {
            nodes = new Node[1000];
            for (int i = 0; i < 1000; ++i)
                nodes[i].set(1, 1);

            nptr.ptr = nodes;
            nptr.bb.min = make_uint3( 0, 10 * j, 10 * k );
            nptr.bb.max = make_uint3( 10, 10 * (j+1), 10 * (k+1) );
            nptr.dev = k * splits.x + j;

            result.push_back(nptr);
        }
    }

    return result;
}

template <class Node>
uint3 fetchResolution(
    std::vector<vox::NodePointer<Node>> const & nps,
    uint2 splits )
{
    vox::Bounds<uint3> bb = { make_uint3(UINT_MAX), make_uint3(0) };

    uint3 res = { nps[0].dim.x, 0, 0 };

    for ( auto it = nps.begin(); it != nps.end(); ++it )
        res += make_uint3( 0, it->dim.y - 2, it->dim.z - 2 );

    res.y /= splits.y; res.y += 2;
    res.z /= splits.x; res.z += 2;

    return res;
}

template <class Node>
void compareNodeArrays(Node const * nodes1, Node const * nodes2, uint3 res)
{
    uint nrOfNodes = res.x * res.y * res.z;
    Node n1, n2;
    uint bidMismatches = 0;
    uint matMismatches = 0;
    for (uint i = 0; i < nrOfNodes; ++i) {
        n1 = nodes1[i];
        n2 = nodes2[i];

        if (n1.bid() != n2.bid()) {
            uint x = i % res.x;
            uint y = (i % (res.x * res.y)) / res.x;
            uint z = i / (res.x * res.y);

            //cout << "Node: (" << x << ", " << y << ", " << z << ") -- BIDs don't match!\n";

            bidMismatches++;
        }
        if (n1.mat() != n2.mat()) {
            uint x = i % res.x;
            uint y = (i % (res.x * res.y)) / res.x;
            uint z = i / (res.x * res.y);

            //cout << "Node: (" << x << ", " << y << ", " << z << ") -- MATs don't match!\n";

            matMismatches++;
        }
    }

    std::cout << "\n";
    std::cout << "Found " << bidMismatches << " nodes with mismatching BIDs.\n";
    std::cout << "Found " << matMismatches << " nodes with mismatching MATs.\n";
}

template <class Node>
void applySlice( Node * slice
               , uint sliceNr
               , Node * nodes
               , uint3 res
               , uint direction )
{
    if ( Node::isFCCNode() )
    {
        if ( direction == 0 )
        {
            uint s = 2;
            uint n = sliceNr * 2;

            while ( s < 6 * res.y * res.z )
            {
                nodes[n] = slice[s];
                nodes[n + 1] = slice[s + 1];

                n += res.x;
                s += 6;
            }
        }
        else if ( direction == 1 )
        {
            uint s = res.x;
            uint n = sliceNr * res.x;

            while ( s < res.x * 3 * res.z )
            {
                for ( uint t = 0; t < res.x; ++t )
                    nodes[n + t] = slice[s + t];

                s += 3 * res.x;
                n += res.y * res.x;
            }
        }
        else if ( direction == 2 )
        {
            uint s = 2 * res.x * res.y;
            uint n = 2 * sliceNr * res.x * res.y;

            for ( uint t = 0; t < 2 * res.x * res.y; ++t )
                nodes[n + t] = slice[s + t];
        }
    }
    else
    {
        if ( direction == 0 )
        {
            uint s = 1;
            uint n = sliceNr;

            while ( s < 3 * res.y * res.z )
            {
                nodes[n] = slice[s];

                n += res.x;
                s += 3;
            }
        }
        else if ( direction == 1 )
        {
            uint s = res.x;
            uint n = sliceNr * res.x;

            while ( s < res.x * 3 * res.z )
            {
                for ( uint t = 0; t < res.x; ++t )
                    nodes[n + t] = slice[s + t];

                s += 3 * res.x;
                n += res.y * res.x;
            }
        }
        else if ( direction == 2 )
        {
            uint s = res.x * res.y;
            uint n = sliceNr * res.x * res.y;

            for ( uint t = 0; t < res.x * res.y; ++t )
                nodes[n + t] = slice[s + t];
        }
    }
}

template <class NT>
void renderFCCNodeOutput(NT* nodes, bool onlySurface, bool materials, uint3 res)
{
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

    // Colors
    unsigned char black[3] = { 0, 0, 0 };
    unsigned char red[3] = { 255, 0, 0 };
    unsigned char green[3] = { 0, 255, 0 };
    unsigned char blue[3] = { 0, 0, 255 };
    unsigned char yellow[3] = { 255, 255, 0 };
    unsigned char purple[3] = { 255, 0, 255 };
    unsigned char aquamarine[3] = { 0, 255, 255 };
    unsigned char grey[3] = { 127, 127, 127 };

    unsigned char white[3] = { 255, 255, 255 };

    unsigned char * colorList[8] = { white, red, green, blue, yellow, purple, aquamarine, grey };

    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);
    colors->SetName("Color");

    for (uint nodeIdx = 0; nodeIdx < res.x * res.y * res.z; nodeIdx++)
    {
        if (onlySurface)
        {
            // Ignore air nodes.
            if (nodes[nodeIdx].bid() == 4095)
                continue;
        }

        if (materials)
        {
            // Ignore nodes with zero material.
            if (nodes[nodeIdx].mat() == 0)
                continue;
        }
        
        // Ignore non-solid nodes.
        if (nodes[nodeIdx].bid() == 0)
            continue;
        

        uint3 m = make_uint3( nodeIdx % res.x
                            , (nodeIdx % (res.x * res.y)) / res.x
                            , nodeIdx / (res.x * res.y) );

         m.y += m.y + (m.x + m.z) % 2;

        points->InsertNextPoint(double(m.x), double(m.y), double(m.z));

        if (materials)
            colors->InsertNextTupleValue(colorList[nodes[nodeIdx].mat()]);
        else
            colors->InsertNextTupleValue(white);
    }

    vtkSmartPointer<vtkPolyData> data = vtkSmartPointer<vtkPolyData>::New();
    data->SetPoints(points);

    vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    vertexFilter->SetInputConnection(data->GetProducerPort());
    vertexFilter->Update();

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->ShallowCopy(vertexFilter->GetOutput());

    polydata->GetPointData()->SetScalars(colors);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInput(polydata);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetPointSize(1);

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderer->AddActor(actor);

    renderWindow->Render();
    renderWindowInteractor->Start();
}

template <class Node>
void initTests
    ( const boost::program_options::variables_map & vm
    , const vtkSmartPointer<vtkPolyData> & polys 
    )
{
    uint 
        nrOfTriangles = 0, 
        nrOfVertices = 0, 
        nrOfUniqueMaterials = 0;

    auto modelData = loadModel( polys, nrOfVertices, nrOfTriangles );
    
    vox::Voxelizer<Node> voxelizer( modelData.first.get()
                                  , modelData.second.get()
                                  , nrOfVertices
                                  , nrOfTriangles );

    if ( vm.count( "verbose" ) )
        voxelizer.verboseOutput( true );

    modelData.first.reset();
    modelData.second.reset();

    if ( vm.count( "materials" ) )
    {
        auto materials = vox::make_unique<vox::uchar>( nrOfTriangles );
        nrOfUniqueMaterials = 8;

        for ( uint i = 0; i < nrOfTriangles; i++ )
            materials[i] = vox::uchar(1 + (i % 7));

        voxelizer.setMaterials( materials.get(), nrOfUniqueMaterials );
        voxelizer.setMaterialOutput( true );
    }

    if ( vm.count( "voxels" ) )
    {
        performPlainVoxelTest( voxelizer, vm );
    }
    else if ( vm.count( "slice" ) )
    {
        performSliceTest( voxelizer, vm );
    }
    else if ( vm.count( "allSlice" ) )
    {
        performAllSliceTest( voxelizer, vm );
    }
    else
    {
        performNodeTest( voxelizer, vm );
    }
}

template<class Node>
void performPlainVoxelTest
    ( vox::Voxelizer<Node> & voxelizer
    , const boost::program_options::variables_map & vm
    )
{
    std::vector<vox::NodePointer<Node>> result;

    uint3 voxSplitRes = { 1024, 512, 512 };
    if ( vm.count( "voxRes" ) )
    {
        auto newVoxSplitRes = vm[ "voxRes" ].as<std::vector<uint>>();
        voxSplitRes.x = newVoxSplitRes.at(0);
        voxSplitRes.y = newVoxSplitRes.at(1);
        voxSplitRes.z = newVoxSplitRes.at(2);
    }

    if ( vm.count( "resolution" ) )
    {
        auto resolution = vm[ "resolution" ].as<uint>();

        if ( vm.count( "multiDevice" ) )
        {
            auto dc = vm[ "multiDevice" ].as<std::vector<uint>>();
            uint2 devConfig = { dc[0], dc[1] };

            result = voxelizer.simulateMultidevice(
                [&] () { return voxelizer.voxelize( resolution
                                                  , devConfig
                                                  , voxSplitRes ); } );
        }
        else
            result.push_back( voxelizer.voxelizeToRAM( resolution
                                                     , voxSplitRes ) );
    }
    else if ( vm.count( "distance" ) )
    {
        auto distance = vm[ "distance" ].as<double>();

        result.push_back( voxelizer.voxelizeToRAM( distance
                                                 , voxSplitRes ) );
    }

    bool renderSurfaceOnly = vm.count( "surface" ) ? true : false;

    std::cout << "Size of result: " << result.size() << "\n";

    for ( auto it = result.begin(); it != result.end(); ++it )
    {
        renderVoxelization( it->vptr, renderSurfaceOnly, it->dim );
        delete[] it->vptr;
    }
}

template<class Node>
void performSliceTest
    ( vox::Voxelizer<Node> & voxelizer
    , const boost::program_options::variables_map & vm
    )
{
    vox::NodePointer<Node> result;

    uint2 voxSplitRes = { 1024, 512 };
    if ( vm.count( "voxRes" ) )
    {
        auto newVoxSplitRes = vm[ "voxRes" ].as<std::vector<uint>>();
        voxSplitRes.x = newVoxSplitRes.at(0);
        voxSplitRes.y = newVoxSplitRes.at(1);
    }
    uint2 matSplitRes = { 1024, 512 };
    if ( vm.count( "matRes" ) )
    {
        auto newMatSplitRes = vm[ "matRes" ].as<std::vector<uint>>();
        matSplitRes.x = newMatSplitRes.at(0);
        matSplitRes.y = newMatSplitRes.at(1);
    }

    auto sliceOptions = vm[ "slice" ].as<std::vector<uint>>();
    int direction = int( sliceOptions.at(0) );
    uint slice = sliceOptions.at(1);

    std::cout << "Slice options: Direction: " << direction << ", slice number"
        ": " << slice << "\n";

    if ( vm.count( "resolution" ) )
    {
        auto resolution = vm[ "resolution" ].as<uint>();
        result = voxelizer.voxelizeSliceToRAM( resolution
                                             , direction
                                             , slice
                                             , voxSplitRes
                                             , matSplitRes );
    }
    else if ( vm.count( "distance" ) )
    {
        auto distance = vm[ "distance" ].as<double>();

        result = voxelizer.voxelizeSliceToRAM( distance
                                             , direction
                                             , slice
                                             , voxSplitRes
                                             , matSplitRes );
    }

    bool renderSurfaceOnly = false;
    if ( vm.count( "surface" ) )
        renderSurfaceOnly = true;

    bool renderMaterials = false;
    if ( vm.count( "materials" ) )
        renderMaterials = true;

    std::cout << "Dimensions of grid: " << printUint3( result.dim );

    std::cout << "Successfully voxelized -- entering renderNodeOutput.\n";

    if ( Node::isFCCNode() )
        renderFCCNodeOutput( result.ptr
                           , renderSurfaceOnly
                           , renderMaterials
                           , result.dim );
    else
        renderNodeOutput( result.ptr
                        , renderSurfaceOnly
                        , renderMaterials
                        , result.dim );

    if ( vm.count( "materials" ) )
    {
        uint badNodes = numberOfNodesWithNoMaterial( result.ptr, result.dim );
        std::cout << "Found " << badNodes << " boundary nodes with no material"
            "\n";
    }

    delete[] result.ptr;
}

template<class Node>
void performNodeTest
    ( vox::Voxelizer<Node> & voxelizer
    , const boost::program_options::variables_map & vm
    )
{
    std::vector<vox::NodePointer<Node>> result;

    uint3 voxSplitRes = { 1024, 512, 512 };
    if ( vm.count( "voxRes" ) )
    {
        auto newVoxSplitRes = vm[ "voxRes" ].as<std::vector<uint>>();
        voxSplitRes.x = newVoxSplitRes.at(0);
        voxSplitRes.y = newVoxSplitRes.at(1);
        voxSplitRes.z = newVoxSplitRes.at(2);
    }
    uint3 matSplitRes = { 1024, 512, 512 };
    if ( vm.count( "matRes" ) )
    {
        auto newMatSplitRes = vm[ "matRes" ].as<std::vector<uint>>();
        matSplitRes.x = newMatSplitRes.at(0);
        matSplitRes.y = newMatSplitRes.at(1);
        matSplitRes.z = newMatSplitRes.at(2);
    }

    uint2 devConf = { 1, 1 };

    if ( vm.count( "resolution" ) )
    {
        auto resolution = vm[ "resolution" ].as<uint>();

        if ( vm.count( "multiDevice" ) )
        {
            auto dc = vm[ "multiDevice" ].as<std::vector<uint>>();
            devConf.x = dc[0];
            devConf.y = dc[1];

            result = voxelizer.simulateMultidevice( 
                [&] () { return voxelizer.voxelizeToNodes( resolution
                                                         , devConf
                                                         , voxSplitRes
                                                         , matSplitRes ); 
                } );
        }
        else
            result.push_back( voxelizer.voxelizeToNodesToRAM( resolution
                                                            , voxSplitRes
                                                            , matSplitRes ) );
    }
    else if ( vm.count( "distance" ) )
    {
        auto distance = vm[ "distance" ].as<double>();

        if ( vm.count( "multiDevice" ) )
        {
            auto dc = vm[ "multiDevice" ].as<std::vector<uint>>();
            devConf.x = dc[0];
            devConf.y = dc[1];

            result = voxelizer.simulateMultidevice( 
                [&] () { return voxelizer.voxelizeToNodes( distance
                                                         , devConf
                                                         , voxSplitRes
                                                         , matSplitRes ); 
                } );
        }
        else
            result.push_back( voxelizer.voxelizeToNodesToRAM( distance
                                                            , voxSplitRes
                                                            , matSplitRes ) );
    }

    bool renderSurfaceOnly = vm.count( "surface" ) ? true : false;
    bool renderMaterials = vm.count( "materials" ) ? true : false;

    uint3 res = vm.count( "multiDevice" ) ? fetchResolution( result, devConf )
                                          : result[0].dim;
    auto nodes = std::unique_ptr<Node>( 
        vm.count( "multiDevice" ) ? stitchNodeArrays( result, devConf ) 
                                  : result[0].ptr );

    if ( Node::isFCCNode() )
        renderFCCNodeOutput( nodes.get()
                           , renderSurfaceOnly
                           , renderMaterials
                           , res );
    else
        renderNodeOutput( nodes.get()
                        , renderSurfaceOnly
                        , renderMaterials
                        , res );

    if ( vm.count( "materials" ) )
    {
        uint badNodes = numberOfNodesWithNoMaterial( nodes.get(), res );
        std::cout << "Found " << badNodes << " boundary nodes with no material"
            "\n";
    }
}

template<class Node>
void performAllSliceTest
    ( vox::Voxelizer<Node> & voxelizer
    , const boost::program_options::variables_map & vm
    )
{
    uint2 voxSplitRes = { 1024, 512 };
    if ( vm.count( "voxRes" ) )
    {
        auto newVoxSplitRes = vm[ "voxRes" ].as<std::vector<uint>>();
        voxSplitRes.x = newVoxSplitRes.at(0);
        voxSplitRes.y = newVoxSplitRes.at(1);
    }
    uint2 matSplitRes = { 1024, 512 };
    if ( vm.count( "matRes" ) )
    {
        auto newMatSplitRes = vm[ "matRes" ].as<std::vector<uint>>();
        matSplitRes.x = newMatSplitRes.at(0);
        matSplitRes.y = newMatSplitRes.at(1);
    }

    vox::NodePointer<Node> result;
    const auto direction = vm[ "allSlice" ].as<int>();
    Node * nodes = nullptr;
    uint3 res;

    if ( vm.count( "resolution" ) )
    {
        const auto resolution = vm[ "resolution" ].as<uint>();
        res = voxelizer.getArrayDimensions( resolution
                                          , voxSplitRes.x
                                          , direction == 0 );

        nodes = new Node[ res.x * res.y * res.z ];

        const uint maxSlice = direction == 0 ? res.x :
                              direction == 1 ? res.y : res.z;

        std::cout << "Dimensions of big grid: " << printUint3( res );

        for ( uint slice = 0; slice < maxSlice; ++slice )
        {
            result = voxelizer.voxelizeSliceToRAM( resolution
                                                 , direction
                                                 , slice
                                                 , voxSplitRes
                                                 , matSplitRes );

            applySlice( result.ptr, slice, nodes, res, direction );

            delete[] result.ptr;
        }

        std::cout << "Dimensions of big grid: " << printUint3( res );
    }

    if ( vm.count( "distance" ) )
    {
        const auto distance = vm[ "distance" ].as<double>();
        res = voxelizer.getArrayDimensions( distance
                                          , voxSplitRes.x
                                          , direction == 0 );

        nodes = new Node[ res.x * res.y * res.z ];

        uint maxSlice = 0;
        if ( Node::isFCCNode() )
        {
            maxSlice = direction == 0 ? res.x / 2 :
                       direction == 1 ? res.y : res.z / 2;
        }
        else
        {
            maxSlice = direction == 0 ? res.x :
                       direction == 1 ? res.y : res.z;
        }

        std::cout << "maxSlice : " << maxSlice << "\n";

        std::cout << "Dimensions of big grid: " << printUint3( res );

        for ( uint slice = 0; slice < maxSlice; ++slice )
        {
            result = voxelizer.voxelizeSliceToRAM( distance
                                                 , direction
                                                 , slice
                                                 , voxSplitRes
                                                 , matSplitRes );

            applySlice( result.ptr, slice, nodes, res, direction );

            delete[] result.ptr;
        }
    }

    bool renderSurfaceOnly = false;
    if ( vm.count( "surface" ) )
        renderSurfaceOnly = true;

    bool renderMaterials = false;
    if ( vm.count( "materials" ) )
        renderMaterials = true;

    std::cout << "Successfully voxelized -- entering renderNodeOutput.\n";

    if ( Node::isFCCNode() )
        renderFCCNodeOutput( nodes
                           , renderSurfaceOnly
                           , renderMaterials
                           , res );
    else
        renderNodeOutput( nodes
                        , renderSurfaceOnly
                        , renderMaterials
                        , res );

    if ( vm.count( "materials" ) )
    {
        uint badNodes = numberOfNodesWithNoMaterial( nodes, res );
        std::cout << "Found " << badNodes << " boundary nodes with no material"
            "\n";
    }
}

void testFunction()
{
    std::vector<TestClass> vec;
    vec.reserve( 3 );

    std::cout << "Push_back:\n";
    vec.push_back( TestClass( 1 ) );

    std::cout << "Push_back:\n";
    vec.push_back( TestClass( 2 ) );

    std::cout << "Emplace_back:\n";
    vec.emplace_back( 3 );

    std::cout << "Allocate:\n";
    auto ptr = std::unique_ptr<TestClass>( new TestClass( 4 ) );

    vec.push_back( *ptr );

    std::cout << "Printing values:\n";
    std::for_each( vec.begin()
                 , vec.end()
                 , [] ( TestClass & t ) { std::cout << t.get() << "\n"; } );

    std::cout << ptr->get() << "\n";

    std::vector<TestClass> vec2;
    vec2.push_back( TestClass( 5 ) );
    vec2.push_back( TestClass( 6 ) );
    vec2.push_back( TestClass( 7 ) );

    std::cout << "Assigning vec2 to vec.\n";

    vec = vec2;

    std::for_each( vec.begin()
                 , vec.end()
                 , [] ( TestClass & t ) { std::cout << t.get() << "\n"; } );

    auto jack = std::unique_ptr<Printable>( new Person( "Jack" ) );
    auto jill = std::unique_ptr<Printable>( new Person( "Jill" ) );
    auto ride = std::unique_ptr<Printable>( new Car( "Opel Astra" ) );

    std::cout << jack->print();
    std::cout << jill->print();
    std::cout << ride->print();

    std::cout << "Exiting function.\n";
}

inline std::string printUint3( uint3 & vec )
{
    std::string result;

    std::string x = std::to_string( (long long) vec.x );
    std::string y = std::to_string( (long long) vec.y );
    std::string z = std::to_string( (long long) vec.z );

    result += "( " + x + ", " + y + ", " + z + " )\n";
    
    return std::move( result );
}

template <typename T>
void validate( std::string var
             , std::string msg
             , boost::program_options::variables_map & vm
             , std::function<bool( T & )> func )
{
    if ( vm.count( var ) )
    {
        T val = vm[ var ].as<T>();

        if ( !func(val) )
        {
            std::cout << "--" << var << " : " << msg << "\n";
            std::exit( 1 );
        }
    }
}