Voxelizer
=========

A CUDA-based voxelizer used in acoustics FDTD calculations.

Dependencies: - CUDA Toolkit 5.0 or higher
			  - Boost 1.53 or higher
			  - (For the tester) VTK (Tested with version 5.10.1)
			  - (For the documentation) Doxygen

Installation instructions:

Windows:

The library compiles on Visual Studio 10. Compatibility with other versions is
unknown. It is designed to be compiled as a 64-bit binary, but there should not 
be anything preventing it from being compiled as a 32-bit binary.

Open the command prompt provided by Visual Studio or the Windows SDK to make 
sure the build environment is properly set up. Create a directory for the build 
files and enter the directory. Run cmake (with or without a GUI) from the 
command line and configure the dependencies.

You can enable: 

GENERATE_TESTS        to also compile a test application that uses the 
                      voxelizer.
GENERATE_DOXYGEN_DOCS to have Doxygen generate documentation.

As an example, the process I use to build the Voxelizer is the following:

<Open up the command prompt of the Windows 7.1 SDK>
setenv /Release
cd <build directory>
cmake -G "Visual Studio 10 Win64" -T "Windows7.1SDK" -D BOOST_ROOT="<path to boost root>" -D GENERATE_TESTS=ON <path to sources>
<Open up the generated VS project and build the voxelizer>

Linux:

Currently there are some issues with building on Linux that I haven't quite 
figured out yet. Undefined references pop up during the linking of the test 
application for some reason.

In any case, once I get it working it will need to be compiled with GCC 4.6 if 
CUDA toolkit is version 5.0.