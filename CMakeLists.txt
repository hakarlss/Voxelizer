cmake_minimum_required( VERSION 2.8 )

project( Voxelizer )

# Collect all relevant files to be sent to the compilers.
# Also includes these files into the project for easy access.
# If new files are added, the project files need to be regenerated with cmake.
file( GLOB SOURCE_FILES . src/*.cpp src/*.cu )
file( GLOB HEADER_FILES . include/*.h )

# Limits the configuration types available. Doesn't seem to work, though...
set( CMAKE_CONFIGURATION_TYPES "Debug" "Release" )

# Boost options.
set( Boost_USE_STATIC_LIBS ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_USE_STATIC_RUNTIME OFF )
set( Boost_NO_BOOST_CMAKE ON )

# Require Boost >= 1.53 with system and thread components.
find_package( Boost 1.53 REQUIRED COMPONENTS system thread chrono date_time )

# If boost cannot be found, make a cache variable that allows the user to 
# specify the Boost root directory where the libraries and includes are found.
if ( Boost_FOUND )
	include_directories( ${Boost_INCLUDE_DIRS} ) 
else ()
	set( BOOST_ROOT "" CACHE PATH "Path to Boost root directory." )
endif ()

# Require CUDA >= 5.0
find_package( CUDA 5.0 REQUIRED )

option( GENERATE_DOXYGEN_DOCS "Enables or disables the generation of documentation with Doxygen." OFF )
option( GENERATE_TESTS "Adds the compilation of a test program as a target." OFF )

# Configuration-dependent settings.
set( CMAKE_DEBUG_POSTFIX "_d" )
set( CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};-gencode arch=compute_30,code=sm_30 )
set( CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-gencode arch=compute_30,code=sm_30 )

# C++ compiler flags
set( CMAKE_CXX_FLAGS_RELEASE "-std=c++0x ${CMAKE_CXX_FLAGS_RELEASE}" )

message( STATUS "R Flags: ${CMAKE_CXX_FLAGS_RELEASE}" )
	
# Include files from the subdirectories and process the CMakeLists.txt in those 
# directories.
include_directories( include )
add_subdirectory( src )

if ( GENERATE_DOXYGEN_DOCS )
	add_subdirectory( doc )
endif ()

if ( GENERATE_TESTS )
	add_subdirectory( tests )
endif ()
