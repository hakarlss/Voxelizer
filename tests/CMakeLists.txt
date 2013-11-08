# Collect all relevant files to be sent to the compilers.
# Also includes these files into the project for easy access.
# If new files are added, the project files need to be regenerated with cmake.
file( GLOB TEST_SOURCES . src/*.cpp )
file( GLOB TEST_HEADERS . include/*.h )

link_directories( ${CMAKE_CURRENT_BINARY_DIR}/../lib ) 
find_package( VTK REQUIRED )

# Require Boost >= 1.53 with system and thread components.
find_package( Boost 1.53 REQUIRED COMPONENTS system program_options thread date_time chrono )

include_directories( ${CUDA_INCLUDE_DIRS} )
include_directories( ${VTK_INCLUDE_DIRS} )

# Configuration-dependent settings.
set( CMAKE_DEBUG_POSTFIX "_d" )

if ( UNIX )
set( CMAKE_CXX_FLAGS_RELEASE "-std=c++0x ${CMAKE_CXX_FLAGS_RELEASE}" )
set( CMAKE_CXX_FLAGS_DEBUG "-std=c++0x ${CMAKE_CXX_FLAGS_DEBUG}" )
endif ()

# Include files from the subdirectories and process the CMakeLists.txt in those 
# directories.
include_directories( src )
include_directories( include )
add_subdirectory( src )
