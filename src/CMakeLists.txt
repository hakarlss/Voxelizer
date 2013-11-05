# Sets up a build target for the Voxelizer that makes a library with CUDA.
cuda_add_library( Voxelizer STATIC ${SOURCE_FILES} )

# Sets library directories.
target_link_libraries( Voxelizer ${Boost_LIBRARIES} )

# Install: Copies the library to <InstallDir>/lib
install( TARGETS Voxelizer
		 ARCHIVE DESTINATION lib 
		 CONFIGURATIONS Debug Release )

# Install: Copies the .h files to <InstallDir>/include
install( DIRECTORY ../include
		 DESTINATION .
		 CONFIGURATIONS Debug Release )

# Separates the Boost library paths into separate containers depending on 
# if they are release or debug libraries.
foreach( ELEM ${Boost_LIBRARIES} )
	if ( ${ELEM} MATCHES "^optimized$" )
		set( BLN_LOOP_RELEASE ON )
	elseif( ${ELEM} MATCHES "^debug$" )
		set( BLN_LOOP_RELEASE OFF )
	elseif( ${BLN_LOOP_RELEASE} )
		list( APPEND BOOST_LIBNAMES_RELEASE ${ELEM} )
	else()
		list( APPEND BOOST_LIBNAMES_DEBUG ${ELEM} )
	endif()
endforeach()

unset( BLN_LOOP_RELEASE )

# Install: Copies the appropriate Boost libraries to <InstallDir>/lib
install( FILES ${BOOST_LIBNAMES_RELEASE} 
		 DESTINATION lib
		 CONFIGURATIONS Release )
		 
install( FILES ${BOOST_LIBNAMES_DEBUG} 
		 DESTINATION lib
		 CONFIGURATIONS Debug )

# Copies the library to <BuildDir>/lib for easier access without an install.
set_target_properties( Voxelizer PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/ )
