find_package( Doxygen REQUIRED )

if( DOXYGEN_FOUND )
	# Process Doxyfile.in and expand certain cmake variables within the file 
	# and copy the file to the build directory as Doxyfile
	configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
					${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY )
	# Creates the doc target. Basically just runs doxygen.
	add_custom_target( doc ALL
					   ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
					   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
					   COMMENT "Generating API documentation with Doxygen" VERBATIM 
					   SOURCES Doxyfile.in sources.bib )
	# Install: Copies the documentation to <InstallDir>/doc/html
	install( DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html
			 DESTINATION doc
			 CONFIGURATIONS Debug Release )
	# Install: Copies the link to the documentation to <InstallDir>/doc
	install( FILES Voxelizer.html
			 DESTINATION doc
			 CONFIGURATIONS Debug Release )
endif()