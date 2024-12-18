# Main fortran wrapper
add_library(Spglib_fortran)
add_library(Spglib::fortran ALIAS Spglib_fortran)
configure_file(spglib_version.f90.in spglib_version.f90)

# This target is only used in the SpglibConfig.cmake
add_library(Spglib_fortran_include INTERFACE)
target_sources(Spglib_fortran_include INTERFACE
		"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/spglib_f08.F90>"
)
set_target_properties(Spglib_fortran_include PROPERTIES
		EXPORT_NAME fortran_include
)

# Concatenate the contents of the fortran file so it can be compiled from source
execute_process(COMMAND ${CMAKE_COMMAND} -E cat
		${CMAKE_CURRENT_BINARY_DIR}/spglib_version.f90
		${CMAKE_CURRENT_SOURCE_DIR}/spglib_f08.F90
		OUTPUT_VARIABLE spglib_f08_concat
		WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/spglib_f08.F90 ${spglib_f08_concat})

# Use the separate files for the project itself so that they are properly re-built
target_sources(Spglib_fortran PRIVATE
		spglib_f08.F90
		${CMAKE_CURRENT_BINARY_DIR}/spglib_version.f90
)

# Configure Fortran wrapper
set_target_properties(Spglib_fortran PROPERTIES
		VERSION ${PROJECT_VERSION}
		SOVERSION ${PROJECT_VERSION_MAJOR}
		EXPORT_NAME fortran_mod
		OUTPUT_NAME spglib_f08
)
set_target_properties(Spglib_fortran_include PROPERTIES
		PUBLIC_HEADER ${CMAKE_CURRENT_BINARY_DIR}/spglib_f08.F90
)
target_include_directories(Spglib_fortran PUBLIC
		"$<BUILD_INTERFACE:${CMAKE_Fortran_MODULE_DIRECTORY}>")
target_link_libraries(Spglib_fortran PUBLIC Spglib_symspg)
target_link_libraries(Spglib_fortran_include INTERFACE Spglib_symspg)
# Note: Fortran wrapper is not linked to OpenMP library because it should not be defining any such setup

# Install
if (SPGLIB_INSTALL)
	# CMake does not properly support fortran module installation paths.
	# Adapting the standard from fortran-stdlib
	# https://gitlab.kitware.com/cmake/cmake/-/issues/19608
	# https://discourse.cmake.org/t/api-design-c-modules-source-listings-and-interface-properties/5389/14
	cmake_path(APPEND CMAKE_INSTALL_INCLUDEDIR ${PROJECT_NAME} "${CMAKE_Fortran_COMPILER_ID}-${CMAKE_Fortran_COMPILER_VERSION}"
			OUTPUT_VARIABLE _DEFAULT_CMAKE_INSTALL_MODULEDIR)
	set(CMAKE_INSTALL_MODULEDIR ${_DEFAULT_CMAKE_INSTALL_MODULEDIR}
			CACHE STRING
			"Fortran module installation path (Not a cmake native variable)"
	)
	cmake_path(IS_ABSOLUTE CMAKE_INSTALL_MODULEDIR _is_absolute)
	if (_is_absolute)
		set(CMAKE_INSTALL_FULL_MODULEDIR ${CMAKE_INSTALL_MODULEDIR})
	else ()
		cmake_path(APPEND CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_MODULEDIR}
				OUTPUT_VARIABLE CMAKE_INSTALL_FULL_MODULEDIR)
	endif ()
	target_include_directories(Spglib_fortran PUBLIC
			"$<INSTALL_INTERFACE:${CMAKE_INSTALL_MODULEDIR}>")
	install(TARGETS Spglib_fortran Spglib_fortran_include
			EXPORT SpglibTargets-fortran
			LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Spglib_Runtime
			NAMELINK_COMPONENT Spglib_Development
			ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Spglib_Development
			PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT Spglib_Development
			RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Spglib_Runtime
	)
	export_components(COMPONENT fortran LIB_TYPE ${SPGLIB_LIB_TYPE})
	install(FILES try_compile.f90
			DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Spglib
			COMPONENT Spglib_Development
	)

	# Maybe it is handled automatically
	install(FILES ${CMAKE_Fortran_MODULE_DIRECTORY}/spglib_f08.mod
			DESTINATION ${CMAKE_INSTALL_MODULEDIR}
			COMPONENT Spglib_Development)

	configure_file(spglib_f08.pc.in spglib_f08.pc @ONLY)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/spglib_f08.pc
			DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
			COMPONENT Spglib_Development)
endif ()
