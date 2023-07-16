cmake_minimum_required(VERSION 3.16.1)

project(pyged)

find_package(pybind11)
pybind11_add_module(pyged src/pyged.cpp)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

add_custom_command(TARGET pyged POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy
		${CMAKE_SOURCE_DIR}/build/pyged.*.so
		${CMAKE_SOURCE_DIR}/lib/
)

target_compile_options(pyged PUBLIC
	-fpermissive
)

target_include_directories(pyged PUBLIC
	${CMAKE_SOURCE_DIR}/ext/gedlib
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/boost_1_82_0
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/eigen.3.3.4/Eigen
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/nomad.3.8.1/src
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/nomad.3.8.1/ext/sgtelib/src
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/lsape.5/include
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/libsvm.3.22
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/fann.2.2.0/include
	${CMAKE_SOURCE_DIR}/ext/gurobi911/linux64/include
)

target_link_directories(pyged PUBLIC
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/nomad.3.8.1/lib
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/libsvm.3.22
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/fann.2.2.0/lib
	${CMAKE_SOURCE_DIR}/ext/gurobi911/linux64/src/build
	${CMAKE_SOURCE_DIR}/ext/gurobi911/linux64/lib
)

target_link_libraries(pyged PUBLIC
	doublefann
	svm
	nomad
	gurobi_c++
	gurobi91
)

