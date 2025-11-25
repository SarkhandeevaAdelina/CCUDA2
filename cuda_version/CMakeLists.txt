cmake_minimum_required(VERSION 3.23)
project(DiffractionCuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif ()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  /usr/local/cuda/include
)

set(DIFFRACTION_SOURCES
    ${SRC_DIR}/main.cu
    ${SRC_DIR}/DifrOnLentaCuda.cu
    ${SRC_DIR}/Bessel.cu
    ${SRC_DIR}/Chebyshev.cu
    ${SRC_DIR}/Matrix.cpp
    ${SRC_DIR}/Gauss.cpp
)

add_executable(CudaDiffr ${DIFFRACTION_SOURCES})
target_include_directories(CudaDiffr PRIVATE
    ${INCLUDE_DIR}
    ${CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(CudaDiffr PRIVATE 
    CUDA::cudart 
    CUDA::cusolver 
    CUDA::cublas
)
set_target_properties(CudaDiffr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
