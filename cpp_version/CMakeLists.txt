cmake_minimum_required(VERSION 3.10)
project(DiffractionSolver VERSION 1.0 LANGUAGES CXX)

# Установка стандарта C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Опции компиляции для оптимизации
if(MSVC)
    # Visual Studio
    # Опции управляются CMake (Debug/Release)
    # add_compile_options(/W4 /O2)
else()
    # GCC/Clang
    add_compile_options(-Wall -Wextra -O3)
endif()

# Включаемые директории
include_directories(${CMAKE_SOURCE_DIR}/include)

# Исходные файлы
set(SOURCES
    src/Matrix.cpp
    src/Gauss.cpp
    src/Bessel.cpp
    src/Chebyshev.cpp
    src/DifrOnLenta.cpp
    src/main.cpp
)

# Заголовочные файлы
set(HEADERS
    include/Matrix.h
    include/Gauss.h
    include/Bessel.h
    include/Chebyshev.h
    include/DifrOnLenta.h
)

# Создание исполняемого файла
add_executable(diffraction_solver ${SOURCES} ${HEADERS})

# Линковка с математической библиотекой (для Linux/Unix)
if(UNIX)
    target_link_libraries(diffraction_solver m)
endif()

# Вывод информации о сборке
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")







