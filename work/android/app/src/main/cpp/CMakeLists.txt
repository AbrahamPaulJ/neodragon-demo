cmake_minimum_required(VERSION 3.22.1)
project(ndqnn)

add_library(ndqnn SHARED ndqnn.cpp)

# Both roots are needed: this file includes "QNN/QnnBackend.h", while the SDK's own
# sub-headers include their siblings unqualified ("QnnCommon.h" from HTP/QnnHtpDevice.h).
target_include_directories(ndqnn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/QNN)

# The QNN backend is dlopen'd at runtime from the app's nativeLibraryDir rather than
# linked, exactly as qnn-net-run does it. That keeps this library independent of which
# HTP variant is present and means no QNN .so is needed at build time.
target_link_libraries(ndqnn android log dl)

target_compile_options(ndqnn PRIVATE -O2 -fvisibility=hidden -Wall)
