cmake_minimum_required(VERSION 3.16.0)
project(prefix_grouper_npu_opp)

find_package(ASC REQUIRED)
set(package_name ${vendor_name})

npu_op_package(${package_name}
    TYPE RUN
    CONFIG INSTALL_PATH ${CMAKE_BINARY_DIR}/
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
