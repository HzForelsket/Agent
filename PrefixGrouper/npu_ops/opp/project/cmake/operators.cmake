# Collect the implementation owned by each operator before CANN code generation.
function(prefix_grouper_add_op op_type op_name)
    set_property(GLOBAL APPEND PROPERTY PREFIX_GROUPER_OP_DEFS
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/${op_name}_def.cpp)
    set_property(GLOBAL APPEND PROPERTY PREFIX_GROUPER_OP_IMPLS
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/${op_name}_infershape.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/${op_name}_tiling.cpp)
    file(RELATIVE_PATH kernel_dir ${PROJECT_SOURCE_DIR}/attention
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
    set_property(GLOBAL APPEND PROPERTY PREFIX_GROUPER_OP_TYPES ${op_type})
    set_property(GLOBAL PROPERTY PREFIX_GROUPER_KERNEL_${op_type}
        ${kernel_dir}/${op_name}.cpp)
endfunction()

function(prefix_grouper_build_ops)
    get_property(op_defs GLOBAL PROPERTY PREFIX_GROUPER_OP_DEFS)
    get_property(op_impls GLOBAL PROPERTY PREFIX_GROUPER_OP_IMPLS)
    get_property(op_types GLOBAL PROPERTY PREFIX_GROUPER_OP_TYPES)
    npu_op_code_gen(SRC ${op_defs} PACKAGE ${package_name}
        OUT_DIR ${ASCEND_AUTOGEN_PATH} JOIN_OP_DEF True)

    file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
    file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc
        ${ASCEND_AUTOGEN_PATH}/group_op_proto/*.cc)
    file(GLOB fallback_src ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)
    set_source_files_properties(${aclnn_src} ${proto_src} ${fallback_src}
        PROPERTIES GENERATED TRUE)
    npu_op_library(cust_opapi ACLNN ${aclnn_src})
    npu_op_library(cust_op_proto GRAPH ${op_defs} ${proto_src})
    npu_op_library(cust_optiling TILING ${op_impls} ${fallback_src}
        ${PROJECT_SOURCE_DIR}/attention/common/op_host/attention_tiling.cpp
        ${PROJECT_SOURCE_DIR}/attention/common/op_host/vector_tiling.cpp)
    foreach(target cust_opapi cust_op_proto cust_optiling)
        target_compile_options(${target} PRIVATE -fvisibility=hidden)
    endforeach()
    npu_op_package_add(${package_name} LIBRARY cust_optiling cust_opapi cust_op_proto)

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g -O0)
    endif()
    # Preserve per-operator paths and the shared headers in CANN's staged sources.
    npu_op_kernel_sources(ascendc_kernels KERNEL_DIR ./)
    foreach(op_type ${op_types})
        get_property(kernel_file GLOBAL PROPERTY PREFIX_GROUPER_KERNEL_${op_type})
        npu_op_kernel_sources(ascendc_kernels OP_TYPE ${op_type}
            KERNEL_FILE ${kernel_file})
    endforeach()
    npu_op_kernel_library(ascendc_kernels
        SRC_BASE ${PROJECT_SOURCE_DIR}/attention TILING_LIBRARY cust_optiling)
    npu_op_package_add(${package_name} LIBRARY ascendc_kernels)
endfunction()
