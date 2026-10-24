cmake_minimum_required(VERSION 3.20)
project(sdvk LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# SDXL text-to-image on the ofc Vulkan engine (RX 580 / 6 GiB target).
#   sdvkgen "<prompt>" --vpred --spacing trailing --steps 28 --cfg 3.0 --size 768
#     --no-vae --step-mb 1500 --scratch-mb 500 --neg "nsfw, worst quality, old, early, low quality, lowres"
# Default data paths live in src/Modules/Diffusion/sdvkgen.cpp (E:/ofc-data/noobai).

find_program(GLSLC NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin")
set(VK_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${VK_SHADER_DIR})
set(VK_SPVS "")
foreach(SHADER vadd gemm gemm_nn gemm_nt gemm_tn copy_nd ew2 mul_grad rowdot colsum_mul_add
        colsum_add bcast_scalar unary_fwd unary_bwd softmax_fwd softmax_bwd rmsnorm_fwd
        rmsnorm_bwd rowsum_fwd mean_seq reduce_all crossentropy_fwd crossentropy_bwd gather_rows
        scatter_add_rows repeat_batch_bwd rope_fwd rope_bwd mask_fill_fwd mask_fill_bwd
        topk_sparse_fwd topk_sparse_bwd compact compact_bwd onehot_balance_fwd
        onehot_balance_bwd adamw norm_sq clip_finalize adaf_stats adafactor transpose
        opt_norm opt_stats opt_meta opt_update
        bias_gelu_fwd gelu_bias_bwd moe_merge_fwd moe_merge_dy moe_merge_wg_grad reduce_g
        copy2 mul2 attn_softmax_fwd attn_softmax_bwd
        im2col groupnorm upsample2x bias_add add2 im2col_s2 layernorm_fwd causal_mask
        gemm_nt_q8 gemm_nn_q8)
    set(SPV ${VK_SHADER_DIR}/${SHADER}.spv)
    add_custom_command(OUTPUT ${SPV}
        COMMAND ${GLSLC} ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/shaders/${SHADER}.comp -o ${SPV}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/shaders/${SHADER}.comp
        COMMENT "glslc ${SHADER}.comp")
    list(APPEND VK_SPVS ${SPV})
endforeach()

add_executable(sdvkgen
    src/Modules/Diffusion/sdvkgen.cpp
    src/Modules/Diffusion/Safetensors.cpp
    src/Modules/Model/Gguf.cpp
    src/Modules/GPU/gpu_ops.cpp
    src/Modules/GPU/vk_engine.cpp
    src/Modules/GPU/volk/volk.c
    src/Modules/Core/Tensor.cpp
    ${VK_SPVS}
)
target_include_directories(sdvkgen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    "$ENV{VULKAN_SDK}/Include"
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/volk
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU
)
target_compile_definitions(sdvkgen PRIVATE VK_NO_PROTOTYPES)
if(MSVC)
    target_compile_options(sdvkgen PRIVATE /O2 /W3)
else()
    target_compile_options(sdvkgen PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
endif()
if(WIN32)
    target_link_libraries(sdvkgen PRIVATE shell32)
endif()
