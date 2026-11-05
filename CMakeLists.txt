cmake_minimum_required(VERSION 3.20)
project(sdvk LANGUAGES C CXX VERSION 0.3.0)

set(SDVK_LICENSE "GPL-3.0-only")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(SDVK_BUILD_CLI "Build the sdvkgen command line tool" ON)
option(SDVK_BUILD_SERVER "Build the sdvk-server HTTP API" ON)

find_program(GLSLC NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin")
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found; set VULKAN_SDK or add glslc to PATH")
endif()
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
        gemm_nt_q8 gemm_nt_q8_v gemm_nn_q8 gemm_nn_q8_v fused_attn fused_attn_tiled)
    set(SPV ${VK_SHADER_DIR}/${SHADER}.spv)
    add_custom_command(OUTPUT ${SPV}
        COMMAND ${GLSLC} --target-env=vulkan1.1 ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/shaders/${SHADER}.comp -o ${SPV}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/shaders/${SHADER}.comp
        COMMENT "glslc ${SHADER}.comp")
    list(APPEND VK_SPVS ${SPV})
endforeach()

add_custom_command(OUTPUT ${VK_SHADER_DIR}/gemm_nt_q8_v_bm128.spv
    COMMAND ${GLSLC} --target-env=vulkan1.1 -DQBM=128 -DQBN=64 ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/shaders/gemm_nt_q8_v.comp -o ${VK_SHADER_DIR}/gemm_nt_q8_v_bm128.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/shaders/gemm_nt_q8_v.comp
    COMMENT "glslc gemm_nt_q8_v bm128")
list(APPEND VK_SPVS ${VK_SHADER_DIR}/gemm_nt_q8_v_bm128.spv)

# BK=32 probes
add_custom_command(OUTPUT ${VK_SHADER_DIR}/gemm_nt_q8_v_bk32.spv
    COMMAND ${GLSLC} --target-env=vulkan1.1 -DQBM=64 -DQBN=64 -DQBK=32 ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/shaders/gemm_nt_q8_v.comp -o ${VK_SHADER_DIR}/gemm_nt_q8_v_bk32.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/shaders/gemm_nt_q8_v.comp
    COMMENT "glslc gemm_nt_q8_v bk32")
list(APPEND VK_SPVS ${VK_SHADER_DIR}/gemm_nt_q8_v_bk32.spv)
add_custom_command(OUTPUT ${VK_SHADER_DIR}/gemm_nt_q8_v_bm128_bk32.spv
    COMMAND ${GLSLC} --target-env=vulkan1.1 -DQBM=128 -DQBN=64 -DQBK=32 ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/shaders/gemm_nt_q8_v.comp -o ${VK_SHADER_DIR}/gemm_nt_q8_v_bm128_bk32.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/shaders/gemm_nt_q8_v.comp
    COMMENT "glslc gemm_nt_q8_v bm128 bk32")
list(APPEND VK_SPVS ${VK_SHADER_DIR}/gemm_nt_q8_v_bm128_bk32.spv)

# bias-fused epilogue variants (5 bindings)
add_custom_command(OUTPUT ${VK_SHADER_DIR}/gemm_nt_q8_v_b.spv
    COMMAND ${GLSLC} --target-env=vulkan1.1 -DQBIAS=1 -DQBM=64 -DQBN=64 -DQBK=16 ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/shaders/gemm_nt_q8_v.comp -o ${VK_SHADER_DIR}/gemm_nt_q8_v_b.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/shaders/gemm_nt_q8_v.comp
    COMMENT "glslc gemm_nt_q8_v +bias")
list(APPEND VK_SPVS ${VK_SHADER_DIR}/gemm_nt_q8_v_b.spv)
add_custom_command(OUTPUT ${VK_SHADER_DIR}/gemm_nt_q8_v_bm128_b.spv
    COMMAND ${GLSLC} --target-env=vulkan1.1 -DQBIAS=1 -DQBM=128 -DQBN=64 -DQBK=16 ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/shaders/gemm_nt_q8_v.comp -o ${VK_SHADER_DIR}/gemm_nt_q8_v_bm128_b.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/shaders/gemm_nt_q8_v.comp
    COMMENT "glslc gemm_nt_q8_v bm128 +bias")
list(APPEND VK_SPVS ${VK_SHADER_DIR}/gemm_nt_q8_v_bm128_b.spv)

# ---- core: the SDXL pipeline as a library, no CLI, no sockets ----
add_library(sdvkcore STATIC
    src/Modules/Diffusion/sdvkgen.cpp
    src/Modules/Diffusion/Safetensors.cpp
    src/Modules/Diffusion/GenParams.cpp
    src/Modules/Diffusion/ModelSet.cpp
    src/Modules/Audio/Wav.cpp
    src/Modules/Audio/Dsp.cpp
    src/Modules/Model/Gguf.cpp
    src/Modules/GPU/gpu_ops.cpp
    src/Modules/GPU/fused_attn.cpp
    src/Modules/GPU/vk_engine.cpp
    src/Modules/GPU/volk/volk.c
    src/Modules/Core/Tensor.cpp
    ${VK_SPVS}
)
target_include_directories(sdvkcore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/Diffusion
    "$ENV{VULKAN_SDK}/Include"
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU/volk
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Modules/GPU
)
target_compile_definitions(sdvkcore PUBLIC VK_NO_PROTOTYPES)
if(MSVC)
    target_compile_options(sdvkcore PRIVATE /O2 /W3)
else()
    target_compile_options(sdvkcore PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
endif()
if(WIN32)
    target_link_libraries(sdvkcore PUBLIC shell32)
endif()

# ---- licensing ----
# GPL-3.0-only covers everything in src/ except the vendored volk, which stays MIT.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE DESTINATION share/doc/sdvk)
set(SDVK_GPL_NOTE "sdvk is free software under the GNU GPL v3.0 or later; see LICENSE. volk (src/Modules/GPU/volk) is MIT-licensed.")

# ---- cli ----
if(SDVK_BUILD_CLI)
    add_executable(sdvkgen src/Modules/Diffusion/main.cpp)
    target_link_libraries(sdvkgen PRIVATE sdvkcore)
    if(MSVC)
        target_compile_options(sdvkgen PRIVATE /O2 /W3)
    else()
        target_compile_options(sdvkgen PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
    endif()
endif()

# ---- http server: same core, thin socket layer ----
if(SDVK_BUILD_SERVER)
    add_executable(sdvk-server src/Modules/Server/server.cpp)
    target_link_libraries(sdvk-server PRIVATE sdvkcore)
    if(WIN32)
        target_link_libraries(sdvk-server PRIVATE ws2_32)
    endif()
    if(MSVC)
        target_compile_options(sdvk-server PRIVATE /O2 /W3)
    else()
        target_compile_options(sdvk-server PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
    endif()
endif()

# ---- tools: read-only diagnostics ----
add_executable(vkbudget tools/vkbudget.cpp)
target_link_libraries(vkbudget PRIVATE sdvkcore)

add_executable(ggufinfo tools/ggufinfo.cpp)
target_link_libraries(ggufinfo PRIVATE sdvkcore)

# ---- host-side allocator property test ----
add_executable(lolavamem_test tests/lolavamem_test.cpp src/Modules/GPU/lolavamem.cpp)
target_include_directories(lolavamem_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- audio module: pure CPU, links nothing GPU ----
add_executable(audio_test tests/audio_test.cpp src/Modules/Audio/Wav.cpp src/Modules/Audio/Dsp.cpp)
target_include_directories(audio_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- fused attention operator numeric + latency check ----
add_executable(fused_attn_test tests/fused_attn_test.cpp)
target_link_libraries(fused_attn_test PRIVATE sdvkcore)

# ---- q8 GEMM tile-variant microbenchmark ----
add_executable(gemm_q8_bench tests/gemm_q8_bench.cpp)
target_link_libraries(gemm_q8_bench PRIVATE sdvkcore)
