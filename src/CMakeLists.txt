# Pick the trained model based on MODEL_SLUG_BUILD (from platformio.ini).
# Falls back to f172_b6_qat if not set.
if(NOT DEFINED ENV{PLATFORMIO_MODEL_SLUG})
    set(MODEL_SLUG "f172_b6_qat")
else()
    set(MODEL_SLUG $ENV{PLATFORMIO_MODEL_SLUG})
endif()

# Path to model artefacts produced by `python -m export_to_c --slug <MODEL_SLUG>`
set(MODEL_RUN_DIR "${CMAKE_SOURCE_DIR}/nn/results/kws_results_19_models/runs/${MODEL_SLUG}")

if(NOT EXISTS "${MODEL_RUN_DIR}/model_data.c")
    message(WARNING "Model not exported: ${MODEL_RUN_DIR}/model_data.c not found.")
    message(WARNING "Run: python -m export_to_c --slug ${MODEL_SLUG}")
    message(WARNING "Falling back to src/model_data.c if present.")
endif()

set(
  requires
  driver
  esp_partition
  esp_hw_support
  esp-dsp
  spiffs
)

# All sources in src/, plus model_data from selected run
file(GLOB_RECURSE app_sources
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

# Add model_data.c from run dir if present (preferred); otherwise keep src/model_data.c
if(EXISTS "${MODEL_RUN_DIR}/model_data.c")
    # Exclude src/model_data.c if it exists, to avoid duplicate symbols
    list(FILTER app_sources EXCLUDE REGEX "src/model_data\\.c$")
    list(APPEND app_sources "${MODEL_RUN_DIR}/model_data.c")
    message(STATUS "Using model from: ${MODEL_RUN_DIR}/model_data.c")
endif()

idf_component_register(
    SRCS ${app_sources}
    REQUIRES ${requires}
    INCLUDE_DIRS "." "${MODEL_RUN_DIR}"
)
