# Pick the trained model based on MODEL_SLUG_BUILD (from platformio.ini).
# Slug must end with _qat or _ptq.
if(NOT DEFINED ENV{PLATFORMIO_MODEL_SLUG})
    set(MODEL_SLUG "f104_b5_ptq")
else()
    set(MODEL_SLUG $ENV{PLATFORMIO_MODEL_SLUG})
endif()

# Path to model artefacts produced by `python -m export_to_c --slug <MODEL_SLUG>`
set(MODEL_RUN_DIR "${CMAKE_SOURCE_DIR}/nn/results/merged_runs/${MODEL_SLUG}")

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
  esp-tflite-micro
)

# All sources in src/, plus model_data from selected run
file(GLOB_RECURSE app_sources
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

# Model data is now flashed separately into the 'model' partition.
# We don't link model_data.c into the firmware; we only need model_data.h
# for any forward declarations (not used at runtime — loaded via mmap).
list(FILTER app_sources EXCLUDE REGEX "src/model_data\\.c$")

idf_component_register(
    SRCS ${app_sources}
    REQUIRES ${requires}
    INCLUDE_DIRS "." "${MODEL_RUN_DIR}"
)
