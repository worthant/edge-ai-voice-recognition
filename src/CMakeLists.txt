set(
  requires
  driver
  esp_partition
  esp_hw_support
  esp-dsp
  spiffs
)

FILE(
  GLOB_RECURSE 
    app_sources 
      ${CMAKE_SOURCE_DIR}/src/*.c
      ${CMAKE_SOURCE_DIR}/src/*.cpp
)

idf_component_register(
  SRCS ${app_sources} 
  REQUIRES ${requires}
  INCLUDE_DIRS "."
)
