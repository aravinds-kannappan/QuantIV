cmake_minimum_required(VERSION 3.12)
project(VolatilityAlchemist)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Add curl for HTTP downloads
find_package(CURL REQUIRED)

# Include directories
include_directories(include)
include_directories(src)

# Create library for core modules
set(CORE_SOURCES
    src/core/data_loader.cpp
    src/core/volatility_model.cpp
    src/core/strategy_engine.cpp
    src/core/metrics.cpp
    src/core/exporter.cpp
    src/models/black_scholes.cpp
    src/models/heston.cpp
    src/models/sabr.cpp
    src/strategies/covered_call.cpp
    src/strategies/vertical_spread.cpp
    src/strategies/straddle.cpp
    src/strategies/iron_condor.cpp
    src/utils/math_utils.cpp
    src/utils/date_utils.cpp
    src/utils/http_client.cpp
)

add_library(volatility_core ${CORE_SOURCES})

# Link libraries
target_link_libraries(volatility_core ${CURL_LIBRARIES})

# Main executable
add_executable(volatility_alchemist src/main.cpp)
target_link_libraries(volatility_alchemist volatility_core)

# Compiler flags
target_compile_options(volatility_core PRIVATE -Wall -Wextra -O3)
target_compile_options(volatility_alchemist PRIVATE -Wall -Wextra -O3)

# Tests
enable_testing()
add_subdirectory(tests)