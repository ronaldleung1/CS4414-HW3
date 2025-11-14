cmake_minimum_required(VERSION 3.16)
project(rag_cpp CXX)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Where llama.cpp was installed ----
set(LLAMA_PREFIX "$ENV{HOME}/opt_dev" CACHE PATH "Install prefix of llama.cpp")

set(LLAMA_INCLUDE_DIR "${LLAMA_PREFIX}/include")
set(GGML_INCLUDE_DIR  "${LLAMA_PREFIX}/include/ggml")
set(LLAMA_LIB_DIR     "${LLAMA_PREFIX}/lib")

# Threads (pthread)
find_package(Threads REQUIRED)

# Optional: add OpenMP if available
find_package(OpenMP)

# ---- Find FAISS (installed via Homebrew) ----
find_path(FAISS_INCLUDE_DIR
  NAMES faiss/IndexFlat.h
  PATHS /opt/homebrew/include /usr/local/include
)

find_library(FAISS_LIBRARY
  NAMES faiss
  PATHS /opt/homebrew/lib /usr/local/lib
)

if(NOT FAISS_INCLUDE_DIR OR NOT FAISS_LIBRARY)
  message(FATAL_ERROR "Could not find FAISS. Install with: brew install faiss")
endif()

message(STATUS "Found FAISS include: ${FAISS_INCLUDE_DIR}")
message(STATUS "Found FAISS library: ${FAISS_LIBRARY}")

# ---- Find nlohmann/json (installed via Homebrew) ----
find_path(JSON_INCLUDE_DIR
  NAMES nlohmann/json.hpp
  PATHS /opt/homebrew/include /usr/local/include
)

if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find nlohmann/json. Install with: brew install nlohmann-json")
endif()

message(STATUS "Found nlohmann/json include: ${JSON_INCLUDE_DIR}")

# ---- Target 1 - data_preprocess ----
add_executable(data_preprocess data_preprocess.cpp)

target_include_directories(data_preprocess PRIVATE
  "${LLAMA_INCLUDE_DIR}"
  "${GGML_INCLUDE_DIR}"
  "${JSON_INCLUDE_DIR}"
)

target_link_directories(data_preprocess PRIVATE "${LLAMA_LIB_DIR}")

target_link_libraries(data_preprocess PRIVATE
  llama
  ggml
  ggml-base
  ggml-cpu
  Threads::Threads
  dl
  m
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(data_preprocess PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(data_preprocess PROPERTIES
  BUILD_RPATH   "${LLAMA_LIB_DIR}"
  INSTALL_RPATH "${LLAMA_LIB_DIR}"
)

# # ---- Target 2: vector_db_test ----
# add_executable(vector_db_test vector_db.cpp)

# target_include_directories(vector_db_test PRIVATE
#   "${LLAMA_INCLUDE_DIR}"
#   "${GGML_INCLUDE_DIR}"
#   "${JSON_INCLUDE_DIR}"
#   "${FAISS_INCLUDE_DIR}"
# )

# target_link_directories(vector_db_test PRIVATE "${LLAMA_LIB_DIR}")

# target_link_libraries(vector_db_test PRIVATE
#   llama
#   ggml
#   ggml-base
#   ggml-cpu
#   ${FAISS_LIBRARY}
#   Threads::Threads
#   dl
#   m
# )

# if(OpenMP_CXX_FOUND)
#   target_link_libraries(vector_db_test PRIVATE OpenMP::OpenMP_CXX)
# endif()

# set_target_properties(vector_db_test PROPERTIES
#   BUILD_RPATH   "${LLAMA_LIB_DIR}"
#   INSTALL_RPATH "${LLAMA_LIB_DIR}"
# )