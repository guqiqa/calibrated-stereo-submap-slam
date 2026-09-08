# The board image contains spdlog headers and libspdlog.so.1 but omits the
# upstream CMake package file.  Keep this compatibility target within the
# canary source tree instead of modifying the system SDK.
set(spdlog_FOUND TRUE)
set(spdlog_TARGETS spdlog::spdlog)
if(NOT TARGET spdlog::spdlog)
  add_library(spdlog::spdlog SHARED IMPORTED)
  set_target_properties(spdlog::spdlog PROPERTIES
    IMPORTED_LOCATION "/usr/lib/aarch64-linux-gnu/libspdlog.so.1"
    INTERFACE_INCLUDE_DIRECTORIES "/usr/include")
endif()
