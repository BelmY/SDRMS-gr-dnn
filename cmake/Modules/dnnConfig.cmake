INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_DNN dnn)

FIND_PATH(
    DNN_INCLUDE_DIRS
    NAMES dnn/api.h
    HINTS $ENV{DNN_DIR}/include
        ${PC_DNN_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    DNN_LIBRARIES
    NAMES gnuradio-dnn
    HINTS $ENV{DNN_DIR}/lib
        ${PC_DNN_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/dnnTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(DNN DEFAULT_MSG DNN_LIBRARIES DNN_INCLUDE_DIRS)
MARK_AS_ADVANCED(DNN_LIBRARIES DNN_INCLUDE_DIRS)
