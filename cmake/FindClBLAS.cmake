# - Find clBLAS

include(FindPackageHandleStandardArgs)

find_library(CLBLAS_LIBRARY "clBLAS")
find_path(CLBLAS_INCLUDE_DIR "clBLAS.h")

find_package_handle_standard_args(
  ClBLAS
  REQUIRED_VARS CLBLAS_LIBRARY CLBLAS_INCLUDE_DIR
  )
