#ifndef SMOL_H
#define SMOL_H

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#define SMOL_INVALID_PROGRAM_FILE 1

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
} _OpenCL;
typedef _OpenCL* OpenCL;

typedef enum {
    opencl_memcpy_host_to_device,
    opencl_memcpy_device_to_host
} opencl_memcpy_kind;

cl_int opencl_init(OpenCL *ocl_dest);

cl_int opencl_load_program_file(OpenCL ocl, const char *filename);

cl_int opencl_load_kernel(
    OpenCL ocl,
    cl_kernel *kernel_dest,
    const char *kernel_name,
    void *arg_values[],
    size_t arg_sizes[],
    size_t n_args
);

cl_int opencl_launch_loaded_kernel(
    OpenCL ocl,
    cl_kernel kernel,
    size_t dim_x,
    size_t dim_y,
    size_t dim_z
);

cl_int opencl_kernel_call(
    OpenCL ocl,
    const char *kernel_name,
    size_t dim_x,
    size_t dim_y,
    size_t dim_z,
    void *arg_values[],
    size_t arg_sizes[],
    size_t n_args
);

cl_int opencl_synchronize(OpenCL ocl);

// Memory

cl_int opencl_malloc(
    OpenCL ocl,
    cl_mem *buf_ptr,
    size_t size,
    void *host_ptr,
    cl_mem_flags flags
);

cl_int opencl_memcpy(
    OpenCL ocl,
    void *destination,
    void *source,
    size_t n_bytes,
    opencl_memcpy_kind kind
);

cl_int opencl_memcpy_async(
    OpenCL ocl,
    void *destination,
    void *source,
    size_t n_bytes,
    opencl_memcpy_kind kind
);

cl_int opencl_free(cl_mem buf);

cl_int opencl_end(OpenCL ocl);

#endif