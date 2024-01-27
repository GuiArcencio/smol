#include "smol.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

char *load_program_source(const char *filename, cl_int *error) {
    struct stat st;
    int result = stat(filename, &st);
    if (result == -1) {
        *error = SMOL_INVALID_PROGRAM_FILE;
        return NULL;
    }

    long filesize = st.st_size;
    char *data = (char*) malloc(filesize * sizeof(char) + 1);

    FILE *f = fopen(filename, "r");
    fread(data, sizeof(char), filesize, f);
    data[filesize] = '\0';
    fclose(f);

    *error = CL_SUCCESS;
    return data;
}

cl_int opencl_init(OpenCL *ocl_dest) {
    cl_int err;
    OpenCL ocl = malloc(sizeof(_OpenCL));

    err = clGetPlatformIDs(1, &ocl->platform, NULL);
    if (err != CL_SUCCESS) return err;

    err = clGetDeviceIDs(ocl->platform, CL_DEVICE_TYPE_GPU, 1, &ocl->device, NULL);
    if (err != CL_SUCCESS) return err;

    ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return err;

    ocl->queue = clCreateCommandQueueWithProperties(ocl->context, ocl->device, NULL, &err);
    if (err != CL_SUCCESS) return err;

    *ocl_dest = ocl;
    return CL_SUCCESS;
}

cl_int opencl_load_program_file(OpenCL ocl, const char *filename) {
    cl_int err;

    char *source = load_program_source(filename, &err);
    if (err != CL_SUCCESS) return err;

    ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**) &source, NULL, &err);
    if (err != CL_SUCCESS) {
        free(source);
        return err;
    }

    err = clBuildProgram(ocl->program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        free(source);
        return err;
    }

    free(source);
    return CL_SUCCESS;
}

cl_int opencl_load_kernel(
    OpenCL ocl,
    cl_kernel *kernel_dest,
    const char *kernel_name,
    void *arg_values[],
    size_t arg_sizes[],
    size_t n_args
) {
    cl_int err;
    cl_kernel k;

    k = clCreateKernel(ocl->program, kernel_name, &err);
    if (err != CL_SUCCESS) return err;

    for (size_t i = 0; i < n_args; i++) {
        err = clSetKernelArg(k, i, arg_sizes[i], arg_values[i]);
        if (err != CL_SUCCESS) return err;
    }

    *kernel_dest = k;
    return CL_SUCCESS;
}

cl_int opencl_launch_loaded_kernel(
    OpenCL ocl,
    cl_kernel kernel,
    size_t dim_x,
    size_t dim_y,
    size_t dim_z
) {
    cl_int err;
    const size_t sizes[] = { dim_x, dim_y, dim_z };

    err = clEnqueueNDRangeKernel(
        ocl->queue,
        kernel,
        3,
        NULL,
        sizes,
        NULL,
        0, NULL, NULL
    );
    if (err != CL_SUCCESS) return err;

    clReleaseKernel(kernel);
    return CL_SUCCESS;
}

cl_int opencl_kernel_call(
    OpenCL ocl,
    const char *kernel_name,
    size_t dim_x,
    size_t dim_y,
    size_t dim_z,
    void *arg_values[],
    size_t arg_sizes[],
    size_t n_args
) {
    cl_int err;
    cl_kernel k;
    
    err = opencl_load_kernel(
        ocl,
        &k,
        kernel_name,
        arg_values,
        arg_sizes,
        n_args
    );
    if (err != CL_SUCCESS) return err;

    err = opencl_launch_loaded_kernel(
        ocl,
        k,
        dim_x,
        dim_y,
        dim_z
    );
    
    return err;
}

cl_int opencl_synchronize(OpenCL ocl) {
    return clFinish(ocl->queue);
}

cl_int _opencl_malloc_raw(
    OpenCL ocl,
    cl_mem *buf_ptr,
    size_t size,
    void *host_ptr,
    cl_mem_flags flags
) {
    cl_int err;

    cl_mem buf = clCreateBuffer(
        ocl->context,
        flags,
        size,
        host_ptr, 
        &err
    );
    if (err != CL_SUCCESS) return err;

    *buf_ptr = buf;
    return CL_SUCCESS;

}

cl_int opencl_malloc(
    OpenCL ocl,
    cl_mem *buf_ptr,
    size_t size
) {
    return _opencl_malloc_raw(
        ocl,
        buf_ptr,
        size,
        NULL,
        CL_MEM_READ_WRITE
    );
}

cl_int _opencl_memcpy(
    OpenCL ocl,
    void *destination,
    void *source,
    size_t n_bytes,
    opencl_memcpy_kind kind,
    cl_bool blocking
) {
    cl_int err;
    cl_mem device_buffer;

    switch (kind) {
        case opencl_memcpy_device_to_host:
            device_buffer = (cl_mem) source;
            err = clEnqueueReadBuffer(
                ocl->queue,
                device_buffer,
                blocking,
                0,
                n_bytes,
                destination,
                0, NULL, NULL
            );
            if (err != CL_SUCCESS) return err;
            break;
        case opencl_memcpy_host_to_device:
            device_buffer = (cl_mem) destination;
            err = clEnqueueWriteBuffer(
                ocl->queue,
                device_buffer,
                blocking,
                0,
                n_bytes,
                source,
                0, NULL, NULL
            );
            if (err != CL_SUCCESS) return err;
        default:
            break;
    }

    return CL_SUCCESS;
}

cl_int opencl_memcpy(
    OpenCL ocl,
    void *destination,
    void *source,
    size_t n_bytes,
    opencl_memcpy_kind kind
) {
    return _opencl_memcpy(
        ocl,
        destination,
        source,
        n_bytes,
        kind,
        CL_TRUE
    );
}

cl_int opencl_memcpy_async(
    OpenCL ocl,
    void *destination,
    void *source,
    size_t n_bytes,
    opencl_memcpy_kind kind
) {
    return _opencl_memcpy(
        ocl,
        destination,
        source,
        n_bytes,
        kind,
        CL_FALSE
    );
}

cl_int opencl_free(cl_mem buf) {
    return clReleaseMemObject(buf);
}

cl_int opencl_end(OpenCL ocl) {
    cl_int err;

    err = clReleaseProgram(ocl->program);
    if (err != CL_SUCCESS) return err;

    err = clReleaseCommandQueue(ocl->queue);
    if (err != CL_SUCCESS) return err;

    err = clReleaseContext(ocl->context);
    if (err != CL_SUCCESS) return err;

    free(ocl);
    return CL_SUCCESS;
}
