# SMOL - Simple and Minimal OpenCL

This is a very (*very!*) simple wrapper for the OpenCL C library. It is supposed to be similar to CUDA (a framework I'm more familiar with).

## Example

Create a file to store the kernel code:
```C
// kernel.cl

__kernel void multiply(
    __global const double *A,
    __global double *B
)
{
    int i = get_global_id(0);
    B[i] = 2.0 * A[i];
}
```

The program could then be:
```C
// main.c

#include <stdio.h>
#include <stdlib.h>

#include "smol.h"

#define SIZE 10

int main() {
    OpenCL ocl;
    opencl_init(&ocl);
    opencl_load_program_file(ocl, "kernel.cl");

    cl_mem d_A, d_B;
    double *A = (double*) malloc(SIZE * sizeof(double));
    double *B = (double*) malloc(SIZE * sizeof(double));
    for (int i = 0; i < SIZE; i++)
        A[i] = 1.0;

    opencl_malloc(
        ocl, &d_A, SIZE * sizeof(double)
    );
    opencl_malloc(
        ocl, &d_B, SIZE * sizeof(double)
    );
    opencl_memcpy(
        ocl, d_A, A, SIZE * sizeof(double), opencl_memcpy_host_to_device
    );

    void *kernel_args[] = { &d_A, &d_B };
    size_t kernel_arg_sizes[] = { sizeof(cl_mem), sizeof(cl_mem) };
    opencl_kernel_call(
        ocl,
        "multiply",
        SIZE, 1, 1,
        kernel_args,
        kernel_arg_sizes,
        2
    );

    opencl_memcpy(
        ocl, B, d_B, SIZE * sizeof(double), opencl_memcpy_device_to_host
    );

    for (int i = 0; i < SIZE; i++)
        printf("%lf ", B[i]);
    printf("\n");

    opencl_free(d_A);
    opencl_free(d_B);

    free(A);
    free(B);
    
    opencl_end(ocl);

    return 0;
}
```

Remember to compile with the proper flags!

```bash
gcc -lOpenCL main.c smol.c -o prog && ./prog
```

## Warning

This library is intended to be as minimal as possible. The initialization code will simply choose the first device it can find. Do not trust this code for serious work.