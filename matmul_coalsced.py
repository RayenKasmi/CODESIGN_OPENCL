#
# Matrix Multiplication Element By Element
# compute C=A*B
# C[i][j] for i: 0--> N-1, j! 0--> N-1
# 1) Kernel version to execute multiplication with j=0, i=1

# 2) The Program asks for 2 inputs:
# localsize (4,8,16 or 32) --> Block Size = localsize*localsize

from helper import *
from definitions import *

import numpy

import pyopencl as cl

from time import time
from time import sleep

# A[N][N], B[N][N], C[N][N]
N = 2048

# Number of elements in the matrix
size = N * N
#true value
cval = float(N) * AVAL * BVAL

# A matrix
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)

# B matrix
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)

# C matrix
h_C = numpy.empty(size).astype(numpy.float32)


#--------------------------------------------------------------------------------
# CHOOSE KERNEL TO EXECUTE (0: i=dim(0),j=dim(1) ; 1:i=dim(1), j=dim(0)
#--------------------------------------------------------------------------------
print ("Matrix multiplication",N,"*",N," repeated 20 times, j=0, i=1 :\n")
kernel_name="C_elem_ji.cl"
#--------------------------------------------------------------------------------
# CHOOSE localsize : 2, 4, 8 , 16 or 32
#--------------------------------------------------------------------------------
kernel_size = input("Please enter a value for localsize. Possible values: 4, 8, 16 and 32 :\n")

if (kernel_size in ['4','8','16','32'] ):
    localsize=int(kernel_size)
    print ("Blocks Size is",localsize,"*",localsize,"\n")
else:
    print ("=== No valid input. Default Size 16 will be used. Block Size = 16*16")
    localsize=16


# Set up OpenCL
# Get platform selection from user
platform_idx_str = input("Please enter platform index:\n")
try:
    platform_idx = int(platform_idx_str)
    platforms = cl.get_platforms()
    if 0 <= platform_idx < len(platforms):
        platform = platforms[platform_idx]
        devices = platform.get_devices()
        if devices:
            context = cl.Context([devices[0]])
            print(f"Using platform: {platform.name}")
        else:
            print("No devices found on selected platform. Using default.")
            context = cl.create_some_context(interactive=False)
    else:
        print(f"Invalid platform index {platform_idx}. Using default.")
        context = cl.create_some_context(interactive=False)
except ValueError:
    print("Invalid platform index. Using default.")
    context = cl.create_some_context(interactive=False)

queue = cl.CommandQueue(context)

# Reset host buffers - just to play it safe
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)
h_C = numpy.empty(size).astype(numpy.float32)

# Create OpenCL buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=h_C.nbytes)

#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... Naive: Each WI computes one element
# C_elemnt.cl : i= get_global_id(0) - j=get_global_id(1)
#--------------------------------------------------------------------------------
kernelsource = open(kernel_name).read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None])

# Do the multiplication COUNT times

print ("\n Starting ", COUNT, " OpenCL Matrix Multiplications")
start_time = time()


for i in range(COUNT):    
    #h_C.fill(0.0)
    try:
        mmul(queue, (N,N), (1,localsize), numpy.int32 (N), d_a, d_b, d_c)
        queue.finish()
    except:
        print (" ===  Error for localsize =", localsize, "===\n")    

run_time = time() - start_time


print ("\n End of", COUNT, "Matrix Multiplications\n")

results (N, COUNT , run_time)

#reading the result h_C
cl.enqueue_copy(queue, h_C, d_c)