#!/usr/bin/env python

"""
Multi-Device Matrix Multiplication with Uncoalesced Access Pattern

For N=8192 matrices with 16×16 work group size
"""

import numpy as np

import pyopencl as cl

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from helper import *

from definitions import *

# Matrix size

N = 8192

# Work group size

WORKGROUP_SIZE = 16

  

# Optimization 1: Define CPU-specific parameters

CPU_PLATFORM_ID = 2  # Change based on your system

CPU_RESERVED_CORES = [6, 7]  # Reserve last 2 cores

CPU_MAX_THREADS = 8

  

def create_context_and_queue(platform_idx, device_idx=0):

    """Create an OpenCL context and command queue for the specified platform."""

    platforms = cl.get_platforms()

    if 0 <= platform_idx < len(platforms):

        platform = platforms[platform_idx]

        devices = platform.get_devices()

        if 0 <= device_idx < len(devices):

            device = devices[device_idx]

            context = cl.Context([device])

            queue = cl.CommandQueue(context)

            return context, queue, device

    return None, None, None

  

def load_kernel():

    """Load the uncoalesced kernel code."""

    with open("C_elem_ij.cl", "r") as f:

        return f.read()

  

def matrix_multiply_device(platform_idx, start_row, end_row, A_full, B_full, C_full):

    """Execute matrix multiplication on a specific device for a subset of rows."""

    # Create OpenCL context and queue for this device

    context, queue, device = create_context_and_queue(platform_idx)

    if context is None:

        print(f"Failed to create context for platform {platform_idx}")

        return False

    device_name = device.name

    print(f"Device: {device_name} processing rows {start_row} to {end_row}")

    # Prepare data for this device's portion

    num_rows = end_row - start_row

    A_device = A_full[start_row*N:end_row*N].copy()

    C_device = np.zeros(num_rows*N, dtype=np.float32)

    # Create OpenCL buffers

    mf = cl.mem_flags

    d_a = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_device)

    d_b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_full)

    d_c = cl.Buffer(context, mf.WRITE_ONLY, C_device.nbytes)

    # Build program

    kernel_source = load_kernel()

    program = cl.Program(context, kernel_source).build()

    mmul = program.mmul

    mmul.set_scalar_arg_dtypes([np.int32, None, None, None])

    # Execute kernel

    start_time = time.time()

    # Launch the kernel

    try:

        event = mmul(queue, (num_rows, N), (WORKGROUP_SIZE, WORKGROUP_SIZE),

                     np.int32(N), d_a, d_b, d_c)

        event.wait()

        cl.enqueue_copy(queue, C_device, d_c)

        queue.finish()

    except Exception as e:

        print(f"Error on device {device_name}: {str(e)}")

        return None

    execution_time = time.time() - start_time

    gflops = 2.0 * num_rows * N * N/(1000000000.0 * execution_time)

    print(f"Device: {device_name} completed in {execution_time:.4f}s at {gflops:.2f} GFLOPS")

    # Copy the result back to the appropriate part of C_full

    C_full[start_row*N:end_row*N] = C_device

    return platform_idx,execution_time, gflops

  

def multi_device_matrix_multiply():

    """Execute matrix multiplication across multiple devices."""

    # Initialize matrices

    A = np.full(N*N, AVAL, dtype=np.float32)

    B = np.full(N*N, BVAL, dtype=np.float32)

    C = np.zeros(N*N, dtype=np.float32)

    # Work distribution based on performance analysis

    # NVIDIA GPU: 89.782% (~7355 rows)

    # Intel CPU: 8.522% (~698 rows)

    # Intel iGPU: 1.695% (~139 rows)

    devices = [

        {"platform_idx": 0, "start_row": 0, "end_row": 8048}, #make the rows multiple of 16

        {"platform_idx": 2, "start_row": 8048, "end_row": 8064}, #Intel CPU

        {"platform_idx": 1, "start_row": 8048, "end_row": N}

    ]

    # Measure single device performance first (NVIDIA GPU with all rows)

    print("Starting single device benchmark (NVIDIA GPU only)...")

    _,single_device_time,_ = matrix_multiply_device(0, 0, N, A, B, np.zeros(N*N, dtype=np.float32))

    print(f"Single device time: {single_device_time:.4f}s")

    # Now run multi-device version

    print("\nStarting multi-device execution...")

    # Execute in parallel across all devices

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:

        futures = []

        for device_config in devices:

            futures.append(

                executor.submit(

                    matrix_multiply_device,

                    device_config["platform_idx"],

                    device_config["start_row"],

                    device_config["end_row"],

                    A, B, C

                )

            )

        execution_times = {}

        throughput_dict = {}

        for future in as_completed(futures):

            if future.result() is not None:

                plat_id, exec_time, gflops = future.result()

                execution_times[plat_id] = exec_time

                throughput_dict[plat_id] = gflops

    multi_device_time = max(execution_times.values())

    speedup = single_device_time / multi_device_time

    print("\n===== RESULTS =====")

    print(f"Single NVIDIA GPU: {single_device_time:.6f}s")

    print(f"Multi-device: {multi_device_time:.6f}s")

    print(f"Speedup: {speedup:.6f}x")

    return C, speedup,execution_times,throughput_dict