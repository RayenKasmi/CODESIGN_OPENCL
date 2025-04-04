#!/usr/bin/env python
"""
Multi-Device Matrix Multiplication with Uncoalesced Access Pattern
For N=8192 matrices with 16×16 work group size
"""
import numpy as np
import pyopencl as cl
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from helper import *
from definitions import *

# Matrix size
N = 8192
# Work group size
WORKGROUP_SIZE = 16

# CPU core configuration
CPU_PLATFORM_ID = 2  # Intel CPU platform
CPU_TOTAL_CORES = 16  # Total logical processors
CPU_COMPUTE_CORES = 2  # Cores dedicated to computation
CPU_TRANSFER_CORES = 14  # Cores dedicated to data transfer

# Create a lock for transfer operations
transfer_lock = threading.Lock()

def set_thread_affinity(core_ids):
    """Set thread affinity to specific cores."""
    try:
        # For Windows
        if hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, core_ids)
        else:
            import ctypes
            if hasattr(ctypes.windll, 'kernel32'):
                # Windows-specific thread affinity setting
                mask = sum(1 << i for i in core_ids)
                ctypes.windll.kernel32.SetThreadAffinityMask(
                    ctypes.windll.kernel32.GetCurrentThread(), mask)
    except Exception as e:
        print(f"Warning: Could not set thread affinity: {e}")

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

def transfer_data(context, queue, data, read_only=True, host_to_device=True):
    """Handle data transfer with dedicated cores."""
    with transfer_lock:
        # Set thread affinity to transfer cores
        set_thread_affinity(list(range(CPU_COMPUTE_CORES, CPU_TOTAL_CORES)))
        
        mf = cl.mem_flags
        if host_to_device:
            if read_only:
                buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
            else:
                buffer = cl.Buffer(context, mf.WRITE_ONLY, size=data.nbytes)
            return buffer
        else:
            # Device to host transfer
            cl.enqueue_copy(queue, data, read_only)
            queue.finish()

def matrix_multiply_device(platform_idx, start_row, end_row, A_full, B_full, C_full, 
                          use_dedicated_cores=False, compute_core_ids=None):
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
    
    # Create OpenCL buffers - use dedicated transfer cores if specified
    if use_dedicated_cores and platform_idx == CPU_PLATFORM_ID:
        d_a = transfer_data(context, queue, A_device, read_only=True, host_to_device=True)
        d_b = transfer_data(context, queue, B_full, read_only=True, host_to_device=True)
        d_c = transfer_data(context, queue, C_device, read_only=False, host_to_device=True)
    else:
        mf = cl.mem_flags
        d_a = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_device)
        d_b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_full)
        d_c = cl.Buffer(context, mf.WRITE_ONLY, C_device.nbytes)
    
    # Build program
    kernel_source = load_kernel()
    program = cl.Program(context, kernel_source).build()
    mmul = program.mmul
    mmul.set_scalar_arg_dtypes([np.int32, None, None, None])
    
    # Set thread affinity for computation if using CPU
    if use_dedicated_cores and platform_idx == CPU_PLATFORM_ID and compute_core_ids:
        set_thread_affinity(compute_core_ids)
    
    # Execute kernel
    start_time = time.time()
    
    try:
        event = mmul(queue, (num_rows, N), (WORKGROUP_SIZE, WORKGROUP_SIZE), 
                     np.int32(N), d_a, d_b, d_c)
        event.wait()
        
        # Copy result back - use dedicated transfer cores if specified
        if use_dedicated_cores and platform_idx == CPU_PLATFORM_ID:
            transfer_data(context, queue, C_device, read_only=d_c, host_to_device=False)
        else:
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
    
    return platform_idx, execution_time, gflops

def multi_device_matrix_multiply():
    """Execute matrix multiplication across multiple devices."""
    # Initialize matrices
    A = np.full(N*N, AVAL, dtype=np.float32)
    B = np.full(N*N, BVAL, dtype=np.float32)
    C = np.zeros(N*N, dtype=np.float32)
    
    # Define compute cores for CPU work
    cpu_compute_cores = list(range(CPU_COMPUTE_CORES))
    
    # Work distribution based on performance analysis
    devices = [
        {"platform_idx": 0, "start_row": 0, "end_row": 7856, 
         "use_dedicated_cores": False},
        {"platform_idx": 2, "start_row": 7856, "end_row": 8048, 
         "use_dedicated_cores": True, "compute_core_ids": cpu_compute_cores},
        {"platform_idx": 1, "start_row": 8048, "end_row": N, 
         "use_dedicated_cores": False}
    ]
    
    # Measure single device performance first (NVIDIA GPU with all rows)
    print("Starting single device benchmark (NVIDIA GPU only)...")
    _, single_device_time, _ = matrix_multiply_device(0, 0, N, A, B, np.zeros(N*N, dtype=np.float32))
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
                    A, B, C,
                    device_config.get("use_dedicated_cores", False),
                    device_config.get("compute_core_ids", None)
                )
            )
        
        # Wait for all tasks to complete and get execution times
        execution_times = {}
        throughput_dict = {}
        for future in as_completed(futures):
            if future.result() is not None:
                plat_id, exec_time, gflops = future.result()
                execution_times[plat_id] = exec_time
                throughput_dict[plat_id] = gflops
    
    multi_device_time = max(execution_times.values())
    
    # Calculate speedup
    speedup = single_device_time / multi_device_time
    
    print("\n===== RESULTS =====")
    print(f"Single NVIDIA GPU: {single_device_time:.6f}s")
    print(f"Multi-device: {multi_device_time:.6f}s")
    print(f"Speedup: {speedup:.6f}x")
    
    return C, speedup, execution_times, throughput_dict

def compute_row_distribution(N, throughput_dict):
    """Compute row distribution based on throughput."""
    total_throughput = sum(throughput_dict.values())
    distribution = {}
    #starting with the lowest throught put device
    for device, throughput in sorted(throughput_dict.items(), key=lambda x: x[1]):
        # Calculate the number of rows for this device based on its throughput
        num_rows = int(N * (throughput / total_throughput))
        num_rows = round(num_rows)
        distribution[device] = num_rows
    id_device_lowest_throughput = min(distribution, key=distribution.get)
    id_device_highest_throughput = max(distribution, key=distribution.get)
    # Adjust the distribution to ensure it sums to N
    if(sum(distribution.values()) < N):
        distribution[id_device_highest_throughput] += N - sum(distribution.values())
    elif(sum(distribution.values()) > N):
        distribution[id_device_lowest_throughput] -= sum(distribution.values()) - N 
    leftover = 0

    #start with the lowest distribution device to the highest distribution device
    for device in sorted(distribution, key=distribution.get):
        if(device == id_device_highest_throughput):
            distribution[device] += leftover
        else:
            leftover += distribution[device]%16
            distribution[device] -= distribution[device]%16

    return distribution

result_matrix, speedup, execution_times, throughput_dict = multi_device_matrix_multiply()
print(f"\nFinal speedup achieved: {speedup:.2f}x")
print(f"Execution times for each device: {execution_times}")
print(f"Throughput for each device: {throughput_dict}")
#print(f"there are {error(N,result_matrix)} errors in the multiplication")
print(f"matrix first 10 elements:\n{result_matrix[:10]}")

distribution = compute_row_distribution(N, throughput_dict)
print(f"Better row distribution based on throughput: {distribution}")




