#!/usr/bin/env python

import subprocess
import os
import re
import pandas as pd
from tabulate import tabulate
import sys
import time

def run_benchmark(script_name, platform_idx, localsize):
    """Run a specific benchmark with given platform and localsize"""
    try:
        # Create a process to run the benchmark
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # First input: local size
        print(f"localsize={localsize}")

        process.stdin.write(f"{localsize}\n")
        process.stdin.flush()

        # Wait a bit to ensure the program processes the input
        time.sleep(0.5)
        
        # Second input: platform selection
        process.stdin.write(f"{platform_idx}\n")
        process.stdin.flush()

        # Get the output
        stdout, stderr = process.communicate(timeout=100000)
        
        # Check for errors in stderr or stdout
        if "Error" in stderr or "Error" in stdout:
            return "error"
        
        # Extract GFLOPS using regex
        match = re.search(r'(\d+\.\d+) seconds at (\d+\.\d+) GFLOPS', stdout)
        if match:
            return float(match.group(2))  # Return GFLOPS value
        else:
            return "error"
    
    except subprocess.TimeoutExpired:
        process.kill()
        return "timeout"
    except Exception as e:
        return f"error: {str(e)}"

def get_platform_names():
    """Get the names of available OpenCL platforms"""
    import pyopencl as cl
    platforms = cl.get_platforms()
    return [p.name for p in platforms]

def main():
    # Scripts to benchmark
    scripts = [
        "matmul_coalsced.py",
        "matmul_uncoalsced.py", 
        "matmul_block.py"
    ]
    
    # Local sizes to test
    localsizes = [2, 4, 8, 16, 32]
    
    # Get platform names
    try:
        platform_names = get_platform_names()
        print(f"Detected platforms: {platform_names}")
    except:
        platform_names = ["NVIDIA CUDA", "Intel HD Graphics", "Intel OpenCL"]
        print(f"Using default platform names: {platform_names}")
    
    # Number of platforms
    num_platforms = len(platform_names)
    
    # Storage for results - one DataFrame per platform
    results = {i: pd.DataFrame(index=localsizes, columns=[s.replace(".py", "") for s in scripts]) 
               for i in range(num_platforms)}
    
    # Run all benchmarks
    for platform_idx in range(2):
        platform_name = platform_names[platform_idx]
        print(f"\nRunning benchmarks for platform {platform_idx}: {platform_name}")
        
        for script in scripts:
            script_name = script.replace(".py", "")
            print(f"\n  Testing {script_name}:")
            
            for localsize in localsizes:
                print(f"    - localsize={localsize}: ", end="", flush=True)
                
                # Run the benchmark
                result = run_benchmark(script, platform_idx, localsize)
                
                # Store the result
                results[platform_idx].loc[localsize, script_name] = result
                
                # Print the result
                if isinstance(result, float):
                    print(f"{result:.2f} GFLOPS")
                else:
                    print(result)
    
    # Print results as tables
    print("\n\n===== BENCHMARK RESULTS =====\n")
    for platform_idx, df in results.items():
        platform_name = platform_names[platform_idx]
        print(f"\nPlatform {platform_idx}: {platform_name}")
        
        # Format the DataFrame for display
        formatted_df = df.copy()
        for col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)
            )
        
        print(tabulate(formatted_df, headers="keys", tablefmt="grid"))
        
        # Save results to CSV
        csv_filename = f"benchmark_results_platform_{platform_idx}_N=8192.csv"
        df.to_csv(csv_filename)
        print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    main()