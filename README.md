# CUDA Training Repository

This repository contains a comprehensive collection of CUDA programming examples and exercises, organized into progressive learning modules. The examples cover fundamental CUDA concepts from basic memory management to advanced optimization techniques.

## 📚 Repository Structure

### 📁 cuda00 - Fundamentals
Basic CUDA programming concepts and foundational examples.

#### 🔧 Memory Management (`allocate/`)
- **`double-elements.cu`** - Basic unified memory allocation and kernel execution
- **`grid-stride-double.cu`** - Grid-stride loop pattern for handling large datasets
- **`mismatched.cu`** - Preventing out-of-bounds array access in kernels

#### ⚠️ Error Handling (`errors/`)
- **`error-handler.cu`** - CUDA error checking and debugging techniques

#### 🎨 Computational Examples (`fractal/`)
- **`fractal.cu`** - Complex fractal generation using CUDA (C++)
- **`fractal.py`** - Python implementation using Numba for comparison
- **`fractal.png`** - Generated fractal output

#### 🌡️ Scientific Computing (`heat_conduction/`)
- **`01-heat-conduction.cu`** - 2D heat equation solver with finite difference method

#### 🔢 Linear Algebra (`Matrix/`)
- **`01-matrix-multiply-2d.cu`** - 2D matrix multiplication implementation

#### ➕ Vector Operations (`vectors-add/`)
- **`01-vector-add.cu`** - Basic vector addition kernel
- **`vector.cu`** & **`vector.cuh`** - Modular vector operations
- **`test.py`** - Python testing utilities

### 📁 cuda01 - Advanced Techniques
Advanced CUDA programming concepts and optimization strategies.

#### ⚡ Vector Addition Optimization (`01-vector-add/`)
- **`01-vector-add.cu`** - Basic vector addition with performance profiling
- **`01-vector-add-init-in-kernel-solution.cu`** - Kernel-based initialization
- **`01-vector-add-prefetch-solution.cu`** - Memory prefetching optimization
- **`02-vector-add-prefetch-solution-cpu-also.cu`** - CPU-GPU memory prefetching

#### 🔍 Device Information (`04-device-properties/`)
- **`01-get-device-properties.cu`** - Query GPU device capabilities and specifications

#### 📄 Memory Management (`06-unified-memory-page-faults/`)
- **`01-page-faults.cu`** - Understanding unified memory page fault behavior

#### 🧮 SAXPY Operations (`saxpy/`)
- **`02-saxpy.cu`** - SAXPY (Single-precision A*X Plus Y) implementation
- **`02-saxpy-solution.cu`** - Optimized SAXPY with performance tuning

## 🎯 Learning Objectives

### Beginner Level (cuda00)
- **Memory Management**: Learn unified memory allocation with `cudaMallocManaged()`
- **Kernel Basics**: Understand thread indexing and execution configuration
- **Error Handling**: Implement proper CUDA error checking
- **Grid-Stride Loops**: Handle datasets larger than the grid size
- **2D Computations**: Work with 2D thread blocks and grids

### Advanced Level (cuda01)
- **Performance Optimization**: Use profiling tools (nsys) for optimization
- **Memory Prefetching**: Optimize data movement between CPU and GPU
- **Device Queries**: Programmatically access GPU specifications
- **Unified Memory**: Understand page fault behavior and optimization
- **Performance Tuning**: Achieve specific performance targets

## 🛠️ Building and Running

### Prerequisites
```bash
# NVIDIA CUDA Toolkit (11.0+)
# Compatible GPU with compute capability 3.5+
# GCC/G++ compiler
```

### Compilation
```bash
# Basic compilation
nvcc -o program_name source_file.cu

# With optimization flags
nvcc -O3 -o program_name source_file.cu

# For debugging
nvcc -g -G -o program_name source_file.cu
```

### Example Usage
```bash
# Navigate to specific example
cd cuda00/vectors-add

# Compile and run
nvcc -o vector_add 01-vector-add.cu
./vector_add

# Profile with nsys (for cuda01 examples)
cd cuda01/saxpy
nvcc -o saxpy 02-saxpy.cu
nsys profile --stats=true ./saxpy
```

## 📊 Performance Profiling

Many examples are designed to work with NVIDIA Nsight Systems:

```bash
# Generate profiling report
nsys profile --stats=true --output=profile_report ./your_program

# View timeline
nsys-ui profile_report.nsys-rep
```

## 🧪 Key Concepts Demonstrated

### Memory Models
- **Unified Memory**: Automatic data movement between CPU and GPU
- **Memory Prefetching**: Explicit control over data placement
- **Page Fault Analysis**: Understanding memory access patterns

### Execution Patterns
- **Grid-Stride Loops**: Scalable kernel design for variable data sizes
- **2D Thread Organization**: Efficient mapping for matrix operations
- **Occupancy Optimization**: Maximizing GPU utilization

### Error Handling
- **Runtime Error Checking**: `cudaGetLastError()` and `cudaDeviceSynchronize()`
- **Bounds Checking**: Preventing memory access violations
- **Debugging Techniques**: Using CUDA error checking macros

## 📖 Recommended Learning Path

1. **Start with `cuda00/allocate/`** - Master basic memory management
2. **Practice with `cuda00/vectors-add/`** - Understand kernel execution
3. **Explore `cuda00/Matrix/`** - Learn 2D computations
4. **Study `cuda00/errors/`** - Implement proper error handling
5. **Advance to `cuda01/01-vector-add/`** - Learn optimization techniques
6. **Profile with `cuda01/saxpy/`** - Practice performance tuning
7. **Experiment with `cuda01/06-unified-memory-page-faults/`** - Master memory optimization

## 🔧 Common Compilation Issues

### Solution for Common Problems
```bash
# If encountering compute capability issues
nvcc -arch=sm_60 -o program source.cu

# For older GPUs
nvcc -arch=sm_35 -o program source.cu

# For newer GPUs (RTX series)
nvcc -arch=sm_75 -o program source.cu
```

## 📝 Notes

- Examples include both working implementations and exercises requiring completion
- Performance targets are provided for optimization exercises
- Code includes extensive comments explaining CUDA concepts
- Python equivalents provided for some algorithms (e.g., fractal generation)

## 🤝 Contributing

When adding new examples:
1. Include comprehensive comments
2. Provide error checking
3. Add performance notes where applicable
4. Consider both educational value and practical application

## 📚 Additional Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)

---
*This repository serves as a hands-on learning resource for CUDA programming, from basic concepts to advanced optimization techniques.*