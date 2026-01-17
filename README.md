# Accelerate-TinyML-Convolution

## Overview
This project explores how to efficiently accelerate a 2D convolution kernel for TinyML workloads running on resource-constrained RISC-V processors. The goal is to reduce execution cycles and memory traffic while preserving correctness, relying on software-level optimizations with realistic cache configurations.

Techniques include loop tiling, cache-aware optimization, loop unrolling, and pointer arithmetic.

Record : [Accelerate TinyML Convolution](https://hackmd.io/TjglDhFYT6mSTQv2x2LDCg?view)

## Goals
* Establish a Baseline
* Optimize the Algorithm (Loop Tiling)
* Explore Cache Designs (Design Space Exploration)
* Apply Micro-Optimizations

## Performance
The total execution cycles are reduced from 421,710 in the baseline implementation to 21,620 in the final optimized version.
Overall, the optimized implementation achieves a 19.51× speedup over the naive baseline.

|     Optimization Stage      | Total Cycles | Speedup (vs. Baseline) |             Primary Bottleneck Addressed             |
|:---------------------------:|:------------:|:----------------------:|:----------------------------------------------------:|
|   Baseline (Naive C, -O0)   |   421,710    |         1.00×          |  Redundant memory accesses & loop control overhead   |
| Compiler Optimization (-O2) |    98,445    |         4.28×          |  Instruction count reduction & register allocation   |
|     Loop Tiling (16×4)      |    98,901    |         4.26×          | Data locality improvement & conflict miss mitigation |
|     Micro-Optimizations     |   21,620     |         19.51×         |   Load-use hazards & address computation overhead    |

## Workload (brief)
- Input: 32×32 single-channel image  
- Kernel: 3×3 Sobel  
- Output: 30×30 (valid, stride=1)

## Quick start
```bash
git clone https://github.com/RayYang421/Accelerate-TinyML-Convolution
cd Accelerate-TinyML-Convolution
```

**make**
```
make        # compile C program
make run    # run 
make valid  # validate correctness with Python script
```

## Project Structure
```
├── Baseline/
│ └── Baseline.c
│
├── Optimize/
│ └── Optimize.c # Loop-tiling implementation
│
├── Micro-Optimizations/
│ ├── Loop_Unrolling.c
│ ├── Pointer_Arithmetic.c
│ ├── StrengthReduction.c
│ └── Hoisting_Repeated_Loads.c
│
├── Accelerate-TinyML-Convolution.c # Main integration file
├── makefile
├── image.txt           # Input image
├── preprocess.py       # Preprocessing utilities
└── Valid.py            # Python correctness validation
```

**File contains**
- **`Baseline/`** – Naive 3×3 convolution (4-level nested loops).
- **`Optimize/`** – Loop-tiling version (e.g., 16×4 tiles).
- **`Micro-Optimizations/`** – low-level optimizations:
  - Loop unrolling  
  - Pointer arithmetic  
  - Strength reduction  
  - Hoisting repeated loads  
- **`Accelerate-TinyML-Convolution.c`** – final integrated version.
- **`makefile`** – Build, run, and validation scripts.
- **`image.txt`** – Input image data.
- **`preprocess.py`** – Image preprocessing.
- **`Valid.py`** – Python reference convolution + numerical validation.

## Environment
- [Ripes](https://github.com/mortbopet/Ripes) : 5-stage pipeline
- [riscv-none-elf-gcc-15.2.0](https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack)
- Python 3.10
- Cache : LRU, 2-word block, 2-way set-associative D-cache (8 lines)
