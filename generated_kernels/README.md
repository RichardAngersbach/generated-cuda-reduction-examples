# Generated CUDA Reduction Kernels

This repository contains **automatically generated CUDA reduction kernels** for experimentation and demonstration.
The code is **not intended to compile or run as-is**. Instead, it is meant to illustrate different **GPU reduction strategies** and **kernel variants** produced by a code-generation workflow.

The kernels demonstrate multiple implementation techniques for reductions such as atomics, warp shuffles, and CUB-based approaches. They are useful for studying code structure, comparing reduction strategies, or serving as examples for further development.

## Important Note

⚠️ The kernels in this repository **cannot be compiled directly**.

- They were extracted from larger software frameworks.
- Some **framework-specific code and dependencies were intentionally removed**.
- In some cases only the **device kernels** are provided, while the surrounding host-side wrapper code is omitted.

The repository should therefore be viewed as a **reference for generated kernel variants**, not as a ready-to-build project.

## Repository Structure

```
generated_kernels/
├── parflow_vdotprod/
│ ├── base/
│ └── gridstrided_loops/
└── waLBerla_fusedresidualnorm/
  ├── base/
  |── gridstrided_loops/
```

Each kernel collection contains several CUDA implementations of the same reduction using different optimization strategies.

## Kernel Collections

### `parflow_vdotprod`

Implements a **vector dot product reduction**

\[
r = x \cdot y
\]

This kernel originates from the **ParFlow** simulation framework.

The original kernel wrapper code used to launch the CUDA kernels was relatively low-level and framework-independent, therefore **the wrapper kernels are included** in this directory.

### `waLBerla_fusedresidualnorm`

Implements a **fused residual computation and L2 norm reduction** for a linear system.

First the residual is computed:

\[
res = b - A x
\]

Then its **L2 norm** is reduced.

This kernel originates from the **waLBerla** framework.
The **host-side wrapper for launching the CUDA kernel was omitted**, since it is highly specific to waLBerla's infrastructure.

## Implementation Variants

Each kernel directory contains two subfolders:

- **`base/`**
- **`gridstrided_loops/`**

These represent two different kernel structures:

| Folder | Description |
|------|-------------|
| `base` | Standard kernel implementations |
| `gridstrided_loops` | Implementations using grid-strided loops |

Within each of these folders, multiple **reduction strategies** are provided:

| Variant | Description |
|-------|-------------|
| `atomic` | Global-memory atomic reduction |
| `cub_1D` | CUB-based reduction (1D CUDA grid and 1D CUDA blocks) |
| `cub_1Dblocks` | CUB-based reduction (3D CUDA grid and CUDA blocks) |
| `cub` | General CUB-based reduction |
| `shfl` | Warp-level reduction via shuffles |
| `shfl2` | Block-level reduction |

These variants illustrate different **performance-oriented design patterns for GPU reductions**.

## Purpose of the Repository

This repository is intended for:

- Demonstrating **generated CUDA kernels**
- Comparing **reduction implementation strategies**
- Studying **GPU kernel structures**