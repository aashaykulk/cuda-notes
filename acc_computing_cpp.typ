
// ---------- basic style ----------  
#import "@preview/minimal-note:0.10.0": minimal-note

#show: minimal-note.with(
  title: [Getting Started with Accelerated Computing in Modern CUDA C++],
  author: [Aashay Kulkarni],
  date: datetime.today().display("[month repr:long], [year]")
)
// #set page(
//   paper: "a4", 
//   margin: 2.0cm,
//   header: [_Aashay Kulkarni's CUDA notes_ #line(length: 100%)], 
//   numbering: "1",
//   align: "justified",
// )  
// #set text(  
//   font: "Times New Roman",  // Ensure this font is installed or use "DejaVu Sans Mono"  
//   size: 10pt,  
// )  
// #set par(
//   justify: true,  
//   first-line-indent: 0pt,  
// )  
  
// ---------- notes ----------  
  
  

#pagebreak()

= Introduction

- GPU has higher latency than the CPU, however, it has higher bandwidth. (Think of Bus vs Car)
- For problems with high amounts of data (i.e. for really large *N*) on which simultaneous computations can be performed (known as Parallel Computing) GPU Programming is better.

- C++ Code is compiled into architecture specific instructions (x86, ARM, CUDA) by a compiler.
#image("./acc_comp_pics/1.png", width: 80%) // Make sure this file exists  

NVCC (NVIDIA CUDA Compiler Driver) turns C++ Code into GPU instructions. However, the GPU specific code must be stated explicitly. Execution Spaces are divided into host (CPU) and device (GPU). By default, code is run on the host side. By invoking special functions understood by the GPU compiler known as _kernels_, one can switch the execution space to a GPU.
#image("./acc_comp_pics/2.png", width: 80%) // Make sure this file exists  
CUDA provides lots of GPU accelerated libraries such as Thrust, cuBLAS, cuB, libcu++, cuDNN.
#pagebreak()

== Thrust Library

Thrust simplifies GPU Programming by offering pre-built components for simple tasks.
#image("./acc_comp_pics/3.png", width: 80%) // Make sure this file exists  
#image("./acc_comp_pics/4.png", width: 80%) // Make sure this file exists  

== Thrust Execution Policy
The first parameter to ```cpp thrust::transform()``` tells Thrust where to run the algorithm. thrust::host executes algorithms on the CPU, while thrust::device executes algorithms on the GPU.

- Execution Policy vs Execution Specifier: 
  - Execution space specifier ```c  (__host__, __device__)``` indicates where the code can run and it works at compile time.
  - Execution Policy: It works at runtime and indicates where the code will run. It doesn't automatically compile code for that location.
#image("./acc_comp_pics/5.png", width: 80%) // Make sure this file exists  

