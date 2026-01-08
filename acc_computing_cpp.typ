
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

#linebreak()
== Extending Standard algorithms

- Using Iterators will help reduce the \# of memory accesses. 
- Examples include counting iterators, zip iterators, transform iterators.
- Transform iterator - applies a function before returning a value. Zip Iterator - references 2 sequences at once.
- Using pointers means you will access memory every single time subscript operator is used. Using a counting iterator will just return the index at which the relevant data is stored. 
#image("./acc_comp_pics/6.png", width: 80%) // Make sure this file exists  
#pagebreak()
Thrust has inbuilt operators which we can use to perform reduction algorithms, speeding up the execution of the code.
#image("./acc_comp_pics/7.png", width: 80%) // Make sure this file exists  
#image("./acc_comp_pics/8.png", width: 80%) // Make sure this file exists  


#pagebreak()
== Vocabulary Types

- In the below picture, the stencil is the id which is used to find the 2D coordinates of the data.
- Using Thrust containers, using data() will return a typed iterator while std will return a raw pointer.
- Thrust's typed iterator can be converted to a raw pointer using thrust::raw_pointer_cast
#image("./acc_comp_pics/9.png", width: 80%) // Make sure this file exists  
#linebreak()
- Thrust Tabulate can be used to apply a common algorithm which can help parallelize the operation on the GPU.
#image("./acc_comp_pics/10.png", width: 80%) // Make sure this file exists  
#pagebreak()
- Below, we call std::make_pair, a host function, in a host, device function. This is not allowed and will not compile.
#image("./acc_comp_pics/11.png", width: 80%) // Make sure this file exists  
=== libcu++
- To fix this, we use the libcu++ library.
- This contains common std library types such as cuda::std::pair instead of std::pair.
#image("./acc_comp_pics/12.png", width: 80%) // Make sure this file exists  

=== mdspan
- disadvantages of manual linearization become more apparent as number of dimensions increases.
- cuda::std::mdspan is a non owning multidimensional view of sequence.
- mdspan is just a 2D view of the below array, cuda::std::array is the raw data structure which stores the data.
#image("./acc_comp_pics/13.png", width: 80%) // Make sure this file exists  
- The operator() can be used to access the underlying sequence: md(0,0) = 0, md(1, 2) = 5
- size() return the total number of elments in the view
- extent(r) returns extent at rank r, md.extent(0) = 2, md.extent(1) = 3

