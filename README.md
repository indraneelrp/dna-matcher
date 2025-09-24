# dna-matcher
matching virus DNA nucleotides from phred scores. Parallelised with CUDA

### Notes
This project was run on a NUS slurm cluster which provided access to NVIDIA GPUs as part of a parallel computing course, so it is difficult to replicate on a personal laptop. However, the main logic is in this repo!
.

### How it works
Essentially a brute force algorithm that does the following:

```python
for every sample: 
    for every signature:
          Go through string,
          if signature stops matching on current char, 
          then advance to next char
          & start trying to match 
          from start char of signature
         
         if match score exceeds, 
         update best match 
         append to results once all done
```

**Parallelisation strategy**:<br>
Each block handles 1 sample, each thread in a block handles 1 signature
Threads per block: Dynamically set to the nearest multiple of 32 of the given number of signatures<br>
<br>
**How memory is handled**<br>
Arrays of pointers are allocated on the host to store input data.<br>
Using cudaMalloc(), memory is allocated on the device to store input data and intermediate results for each match.<br>
Input data is then copied from the host onto the device local memory.<br>
After the kernel function is executed, the results are copied from the device back to the host.<br>
At the end of the program, memory is freed.
