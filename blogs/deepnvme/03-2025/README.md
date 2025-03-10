<div align="center">

# DeepNVMe: Cost-effective I/O scaling for Deep Learning Applications.

</div>

# Introduction
We introduced [DeepNVMe](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepnvme/08-2024/README.md) in summer 2024 as a suite of optimizations for tackling I/O bottlenecks in Deep Learning (DL). DeepNVMe leverages storage innovations including local NVMe SSDs, NVIDIA Magnum IO<sup>TM</sup> GPUDirect® Storage (GDS), and Linux Asynchronous I/O (libaio) for significant speedups for I/O bound DL workloads. In this update, we are delighted to announce DeepNVMe improvements on multiple dimensions: (i) expanding application coverage to FastPersist model checkpointing and SGLang inference, (ii) performance scaling on faster NVMe SSDs, and (iii) improving usability to CPU-only environments and offset-based I/O operations.

# Evaluation environments
Our experiments are conducted on Azure using VMs from the [ND-H200-v5](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nd-h200-v5-series?tabs=sizebasic) and [ND-MI300X-v5](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ndmi300xv5-series?tabs=sizebasic) SKUs. The software configuration is summarized in the following table. 

|Software | Version 
|---|--|
|PyTorch | 2.6.0+cu126|
|CUDA | 12.6 |
|Ubuntu | 24.0.2|


# Application 
## FastPersist: Faster Model Checkpoint Creation
Although model checkpointing to persistent storage is a critical task in model training, it is also a major performance bottleneck due to the inefficiencies of existing approaches. We have developed [FastPersist](https://arxiv.org/abs/2406.13768) to address the challenge of model checkpointing. FastPersist leverages DeepNVMe optimizations along with domain-specific techniques (e.g., data parallelism) to significantly reduce checkpoint writing costs and the associated model training slowdowns. We demonstrate FastPersist benefits using the popular PyTorch `torch.save()` checkpointing function. 

### Faster Saving of PyTorch Tensors


### Faster Saving of PyTorch Models


## ZeRO-Inference: Affordable SGLang Generation


# I/O performance scaling with faster NVMes


# Usability improvements




