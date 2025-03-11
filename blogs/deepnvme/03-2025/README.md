<div align="center">

# DeepNVMe: Cost-effective I/O scaling for Deep Learning Applications.

</div>

# Introduction
We introduced [DeepNVMe](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepnvme/08-2024/README.md) in summer 2024 as a suite of optimizations for tackling I/O bottlenecks in Deep Learning (DL). DeepNVMe leverages storage innovations including local NVMe SSDs, NVIDIA Magnum IO<sup>TM</sup> GPUDirect® Storage (GDS), and Linux Asynchronous I/O (libaio) for significant speedups for I/O bound DL workloads. In this update, we are delighted to announce DeepNVMe improvements on multiple dimensions: (i) expanding application coverage to FastPersist model checkpointing and SGLang inference, (ii) performance scaling on faster NVMe SSDs, and (iii) improving usability to CPU-only environments and offset-based I/O operations. The results reported in this blog are available in DeepSpeed versions >= XXX. 

# Evaluation environments
Our experiments are conducted on Azure using VMs from the [ND-H200-v5](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nd-h200-v5-series?tabs=sizebasic) and [ND-MI300X-v5](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ndmi300xv5-series?tabs=sizebasic) SKUs. The key software configurations are summarized in the following table. 

|Software | Version 
|---|--|
|PyTorch | 2.6.0+cu126|
|CUDA | 12.6 |
|Ubuntu | 24.0.2|


# Addressing I/O Bottlenecks of Deep Learning
We used DeepNVMe to develop FastPersist and ZeRO-Inference to target I/O challenges in DL training and inference respectively. 

## FastPersist: Faster Model Checkpoint Creation
Although model checkpointing to persistent storage is a critical task in model training, it is also a major performance bottleneck due to the inefficiencies of existing approaches. We have developed [FastPersist](https://arxiv.org/abs/2406.13768) to address the challenge of model checkpointing. FastPersist leverages DeepNVMe optimizations along with domain-specific techniques (e.g., data parallelism) to significantly reduce checkpoint writing costs and the associated model training slowdowns. We demonstrate the speed benefits of FastPersist using the popular PyTorch `torch.save()` checkpointing function. 

### Faster Saving of PyTorch Tensors
In Figure~XXX we compare the latency of serializing PyTorch tensors to persistent storage using `torch.save()` and FastPersist. We observed YYY speedups for tensor sizes ZZZ. 

### Faster Saving of PyTorch Models
In Figure-XXX, we compare latency of saving model checkpoints using `torch.save()` and FastPersist. We observed YYY speedups for model examples, A, B, C, etc.

## ZeRO-Inference: Democratizing Generative AI
[ZeRO-Inference]() is a technique for democratizing access to state-of-the-art models by reducing the GPU costs of model inference. ZeRO-Inference enables inference computations of massive models (hundreds-of-billions of parameters) on as few as one GPU by offloading the model weights to DRAM and NVMe storage. ZeRO-Inference is designed for offline or throughput-oriented inference scenarios. In this blog, we share two updates on ZeRO-Inference. First, we have integrated ZeRO-Inference into SGLang, a state-of-the-art model serving framework. Second, we observed ZeRO-Inference performance scales with the faster NVMe SSDs in the latest Azure SKUs. 

### Democratizing SGLang through ZeRO-Inference integration
[SGLang](https://docs.sglang.ai/) is a state-of-the-art serving framework for large language models (LLMs) and vision language models (VLMs). Our integration of ZeRO-Inference into SGLang makes SGLang available to budget-constrained users, and offers a cost-reduction option for existing SGLang users. 

### Scaling Generation Throughput with Faster NVMe SSDs
We previously [reported](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-gds/README.md#high-performance-offloading-via-nvme-scaling) ZeRO-Inference generation speeds of 6-7 tokens/sec for LLAMA-3-70B using a single GPU on an older [Azure SKU](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series?tabs=sizebasic). We also observed that main bottleneck was the available NVMe bandwidth. We repeated the experiment on newer Azure SKUs with roughly double the NVMe bandwidths, and present the results in FigureXXX below.  We observe a 2X generation speedup with the faster NVMes, achieving XX-YY tokens/sec. These results show that ZeRO-Inference performance can be scaled in a cost-effective manner by upgrading the available NVMe SSDs.  

# I/O performance scaling with faster NVMes


# Usability improvements




