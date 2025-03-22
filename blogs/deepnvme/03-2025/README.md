<div align="center">

# DeepNVMe: Cost-effective I/O scaling for Deep Learning Applications.

</div>

# Introduction
We introduced [DeepNVMe](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepnvme/08-2024/README.md) in summer 2024 as a suite of optimizations for tackling I/O bottlenecks in Deep Learning (DL). DeepNVMe delivers significant speedups for I/O bound DL workloads by leveraging storage innovations including local NVMe SSDs, NVIDIA Magnum IO<sup>TM</sup> GPUDirect® Storage (GDS), and Linux Asynchronous I/O (libaio). In this update, we are delighted to announce DeepNVMe improvements on multiple fronts: (i) expanding application coverage to FastPersist model checkpointing and SGLang inference, (ii) performance scaling on faster NVMe SSDs, and (iii) expanding usability to CPU-only environments and offset-based I/O operations. The results reported in this blog are available in DeepSpeed versions >= XXX. 

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
Although model checkpointing to persistent storage is a critical task in model training, it is also a major performance bottleneck due to the inefficiencies of existing approaches. We have developed [FastPersist](https://arxiv.org/abs/2406.13768) to address the challenge of model checkpointing. FastPersist leverages DeepNVMe optimizations along with domain-specific techniques (e.g., data parallelism) to significantly reduce checkpoint writing costs and the associated model training slowdowns. We demonstrate FastPersist benefits using the popular PyTorch `torch.save()` functionality. 

### Faster Saving of PyTorch Tensors
In Figure~XXX we compare the latency of serializing PyTorch tensors to local NVMes using `torch.save()` and FastPersist. We observed YYY speedups for tensor sizes ZZZ. 

### Faster Saving of PyTorch Models
In Figure-XXX, we compare latency of saving model checkpoints using `torch.save()` and FastPersist. We observed YYY speedups for model examples, A, B, C, etc.

## ZeRO-Inference: Democratizing Generative AI
[ZeRO-Inference]() is a technique for democratizing access to state-of-the-art models by reducing the GPU costs of model inference. ZeRO-Inference enables inference computations of massive models (hundreds-of-billions of parameters) on as few as one GPU by offloading the model weights to DRAM and NVMe storage. ZeRO-Inference is designed for offline or throughput-oriented inference scenarios. In this blog, we share two updates on ZeRO-Inference. First, we have integrated ZeRO-Inference into SGLang, a state-of-the-art model serving framework. Second, we observed ZeRO-Inference performance scales with the faster NVMe SSDs in the latest Azure SKUs. 

### Democratizing SGLang through ZeRO-Inference integration
[SGLang](https://docs.sglang.ai/) is a state-of-the-art serving framework for large language models (LLMs) and vision language models (VLMs). Our integration of ZeRO-Inference into SGLang makes SGLang available to budget-constrained users, and offers a cost-reduction option for existing SGLang users. 

### Scaling HF Transformer Generation with Faster NVMe SSDs
We previously [evaluated](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-gds/README.md#high-performance-offloading-via-nvme-scaling)  LLAMA-3-70B generation on single GPU using HF Transformer inference and NVMe offloading. That experiment measured the generation speed for a prompt of 512 tokens, output of 32 tokens, and batch size 96. We used an Azure [NC_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series?tabs=sizebasic) VM with four Gen4 NVMes. Since NVMe bandwidth was the main bottleneck, we now repeat the experiment on newer Azure SKUs with eight Gen5 NVMes(ND-MI300X-v5 and ND-H200-v5). The results summarized in the Figure below show that ZeRO-Inference effectively exploits the increased NVMe bandwidths to improve generation speeds. For example, with GDS, generation speed improves from 7 tokens per second with four Gen4 NVMes to 17 with four Gen5 NVMes, and further to 26 with eight Gen5 NVMes. We observe similar improvements without GDS. These results show that ZeRO-Inference performance can be improved in cost-effective manner by increasing NVMe bandwidths. 

<img src="./media/hf_zinf_llama_70b.png">
<div align="center">
Figure: ZeRO-Inference leverages available NVMe bandwidth to scale LLAMA-3-70B generation. 
</div>


# I/O performance scaling
We used our `ds_io` benchmarking tool to demonstrate DeepNVMe proportionally scaling I/O performance with available NVMe bandwidths. This empowers users to accelerate I/O bound DL applications at modest cost using more or faster NVMe SSDs. In our experiments, we measure the achieved read and write bandwidths of 1GB data transfers between HBM and NVMes. We evaluate scaling up NVMes from PCIe Gen4 to Gen5, and scaling out from 4 to 8 SSDs. The SSDs are combined into a single RAID-0 volume. We summarize the results in the Figure below which show that DeepNVMe scales I/O performance on both dimensions. Scaling up from 4xGen4 SSDs to 4xGen5 SSDs improves reads from 10GB/sec to 27GB/sec, and writes from 5GB/sec to 11GB/sec. Scaling out from 4xGen5 to 8xGen5 further improves reads to 48GB/sec, and writes to 26GB/sec. 

<img src="./media/dnvme_scaling.png">
<div align="center">
Figure: Scaling I/O performance with available NVMe bandwidth
</div>



# DeepNVMe in CPU-Only environments
Previously, DeepNVMe was unusable in CPU-only environments because it relied on `torch.pin_memory()` for page-locked CPU tensors. However, `torch.pin_memory()` does not work in the CPU versions of `torch` as illustrated below. 

```bash
>>> import torch
>>> torch.__version__
'2.6.0+cpu'
>>> x = torch.empty(1024).pin_memory()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Cannot access accelerator device when none is available.
>>> 
```

We have extended DeepNVMe to usable in CPU environments by adding functionality for allocating (`new_cpu_locked_tensor()`)and releasing (`free_cpu_locked_tensor()`) page-locked CPU tensors. The snippet below illustrates using this functionality to obtain a pinned CPU tensor (`x`).

```bash
>> import torch
>>> torch.__version__
'2.6.0+cpu'
>>> from deepspeed.ops.op_builder import AsyncIOBuilder
>>> h = AsyncIOBuilder().load().aio_handle()
>>> x = h.new_cpu_locked_tensor(1024, torch.Tensor())
>>> x.shape
torch.Size([1024])
>>> x.dtype
torch.float32
```



