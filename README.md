# cnn-gpu-benchmarks
This repo benchmarks:
* architectures: [ResNet-152, ResNet-101, ResNet-50, and ResNet-18](https://pytorch.org/hub/pytorch_vision_resnet/)
* GPUs: [NVIDIA TITAN RTX](https://en.wikipedia.org/wiki/Nvidia_RTX), [EVGA (non-blower) RTX 2080 ti](https://www.evga.com/products/Specs/GPU.aspx?pn=0d3aca30-c8b9-47a8-8aa8-0dda2e7e7ac7), and [GIGABYTE (blower) RTX 2080 ti](https://www.gigabyte.com/us/Graphics-Card/GV-N208TTURBO-11GC-rev-10#kf)
* Datasets: [ImageNet](http://www.image-net.org/), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

This is a simplified version of [[this blog post](https://l7.curtisnorthcutt.com/benchmarking-gpus-for-deep-learning)]. The full post contains images of the workstation and additional details.

These benchmarks extend [Justin Johnson](https://scholar.google.com/citations?user=mS5k4CYAAAAJ&hl=en)'s [CNN benchmarks](https://github.com/jcjohnson/cnn-benchmarks) on the older GTX 1080 GPUs. Divide Justin's tables by 16 to compare with these because he reports millisecond-per-minibatch instead of millisecond-per-image.

### To account for thermal throttling I use three timing measurements
**Why?** Each measurement separates out ideal performance from actual performance by accounting for thermal throttling in isolation and thermal throttling due to surrounding GPU heat dissipation and [GPU positioning](https://l7.curtisnorthcutt.com/the-best-4-gpu-deep-learning-rig/gpu-positioning).

The three benchmark measurements:
1. **SECOND-BATCH-TIME**: Time to finish second training batch. This measures performance before the GPU has heated up, effectively no [thermal throttling](https://en.wikipedia.org/wiki/Dynamic_frequency_scaling).
2. **AVERAGE-BATCH-TIME**: Average batch time after 1 epoch in ImageNet or 15 epochs in CIFAR. This measures takes into account [thermal throttling](https://en.wikipedia.org/wiki/Thermal_design_power).
3. **SIMULTANEOUS-AVERAGE-BATCH-TIME**: Average batch time after 1 epoch in ImageNet or 15 epochs in CIFAR with all GPUs running simultaneously. Instead of [multi-GPU training](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html), each GPU is given its own independent training task. This measures the effect of thermal throttling in the system due to the combined heat given off by all GPUs.


## Key Takeaways

### TITAN RTX > RTX 2080 Ti
 - On ImageNet, the NVIDIA TITAN RTX trains around 10% faster across all benchmarks.
 - On CIFAR, all GPUs perform similarly **in isolation**, but the NVIDIA TITAN RTX trains 2x-4x faster when all GPUs are running simultaneously.
 - To train large-memory ConvNets like [VGG19](https://keras.io/applications/#vgg19) on large images of size 224x224, the TITAN RTX is required to fit everything in GPU memory. TITAN RTX has 24 GB memory whereas 2080 ti has only 11 GB.

### Thermal throttling affects GPU Performance
  - Across all model architectures, GPUs, batch sizes, and datasets, **SECOND-BATCH-TIME** (GPU just started running and is cool) is faster than **AVERAGE-BATCH-TIME** (GPU has been running a while and is hot) and **AVERAGE-BATCH-TIME** is faster than **SIMULTANEOUS-AVERAGE-BATCH-TIME** (multiple GPUs have been running a while next to each other and everything is very hot).
  - Decrease in performance is more significant from **AVERAGE-BATCH-TIME** to **SIMULTANEOUS-AVERAGE-BATCH-TIME** benchmarks.
  - With heat dissipation from all four GPUs (**SIMULTANEOUS-AVERAGE-BATCH-TIME**), the TITAN RTX processed a CIFAR image every 27 microseconds.
  - These results depends on the [case and cooling in the deep learning GPU rig](https://l7.curtisnorthcutt.com/build-pro-deep-learning-workstation) and [GPU positioning](https://l7.curtisnorthcutt.com/gpu-positioning).

### Batch-size affects Training Time
  - Decreasing the batch-size from 128 to 64 using **ResNet-152** on ImageNet with a TITAN RTX gpu, increased training time by around 3.7%.
  - Decreasing the batch-size from 256 to 128 using **ResNet-50** on ImageNet with a TITAN RTX gpu, did not affect training time.
  - The difference is likely due to CPU bottlenecking and architecture size.



## To Reproduce
All benchmarks were computed using PyTorch 2.0 on an Ubuntu 18.04 LTS, Intel 10-core i9 CPU machine with  128 GB memory running CUDA 10.1 with exact workstation specifications [[here](https://l7.curtisnorthcutt.com/the-best-4-gpu-deep-learning-rig)]. The code to reproduce these results is available: [[here for ImageNet](https://github.com/cgnorthcutt/cleanlab/blob/master/examples/imagenet/imagenet_train_crossval.py)], [[here for CIFAR-100](https://github.com/cgnorthcutt/cleanlab/blob/master/examples/cifar100/cifar100_train_crossval.py)], and [[here for CIFAR-10](https://github.com/cgnorthcutt/cleanlab/blob/master/examples/cifar10/cifar10_train_crossval.py)]. I've included copies in this repo.




## Benchmarks

Tables report values in milliseconds/image, where smaller implies better performance.

### ImageNet

For ImageNet, these batch sizes are evaluated:
* ResNet-152 with batch size = 32
* ResNet-101 with batch size = 64
* ResNet-50 with batch size = 64
* ResNet-18 with batch size = 128

Each table reports time in milliseconds/image, computed by taking the average time per batch / batch size. Results are reproducible via [[this ImageNet script](https://github.com/cgnorthcutt/cleanlab/blob/master/examples/imagenet/imagenet_train_crossval.py)]. To reproduce the results in this table, run something like this in your terminal:

```bash
python3 imagenet_train_crossval.py \
	--gpu {INTEGER OF GPU} \
	--arch resnet50 \
	--batch-size 64 \
	/location/of/imagenet/dataset/
```

where `--gpu` specifies the 0-based integer of the GPU you want to train with, `--arch` specifies the model architecture, and `/location/of/imagenet/dataset/` should be replaced with dataset location.

#### SECOND-BATCH-TIME (ImageNet) [milliseconds/image]

This measurement evaluates the speed of each GPU in isolation on the second training batch, before thermal throttling can occur. In terms of performance, Titan RTX > EVGA non-blower > GIGABYTE blower, with speeds increasing with smaller architectures.

| GPU | ResNet152 | ResNet101 | ResNet50 | ResNet18 |
| --- | ---------- | ---------- | --------- | --------- |
| **GIGABYTE (blower) RTX 2080 ti** | 8.69 | 6.06 | 3.73 | 1.12 |
| **EVGA (non-blower) RTX 2080 ti** | 8.56 | 5.78 | 3.61 | 1.09 |
| **NVIDIA TITAN RTX** | 7.91 | 5.55 | 3.52 | 1.02 |

NVIDIA TITAN RTX is around 10% faster in nearly all cases.

#### AVERAGE-BATCH-TIME (ImageNet)

This measurement evaluates the speed of each GPU in isolation after one epoch on ImageNet -- this gives the GPU plenty of time to heat up and takes into account thermal throttling due only to the GPU and airflow. Observe the decrease in performance compared to the table above due to thermal throttling.


| GPU | ResNet152 | ResNet101 | ResNet50 | ResNet18 |
| --- | --------- | --------- | -------- | -------- |
| **GIGABYTE (blower) RTX 2080 ti** | 8.97 | 5.92 | 3.86 | 1.31 |
| **EVGA (non-blower) RTX 2080 ti** | 8.78 | 5.92 | 3.78 | 1.34 |
| **NVIDIA TITAN RTX** | 8.34 | 5.64 | 3.56 | 1.28 |



#### SIMULTANEOUS-AVERAGE-BATCH-TIME (ImageNet)

This measurement includes an additional column to designate the **position** of the GPU in the machine. All 4 GPUs run simultaneously on the same training task independently, with two blower-style GPUs in the middle, the TITAN RTX on the bottom for increased airflow, and the non-blower GPU on the top. You can read about the effect of GPU positioning on thermal throttling [[here](https://l7.curtisnorthcutt.com/the-best-4-gpu-deep-learning-rig/gpu-positioning/)].


| GPU | Position | ResNet152 | ResNet101 | ResNet50 | ResNet18 |
| --- | -------- | ---------- | ---------- | --------- | --------- |
| **EVGA (non-blower) RTX 2080 ti** | top | 9.66 | 6.58 | 4.02 | 2.06 |
| **GIGABYTE (blower) RTX 2080 ti** | mid-top | 9.22 | 6.25 | 3.84 | 2.06 |
| **GIGABYTE (blower) RTX 2080 ti** | mid-bottom | 10.78 | 7.42 | 4.44 | 2.02 |
| **NVIDIA TITAN RTX** | bottom | 8.22 | 5.55 | 3.47 | 1.99 |

Observe an overall decrease in performance compared to the two tables above due to thermal throttling caused by heat dissipation from all GPUs running simultaneously. However, RTX NVIDIA TITAN improves the **SIMULTANEOUS-AVERAGE-BATCH-TIME** by 30% when compared to the mid-bottom 2080 ti.

### How does Batch Size effect Training Speed on ImageNet?

The table below reports the [SECOND-BATCH-TIME](#to-account-for-thermal-throttling-i-use-three-timing-measurements) and [AVERAGE-BATCH-TIME](#to-account-for-thermal-throttling-i-use-three-timing-measurements) benchmarks for the **NVIDIA TITAN RTX**, run in isolation.

| Architecture | ResNet152 | ResNet152 | ResNet101 | ResNet50 | ResNet50 |
| - | ---------- | ---------- | ---------- | --------- | --------- |
| **Batch size** | **128** | **64** | **128** | **256** | **128** |
| **[SECOND-BATCH-TIME](#to-account-for-thermal-throttling-i-use-three-timing-measurements)**  | 7.51 | 7.83 | 5.38 | 3.33 | 3.38 |
| **[AVERAGE-BATCH-TIME](#to-account-for-thermal-throttling-i-use-three-timing-measurements)** | 7.66 | 7.95 | 5.44 | 3.41 | 3.42 |

Observe a slight (at most 4%) decrease in performance on smaller batch sizes for both **SECOND-BATCH-TIME** and **AVERAGE-BATCH-TIME** benchmarks.


### CIFAR-100

For CIFAR-100 and CIFAR-10, the following are benchmarked:
* ResNet-152 with batch size = 256
* ResNet-50 with batch size = 256

Smaller batch sizes and/or model architectures are not benchmarked because GPU utilization is too low on CIFAR for significant differences in GPU performance.

Each table reports time in milliseconds/image, computed by taking the average time per batch / batch size. Results are reproducible via [[this CIFAR-100 script](https://github.com/cgnorthcutt/cleanlab/blob/master/examples/cifar100/cifar100_train_crossval.py)]. To reproduce these results, run something like this in your terminal:

```bash
python3 cifar100_train_crossval.py \
	--gpu {INTEGER OF GPU} \
	--arch resnet50 \
	--batch-size 256 \
	/location/of/cifar100/dataset/
```

#### SECOND-BATCH-TIME (CIFAR-100)

| GPU | RESNET-152 | RESNET-101 | RESNET-50 | RESNET-18 |
| --- | ---------- | ---------- | --------- | --------- |
| **GIGABYTE (blower) RTX 2080 ti** | 0.102 | 0.072 | 0.047 | 0.020 |
| **EVGA (non-blower) RTX 2080 ti** | 0.102 | 0.072 | 0.046 | 0.020 |
| **NVIDIA TITAN RTX** | 0.094 | 0.066 | 0.044 | 0.020 |

#### AVERAGE-BATCH-TIME (CIFAR-100)

For CIFAR, [AVERAGE-BATCH-TIME](#to-account-for-thermal-throttling-i-use-three-timing-measurements) is computed after the 15th epoch to compare with a similar number of total images trained as ImageNet.

| GPU | RESNET-152 | RESNET-101 | RESNET-50 | RESNET-18 |
| --- | ---------- | ---------- | --------- | --------- |
| **GIGABYTE (blower) RTX 2080 ti** | 0.104 | 0.075 | 0.047 | 0.020 |
| **EVGA (non-blower) RTX 2080 ti** | 0.103 | 0.072 | 0.047 | 0.020 |
| **NVIDIA TITAN RTX** | 0.097 | 0.070 | 0.044 | 0.017 |

The NVIDIA TITAN RTX trains 7% faster for ResNet-152. Overall, performance is similar across GPU and architecture.

#### SIMULTANEOUS-AVERAGE-BATCH-TIME (CIFAR-100)


| GPU | Position | ResNet152 | ResNet101 | ResNet50 | ResNet18 |
| --- | -------- | ---------- | ---------- | --------- | --------- |
| **EVGA (non-blower) RTX 2080 ti** | top | 0.117 | 0.111 | 0.125 | 0.095 |
| **GIGABYTE (blower) RTX 2080 ti** | mid-top | 0.082 | 0.080 | 0.088 | 0.072 |
| **GIGABYTE (blower) RTX 2080 ti** | mid-bottom | 0.052 | 0.047 | 0.055 | 0.044 |
| **NVIDIA TITAN RTX** | bottom | 0.027 | 0.027 | 0.027 | 0.027 |

While all GPUs perform similarly **in isolation** on CIFAR, the NVIDIA TITAN RTX trains 2x-4x faster when all GPUs are running simultaneously. Even with heat dissipation from all four GPUs, the NVIDIA TITAN RTX when placed at the bottom (so that its fans are not blocked by other GPUs) processes a CIFAR image with ResNet-152 every 27 microseconds.


### CIFAR-10

Each table reports time in milliseconds/image, computed by taking the average time per batch / batch size. Results are reproducible via [[this CIFAR-10 script](https://github.com/cgnorthcutt/cleanlab/blob/master/examples/cifar10/cifar10_train_crossval.py)]. To reproduce these results, run something like this in your terminal:

```bash
python3 cifar10_train_crossval.py \
	--gpu {INTEGER OF GPU} \
	--arch resnet50 \
	--batch-size 256 \
	/location/of/cifar10/dataset/
```

#### SECOND-BATCH-TIME (CIFAR-10)

| GPU | RESNET-152 | RESNET-101 | RESNET-50 | RESNET-18 |
| --- | ---------- | ---------- | --------- | --------- |
| **GIGABYTE (blower) RTX 2080 ti** | 0.097 | 0.072 | 0.047 | 0.020 |
| **EVGA (non-blower) RTX 2080 ti** | 0.096 | 0.072 | 0.043 | 0.020 |
| **NVIDIA TITAN RTX** | 0.091 | 0.065 | 0.040 | 0.021 |

#### AVERAGE-BATCH-TIME (CIFAR-10)

Like CIFAR-100, [AVERAGE-BATCH-TIME](#to-account-for-thermal-throttling-i-use-three-timing-measurements) is computed after the 15th epoch.

| GPU | RESNET-152 | RESNET-101 | RESNET-50 | RESNET-18 |
| --- | ---------- | ---------- | --------- | --------- |
| **GIGABYTE (blower) RTX 2080 ti** | 0.103 | 0.074 | 0.047 | 0.020 |
| **EVGA (non-blower) RTX 2080 ti** | 0.102 | 0.073 | 0.045 | 0.020 |
| **NVIDIA TITAN RTX** | 0.095 | 0.069 | 0.044 | 0.022 |


#### SIMULTANEOUS-AVERAGE-BATCH-TIME (CIFAR-10)


| GPU | Position | ResNet152 | ResNet101 | ResNet50 | ResNet18 |
| --- | -------- | ---------- | ---------- | --------- | --------- |
| **EVGA (non-blower) RTX 2080 ti** | top | 0.112 | 0.110 | 0.122 | 0.102 |
| **GIGABYTE (blower) RTX 2080 ti** | mid-top | 0.082 | 0.080 | 0.090 | 0.073 |
| **GIGABYTE (blower) RTX 2080 ti** | mid-bottom | 0.047 | 0.051 | 0.058 | 0.043 |
| **NVIDIA TITAN RTX** | bottom | 0.026 | 0.026 | 0.026 | 0.027 |

Like CIFAR-100, all GPUs perform similarly **in isolation** on CIFAR-10, but the NVIDIA TITAN RTX trains 2x-4x faster when all GPUs are running simultaneously.




### If instead of benchmarking, you're looking for...

1. Relative positioning of GPUs for optimal speed: [[view
this post](https://l7.curtisnorthcutt.com/gpu-positioning)]
2. How to build a multi-GPU deep learning machine: [[view this
post](https://l7.curtisnorthcutt.com/build-pro-deep-learning-workstation)]
3. Build Lambda's state-of-the-art 4-GPU rig for $4000 less: [[view this
post](https://l7.curtisnorthcutt.com/the-best-4-gpu-deep-learning-rig)]
