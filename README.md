# Cross-Channel Attention Wave-U-Net

PyTorch implementation of:

[A Cross-Channel Attention-Based Wave-U-Net for Multi-Channel Speech Enhancement](https://www.isca-archive.org/interspeech_2020/ho20_interspeech.html)

---

## Overview

This repository provides an implementation of a **multi-channel speech enhancement model** based on Wave-U-Net with a **cross-channel attention mechanism**.

The model operates in the **time domain** and extends the standard Wave-U-Net by:

- Using **separate encoders for each input channel**
- Introducing **cross-channel attention blocks** to exploit spatial information
- Fusing multi-channel features at both **encoder and bottleneck stages**

---

## Model Architecture

The model follows a U-Net structure with the following key components:

### 1. Dual Encoder
- Each microphone channel is processed independently
- Preserves spatial information before fusion

### 2. Cross-Channel Attention
- Applied after each encoder block
- Learns relationships between channels
- Enhances speech-dominant regions and suppresses noise

### 3. Bottleneck Fusion
- Concatenates features from all channels
- Uses 1D convolution to merge information

### 4. Decoder
- Standard Wave-U-Net style upsampling
- Skip connections use **attention-enhanced features**

---

## Input / Output

- **Input**: multi-channel waveform  
```

[B, C, T]

```
where:
- `B`: batch size  
- `C`: number of channels (typically 2)  
- `T`: number of samples  

- **Output**: enhanced single-channel waveform  
```

[B, 1, T]

````

---

## Algorithmic Latency

- Latency ≈ **0.5 seconds**
- Not suitable for real-time applications without modification

---

## Usage

```python
from model import CrossChannelWaveUNet
import torch

model = CrossChannelWaveUNet()

x = torch.randn(4, 2, 16384)  # batch of 2-channel audio
y = model(x)

print(y.shape)  # [4, 1, 16384]
```

---

## Notes

* This implementation follows the paper structure but includes practical adjustments:

  * BatchNorm added for stability
  * Linear interpolation for upsampling
  * Padding used instead of cropping
* Some architectural details are not fully specified in the paper and are adapted from standard Wave-U-Net implementations

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ho20_interspeech,
  title     = {A Cross-Channel Attention-Based Wave-U-Net for Multi-Channel Speech Enhancement},
  author    = {Minh Tri Ho and Jinyoung Lee and Bong-Ki Lee and Dong Hoon Yi and Hong-Goo Kang},
  booktitle = {Interspeech 2020},
  pages     = {4049--4053},
  year      = {2020},
  doi       = {10.21437/Interspeech.2020-2548}
}
```

---

## Reference

Paper link:
[https://www.isca-archive.org/interspeech_2020/ho20_interspeech.html](https://www.isca-archive.org/interspeech_2020/ho20_interspeech.html)

Code link:
[https://github.com/haoxiangsnr/Wave-U-Net-for-Speech-Enhancement/blob/master/model/unet_basic.py](https://github.com/haoxiangsnr/Wave-U-Net-for-Speech-Enhancement/blob/master/model/unet_basic.py)

