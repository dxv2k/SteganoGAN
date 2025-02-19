# SteganographyUtils Class Guide

The `SteganographyUtils` class provides a collection of utility functions that are essential for a steganography pipeline. It includes methods to convert text to bits and vice versa, compress and encode text using Reed-Solomon error correction, and calculate the Structural Similarity Index (SSIM) between images.

## Table of Contents

- [Overview](#overview)
- [Initialization](#initialization)
- [Text and Bit Conversion](#text-and-bit-conversion)
  - [text_to_bits](#text_to_bits)
  - [bits_to_text](#bits_to_text)
- [Bytearray and Bit Conversion](#bytearray-and-bit-conversion)
  - [bytearray_to_bits](#bytearray_to_bits)
  - [bits_to_bytearray](#bits_to_bytearray)
- [Text Compression and Error Correction](#text-compression-and-error-correction)
  - [text_to_bytearray](#text_to_bytearray)
  - [bytearray_to_text](#bytearray_to_text)
- [SSIM Calculation Utilities](#ssim-calculation-utilities)
  - [gaussian](#gaussian)
  - [create_window](#create_window)
  - [ssim and _ssim](#ssim-and-_ssim)
- [Other Methods](#other-methods)
  - [first_element](#first_element)
- [Demo Code](#demo-code)

---

## Overview

`SteganographyUtils` is designed to help with:

- **Text-to-bit conversion:** Convert plain text into a bit representation.
- **Error correction:** Use Reed-Solomon coding along with zlib compression for robust encoding.
- **Image similarity:** Compute SSIM (Structural Similarity Index Measure) for comparing images.
- **Miscellaneous tasks:** Other helper functions used in the steganography process.

---

## Initialization

```python
def __init__(self, rs_block_size=250):
    self.rs = RSCodec(rs_block_size)
```

- **Purpose:**  
  Initializes the utility class by creating an instance of `RSCodec` with a specified Reed-Solomon block size.
- **Parameter:**  
  - `rs_block_size`: Block size used in error correction encoding (default is 250).

---

## Text and Bit Conversion

### text_to_bits

```python
def text_to_bits(self, text):
    return self.bytearray_to_bits(self.text_to_bytearray(text))
```

- **Purpose:**  
  Converts a given text string into its bit representation.
- **How it works:**  
  1. Converts the text into a compressed bytearray with error correction.
  2. Converts the bytearray into a list of bits.

### bits_to_text

```python
def bits_to_text(self, bits):
    return self.bytearray_to_text(self.bits_to_bytearray(bits))
```

- **Purpose:**  
  Converts a list of bits back into a human-readable text string.
- **How it works:**  
  1. Groups bits back into a bytearray.
  2. Decodes and decompresses the bytearray back into text.

---

## Bytearray and Bit Conversion

### bytearray_to_bits

```python
def bytearray_to_bits(self, x):
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result
```

- **Purpose:**  
  Converts a bytearray into a list of bits (0s and 1s).
- **Details:**  
  Each byte is converted into its 8-bit binary representation and padded with leading zeros if needed.

### bits_to_bytearray

```python
def bits_to_bytearray(self, bits):
    ints = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))
    return bytearray(ints)
```

- **Purpose:**  
  Converts a list of bits back into a bytearray.
- **Details:**  
  Groups every 8 bits, converts the group into an integer, and forms a bytearray.

---

## Text Compression and Error Correction

### text_to_bytearray

```python
def text_to_bytearray(self, text):
    assert isinstance(text, str), "Expected a string."
    x = zlib.compress(text.encode("utf-8"))
    x = self.rs.encode(bytearray(x))
    return x
```

- **Purpose:**  
  Converts a string into a bytearray while compressing and adding Reed-Solomon error correction.

### bytearray_to_text

```python
def bytearray_to_text(self, x):
    try:
        text = self.rs.decode(x)
        text = zlib.decompress(text)
        return text.decode("utf-8")
    except Exception as e:
        print(f"Error during decoding: {e}")
        return False
```

- **Purpose:**  
  Decodes a bytearray back to text, reversing the error correction and decompression.

---

## SSIM Calculation Utilities

### gaussian

```python
def gaussian(self, window_size, sigma):
    _exp = [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    gauss = torch.Tensor(_exp)
    return gauss / gauss.sum()
```

- **Purpose:**  
  Creates a 1D Gaussian window (kernel) for SSIM computation.

### create_window

```python
def create_window(self, window_size, channel):
    _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
```

- **Purpose:**  
  Creates a 2D Gaussian window using the 1D kernel.

---

## Demo Code

```python
import torch
from math import exp
import zlib
from reedsolo import RSCodec

stego_utils = SteganographyUtils(rs_block_size=250)

# Convert text to bits and back
text = "Hello, Steganography!"
bits = stego_utils.text_to_bits(text)
recovered_text = stego_utils.bits_to_text(bits)
print("Recovered text:", recovered_text)

# SSIM Calculation
img1 = torch.rand(1, 3, 256, 256)
img2 = img1.clone()  # identical image for perfect similarity
ssim_value = stego_utils.ssim(img1, img2)
print("SSIM between identical images:", ssim_value.item())
```
