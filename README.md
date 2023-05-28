# Wavelet-Conditional-Renormalization-Group
Code for paper "Conditionally Strongly Log-Concave Generative Model".

This repository implements fast wavelet transform and wavelet packets transform in pytorch.
As well, it implements the wavelet conditional renormalisation group with score matching.  

# Install
For fast wavelet transform and wavelet packets transform in pytorch download the folder *Wavelet_packets* :

```python
import sys
sys.path.append('~/where/you/download/the/script/')
``` 
Then, import it:
```python
import Wavelet_Packet
```
For wavelet conditional renormalisation group, as it is a package that uses Wavelet_Packet, you must download  both folders *Wavelet_Packet* and *WCRG*

```python
import sys
sys.path.append('~/where/you/download/the/script/')
``` 
Then, import both:
```python
import Wavelet_Packet
import WCRG
```

# Running Examples

The folder  *Notebooks Examples* contains a tutorial for the use of wavelet packets and fast wavelet transform. As well, an example of how to use the model to learn energies, free energies and sample, for $\varphi^4$ at critical point. 

# Using the software

You are free to use this software for academic purposes only. If you do so, please cite paper "Conditionally Strongly Log-Concave Generative Model".
Do not hesitate to message me if you have any question.
