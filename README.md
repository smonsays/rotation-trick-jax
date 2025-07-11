# Rotation Trick JAX

A JAX implementation of the [rotation trick](https://arxiv.org/abs/2410.06424) for improved gradient estimation in VQ-VAEs.


## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/smonsays/rotation-trick-jax.git
```

Or with uv:

```bash
uv add git+https://github.com/smonsays/rotation-trick-jax.git
```

## Usage

The rotate_to function applies the rotation with appropriate internal gradient stopping.
It works with arbitrary tensor shapes, treating leading dimensions as batch dimensions.

```python
import jax.numpy as jnp
from rotation_trick_jax import rotate_to

# Your vectors (batch_size, dim)
source = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
target = jnp.array([[2.0, 1.0, 4.0], [3.0, 2.0, 1.0]])

# Apply rotation trick
result = rotate_to(source, target)
print(result.shape)  # (2, 3)
```
