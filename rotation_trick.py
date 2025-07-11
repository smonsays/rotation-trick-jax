"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import einops
import jax
import jax.numpy as jnp
import jaxtyping as jt


def rotation_trick_transform(
  source: jt.Float[jt.Array, 'batch dim'],
  target: jt.Float[jt.Array, 'batch dim'],
  eps: float = 1e-8,
) -> jt.Float[jt.Array, 'batch dim']:
  """
  Rotation trick transform from https://arxiv.org/abs/2410.06424.

  Notation from section 4.2 of the paper:
  R(e) = λ [e - 2 (e·r) r + 2 (e·e_hat) q_hat]
  with rotation axis r = (e_hat + q_hat) / ||(e_hat + q_hat)||.
  """
  e = source

  # Normalize and stop gradients for e_hat and q_hat
  e_norm = jnp.linalg.norm(source, axis=1, keepdims=True)
  q_norm = jnp.linalg.norm(target, axis=1, keepdims=True)
  e_hat = jax.lax.stop_gradient(source / (e_norm + eps))
  q_hat = jax.lax.stop_gradient(target / (q_norm + eps))

  # Normalized rotation axis
  r = e_hat + q_hat
  r = r / (jnp.linalg.norm(r, axis=1, keepdims=True) + eps)

  # Rotate: e - 2 (e·r) r + 2 (e·e_hat) q_hat
  term2 = 2 * jnp.einsum('bi,bi,bj->bj', e, r, r)
  term3 = 2 * jnp.einsum('bi,bi,bj->bj', e, e_hat, q_hat)
  rotated = e - term2 + term3

  # Rescale (denoted λ in paper)
  rescale_factor = jax.lax.stop_gradient(q_norm / (e_norm + eps))

  return rotated * rescale_factor


def rotate_to(source: jax.Array, target: jax.Array) -> jax.Array:
  """
  Rotation trick from https://arxiv.org/abs/2410.06424
  """
  original_shape = source.shape  # flatten to single batch dimension
  source_flat = einops.rearrange(source, '... d -> (...) d')
  target_flat = einops.rearrange(target, '... d -> (...) d')
  rotated = rotation_trick_transform(source_flat, target_flat)

  return jnp.reshape(rotated, original_shape)
