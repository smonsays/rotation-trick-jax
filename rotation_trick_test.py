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

import chex
import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from absl.testing import parameterized

from rotation_trick_jax import rotate_to

jax.config.parse_flags_with_absl()
jax.config.update("jax_numpy_rank_promotion", "raise")


def efficient_rotation_official(
    u: torch.Tensor, q: torch.Tensor, e: torch.Tensor
) -> torch.Tensor:
    """
    Official implementation of the rotation trick.
    See https://arxiv.org/abs/2410.06424
    """
    w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
    e = (
        e
        - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1))
        + 2 * torch.bmm(torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
    )
    return e


def rotate_to_official(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Rotation trick STE (official pytorch implementation).
    See https://arxiv.org/abs/2410.06424
    """
    # Flatten to 2D
    original_shape = src.shape
    src_flat = src.view(-1, src.shape[-1])
    tgt_flat = tgt.view(-1, tgt.shape[-1])

    # Normalize
    src_norm = torch.norm(src_flat, dim=1, keepdim=True) + 1e-6
    tgt_norm = torch.norm(tgt_flat, dim=1, keepdim=True) + 1e-6

    u = src_flat / src_norm
    q = tgt_flat / tgt_norm
    e = src_flat.unsqueeze(1)

    # Apply rotation
    rotated = efficient_rotation_official(u, q, e).squeeze()

    # Rescale
    rotated = rotated * (tgt_norm / src_norm).detach()

    # Reshape back
    return rotated.view(original_shape)


class RotationTrickTest(chex.TestCase):
    rng: chex.PRNGKey = jax.random.key(42)

    @chex.variants(with_jit=True, without_jit=True)
    def test_gradient_flow(self) -> None:
        """Test that gradients flow properly through the rotation trick."""
        key1, key2 = jax.random.split(self.rng)

        src = jax.random.normal(key1, (8, 32))
        tgt = jax.random.normal(key2, (8, 32))

        def loss_fn(src_param):
            result = rotate_to(src_param, tgt)
            return jnp.sum(result**2)

        grad_fn = jax.grad(loss_fn)
        grad_result = self.variant(grad_fn)(src)

        # Check that gradients are computed and non-zero
        chex.assert_tree_all_finite(grad_result)
        self.assertTrue(jnp.any(grad_result != 0))
        self.assertGreater(jnp.linalg.norm(grad_result), 0)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(
        dict(batch_shapes=(64,), dim=8),
        dict(batch_shapes=(32, 11, 7), dim=11),
    )
    def test_mathematical_properties(self, batch_shapes: tuple[int], dim: int) -> None:
        """Test mathematical properties of the rotation trick."""
        key1, key2 = jax.random.split(self.rng)
        src = jax.random.normal(key1, (*batch_shapes, dim))
        tgt = jax.random.normal(key2, (*batch_shapes, dim))

        result = self.variant(rotate_to)(src, tgt)

        # Property 1: When src == tgt, result should be close to tgt
        same_vectors = jnp.ones((4, dim))
        result_same = self.variant(rotate_to)(same_vectors, same_vectors)
        chex.assert_trees_all_close(result_same, same_vectors, atol=1e-5)

        # Property 2: Result should have similar magnitude distribution as target
        src_norms = jnp.linalg.norm(src, axis=1)
        tgt_norms = jnp.linalg.norm(tgt, axis=1)
        result_norms = jnp.linalg.norm(result, axis=1)

        src_to_tgt_diff = jnp.mean(jnp.abs(src_norms - tgt_norms))
        result_to_tgt_diff = jnp.mean(jnp.abs(result_norms - tgt_norms))
        self.assertLessEqual(result_to_tgt_diff, src_to_tgt_diff)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(
        dict(batch_shapes=(64,), dim=8),
        dict(batch_shapes=(32, 11, 7), dim=11),
    )
    def test_jax_vs_pytorch(self, batch_shapes: tuple[int], dim: int) -> None:
        """Test that JAX implementation matches PyTorch implementation."""
        torch.manual_seed(0)

        # Create test data
        src_torch = torch.randn(*batch_shapes, dim)
        tgt_torch = torch.randn(*batch_shapes, dim)

        src_jax = jnp.array(src_torch.detach().numpy())
        tgt_jax = jnp.array(tgt_torch.detach().numpy())

        # Test forward pass
        with torch.no_grad():
            result_torch = rotate_to_official(src_torch.clone(), tgt_torch.clone())

        result_jax = self.variant(rotate_to)(src_jax, tgt_jax)

        chex.assert_trees_all_close(
            np.array(result_jax), result_torch.numpy(), atol=1e-5
        )

        # Test backward pass (gradients)
        def jax_loss_fn(src):
            result = rotate_to(src, tgt_jax)
            return jnp.sum(result)

        jax_grad_fn = jax.grad(jax_loss_fn)
        grad_jax = self.variant(jax_grad_fn)(src_jax)

        # PyTorch gradient
        src_torch_grad = torch.randn(*batch_shapes, dim, requires_grad=True)
        src_torch_grad.data = torch.from_numpy(np.array(src_jax))
        result_torch_grad = rotate_to_official(src_torch_grad, tgt_torch)
        loss_torch = result_torch_grad.sum()
        loss_torch.backward()

        chex.assert_trees_all_close(
            np.array(grad_jax), src_torch_grad.grad.numpy(), atol=1e-5
        )


if __name__ == "__main__":
    absltest.main()
