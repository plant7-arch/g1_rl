"""Quaternion helpers (MuJoCo convention: w, x, y, z)."""

import jax.numpy as jnp


def quat_mul(a, b):
  aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
  bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  return jnp.stack([
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
  ], axis=-1)


def quat_rotate(q, v):
  """Rotate vector v from the body frame to the world frame."""
  w, xyz = q[..., :1], q[..., 1:]
  t = 2.0 * jnp.cross(xyz, v)
  return v + w * t + jnp.cross(xyz, t)


def quat_rotate_inverse(q, v):
  """Rotate vector v from the world frame into the body frame."""
  w, xyz = q[..., :1], q[..., 1:]
  t = 2.0 * jnp.cross(xyz, v)
  return v - w * t + jnp.cross(xyz, t)


def yaw_quat(yaw):
  return jnp.stack([jnp.cos(yaw / 2), 0.0 * yaw, 0.0 * yaw, jnp.sin(yaw / 2)], axis=-1)


def quat_to_yaw(q):
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap_to_pi(a):
  return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi
