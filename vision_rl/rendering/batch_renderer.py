"""Batch camera rendering for MJX rollouts.

Two interchangeable backends behind one interface:

* ``MadronaBatchRenderer`` -- wraps ``madrona_mjx.BatchRenderer`` and renders
  every parallel world on-GPU, keeping rollouts fully jitted. Fast, harder to
  install.
* ``CPUBatchRenderer`` -- renders each world with the standard ``mujoco.Renderer``
  on the host via ``jax.pure_callback``. Trivial to run anywhere, but slow and
  it breaks the pure-GPU pipeline. Good for debugging / tiny runs.

Both expose the same contract::

    token, rgb = renderer.init(data)      # data: batched mjx.Data [B, ...]
    token, rgb = renderer.render(token, data)

``rgb`` is uint8 with shape ``[B, H, W, 3]`` (single camera). ``token`` is an
opaque value that must be threaded back into ``render`` (Madrona needs it; the
CPU backend ignores it).
"""

from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from vision_rl.config import RenderConfig


class BatchRendererBase:
    backend: str

    def init(self, data) -> Tuple[Any, jax.Array]:
        raise NotImplementedError

    def render(self, token: Any, data) -> Tuple[Any, jax.Array]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Madrona (GPU batch) backend
# --------------------------------------------------------------------------- #
class MadronaBatchRenderer(BatchRendererBase):
    backend = "madrona"

    def __init__(self, mj_model, mjx_model, cfg: RenderConfig, num_envs: int):
        from madrona_mjx.renderer import BatchRenderer  # lazy import

        import mujoco

        cam_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.camera
        )
        if cam_id < 0:
            raise ValueError(f"camera '{cfg.camera}' not found in model")

        self._mjx_model = mjx_model
        self._renderer = BatchRenderer(
            m=mjx_model,
            gpu_id=cfg.gpu_id,
            num_worlds=num_envs,
            batch_render_view_width=cfg.width,
            batch_render_view_height=cfg.height,
            enabled_geom_groups=np.array(cfg.enabled_geom_groups, dtype=np.int32),
            enabled_cameras=np.array([cam_id], dtype=np.int32),
            add_cam_debug_geo=False,
            use_rasterizer=cfg.use_rasterizer,
            viz_gpu_hdls=None,
        )

    @staticmethod
    def _to_uint8_rgb(rgb: jax.Array) -> jax.Array:
        # Madrona returns [B, num_cams, H, W, 4]; take cam 0, drop alpha.
        rgb = rgb[:, 0, :, :, :3]
        return rgb.astype(jnp.uint8)

    def init(self, data):
        # Some madrona_mjx versions take init(data), others init(data, model);
        # pass the model to match the mujoco_playground usage.
        token, rgb, _depth = self._renderer.init(data, self._mjx_model)
        return token, self._to_uint8_rgb(rgb)

    def render(self, token, data):
        token, rgb, _depth = self._renderer.render(token, data)
        return token, self._to_uint8_rgb(rgb)


# --------------------------------------------------------------------------- #
# CPU (host callback) backend
# --------------------------------------------------------------------------- #
class CPUBatchRenderer(BatchRendererBase):
    backend = "cpu"

    def __init__(self, mj_model, mjx_model, cfg: RenderConfig, num_envs: int):
        import mujoco

        self._mj_model = mj_model
        self._cfg = cfg
        self._num_envs = num_envs
        self._cam_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.camera
        )
        if self._cam_id < 0:
            raise ValueError(f"camera '{cfg.camera}' not found in model")

        # Persistent host-side data + renderer reused across callbacks.
        self._mj_data = mujoco.MjData(mj_model)
        self._renderer = mujoco.Renderer(mj_model, height=cfg.height, width=cfg.width)
        self._nmocap = mj_model.nmocap

    def _host_render(self, qpos, qvel, mocap_pos, mocap_quat) -> np.ndarray:
        import mujoco

        b = qpos.shape[0]
        out = np.empty((b, self._cfg.height, self._cfg.width, 3), dtype=np.uint8)
        d = self._mj_data
        for i in range(b):
            d.qpos[:] = qpos[i]
            d.qvel[:] = qvel[i]
            if self._nmocap:
                d.mocap_pos[:] = mocap_pos[i]
                d.mocap_quat[:] = mocap_quat[i]
            mujoco.mj_forward(self._mj_model, d)
            self._renderer.update_scene(d, camera=self._cam_id)
            out[i] = self._renderer.render()
        return out

    def _render(self, data) -> jax.Array:
        result_shape = jax.ShapeDtypeStruct(
            (self._num_envs, self._cfg.height, self._cfg.width, 3), jnp.uint8
        )
        rgb = jax.pure_callback(
            self._host_render,
            result_shape,
            data.qpos,
            data.qvel,
            data.mocap_pos,
            data.mocap_quat,
        )
        return rgb

    def init(self, data):
        return None, self._render(data)

    def render(self, token, data):
        return token, self._render(data)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_renderer(
    mj_model, mjx_model, cfg: RenderConfig, num_envs: int
) -> BatchRendererBase:
    """Build a renderer, honouring cfg.backend ("madrona" | "cpu" | "auto")."""
    backend = cfg.backend
    if backend == "auto":
        try:
            import madrona_mjx  # noqa: F401

            backend = "madrona"
        except Exception:
            backend = "cpu"

    if backend == "madrona":
        return MadronaBatchRenderer(mj_model, mjx_model, cfg, num_envs)
    if backend == "cpu":
        return CPUBatchRenderer(mj_model, mjx_model, cfg, num_envs)
    raise ValueError(f"unknown render backend: {backend}")
