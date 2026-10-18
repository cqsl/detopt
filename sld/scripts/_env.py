
import jax


def set_xla_flags():
    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

    flag = "--xla_gpu_dot_merger_threshold_mb"
    existing = os.environ.get("XLA_FLAGS", "")
    if flag in existing:
        return
    os.environ["XLA_FLAGS"] = f"{existing} {flag}=0".strip()


def enable_x64():
    jax.config.update("jax_enable_x64", True)


def setup_mesh():
    from jax.sharding import AxisType, Mesh, set_mesh

    set_mesh(Mesh(jax.devices(), "i", AxisType.Explicit))


def setup_jax():
    set_xla_flags()
    enable_x64()
    setup_mesh()
