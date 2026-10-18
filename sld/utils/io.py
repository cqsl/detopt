import sys

import numpy as np


def record_settings(data, args, settings):
    # the given args, overwriting those of an earlier run (e.g. from --start)
    for k in settings:
        v = getattr(args, k, None)
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v


def _to_dataset(v):
    # h5py has no dtype for numpy's fixed-width unicode, so strings are stored as bytes
    if isinstance(v, str):
        return np.bytes_(v)
    v = np.array(v)
    return np.char.encode(v, "utf-8") if v.dtype.kind == "U" else v


def to_str(v):
    # inverse of _to_dataset for strings and lists of strings read back with load_h5
    v = np.asarray(v)
    decode = lambda x: x.decode() if isinstance(x, bytes) else str(x)
    return decode(v.item()) if v.ndim == 0 else [decode(x) for x in v.tolist()]


def save_h5(fname, Xs, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data=None, postprocess_data=None, cart=False, nfreeze=None):
    import h5py

    if isinstance(Xs, tuple):
        Xs = np.array(Xs[0]), np.array(Xs[1])
    else:
        Xs = np.array(Xs)
    mo_coeff = np.array(mo_coeff)
    eigvals = np.array(eigvals)

    extra_data = {k: _to_dataset(v) for k, v in (extra_data or {}).items()}
    postprocess_data = {k: _to_dataset(v) for k, v in (postprocess_data or {}).items()}

    # only the first process writes; without jax loaded there is a single process
    jax = sys.modules.get("jax")
    if jax is None or jax.process_index() == 0:
        with h5py.File(fname, "w") as f:
            if isinstance(Xs, tuple):  # UHF
                f["Xsu"], f["Xsd"] = np.array(Xs[0]), np.array(Xs[1])
            else:  # GHF
                f["Xs"] = np.array(Xs)
            if mo_coeff is not None:
                f["mo_coeff"] = np.array(mo_coeff)
            f["eigvals"] = np.array(eigvals)
            if xyzstr is not None:
                f.attrs["xyz"] = np.bytes_(str(xyzstr))
            if basis is not None:
                f.attrs["basis"] = np.bytes_(str(basis))
            if spin is not None:
                f.attrs["spin"] = int(spin)
            f.attrs["cart"] = bool(cart)
            f.attrs["name"] = np.bytes_(str(name))
            if nfreeze is not None:
                f.attrs["nfreeze"] = int(nfreeze)
            grp = f.create_group("extra")
            for k, v in extra_data.items():
                grp[str(k)] = v
            grp = f.create_group("postprocess")
            for k, v in postprocess_data.items():
                grp[str(k)] = v


def load_h5(fname, jax_arrays=True):
    import h5py

    with h5py.File(fname, "r") as f:
        if "Xs" in f.keys():  # GHF
            Xs = f["Xs"]
        else:  # UHF
            Xs = f["Xsu"], f["Xsd"]

        if "mo_coeff" in f.keys():
            mo_coeff = f["mo_coeff"]
        else:
            mo_coeff = None
        # numpy unless jax arrays are asked for, so reading a file does not need jax
        asarray = np.array
        if jax_arrays:
            import jax.numpy as jnp
            asarray = jnp.asarray
        Xs = tuple(map(asarray, Xs)) if isinstance(Xs, tuple) else asarray(Xs)
        if mo_coeff is not None:
            mo_coeff = asarray(mo_coeff)

        eigvals = np.array(f["eigvals"])
        if "xyz" in f.attrs.keys():
            xyzstr = str(f.attrs["xyz"].decode())
        else:
            xyzstr = None
        if "basis" in f.attrs.keys():
            basis = str(f.attrs["basis"].decode())
        else:
            basis = None
        name = str(f.attrs["name"].decode())
        if "spin" in f.attrs.keys():
            spin = int(f.attrs["spin"])
        else:
            spin = None
        if "cart" in f.attrs.keys():
            cart = bool(f.attrs["cart"])
        else:
            cart = False
        if "nfreeze" in f.attrs.keys():
            nfreeze = int(f.attrs["nfreeze"])
        else:
            nfreeze = None
        postprocess_data = {}
        for k in f["postprocess"].keys():
            postprocess_data[str(k)] = np.array(f["postprocess"][k])
        extra_data = {}
        for k in f["extra"].keys():
            extra_data[str(k)] = np.array(f["extra"][k])
        return Xs, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data, postprocess_data, cart, nfreeze
