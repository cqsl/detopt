import jax
import jax.numpy as jnp
import numpy as np

import jax.experimental.multihost_utils


def save_h5(fname, Xs, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data={}, postprocess_data={}, cart=False):
    import h5py

    if isinstance(Xs, tuple):
        Xs = np.array(Xs[0]), np.array(Xs[1])
    else:
        Xs = np.array(Xs)
    mo_coeff = np.array(mo_coeff)
    eigvals = np.array(eigvals)

    extra_data = {k: np.array(v) for k, v in extra_data.items()}
    postprocess_data = {k: np.array(v) for k, v in postprocess_data.items()}

    if jax.process_index() == 0:
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
            grp = f.create_group("extra")
            for k, v in extra_data.items():
                grp[str(k)] = np.array(v)
            grp = f.create_group("postprocess")
            for k, v in postprocess_data.items():
                grp[str(k)] = np.array(v)


def load_h5(fname, jax_arrays=True):
    import h5py

    with h5py.File(fname, "r") as f:
        if "Xs" in f.keys():  # GHF
            Xs = f["Xs"]
        else:  # GHF
            Xs = f["Xsu"], f["Xsd"]

        if "mo_coeff" in f.keys():
            mo_coeff = f["mo_coeff"]
        else:
            mo_coeff = None
        if jax_arrays:
            Xs = jax.tree.map(jnp.asarray, Xs)
            if mo_coeff is not None:
                mo_coeff = jnp.asarray(mo_coeff)
        else:
            Xs = jax.tree.map(np.array, Xs)
            if mo_coeff is not None:
                mo_coeff = np.array(mo_coeff)

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
        postprocess_data = {}
        for k in f["postprocess"].keys():
            postprocess_data[str(k)] = np.array(f["postprocess"][k])
        extra_data = {}
        for k in f["extra"].keys():
            extra_data[str(k)] = np.array(f["extra"][k])
        # last None element reserved for future use
        return Xs, mo_coeff, eigvals, xyzstr, basis, spin, name, extra_data, postprocess_data, cart, None
