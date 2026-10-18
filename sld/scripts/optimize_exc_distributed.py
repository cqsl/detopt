import jax

jax.distributed.initialize()
print(jax.process_index(), jax.devices())

from .optimize_exc import main

if __name__ == "__main__":
    main()
