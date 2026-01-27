import jax

jax.distributed.initialize()
print(jax.process_index(), jax.devices())

from .optimize_partial import main

if __name__ == "__main__":
    main()
