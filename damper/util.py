from check_and_compile import check_and_compile
from jax import grad, lax, numpy as jnp
from jaxtyping import Array, Float


# What's the max we should allow before exponentiation?
BIG: int = 32


######## SEE HERE:
######## <https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where>


@check_and_compile()
def safe_exp(x: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
    x = jnp.where(
        x > BIG,
        x * lax.stop_gradient(BIG / jnp.where(x < BIG, BIG, x)),
        x,
    )
    return jnp.exp(x)
