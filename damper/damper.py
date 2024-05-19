from damper.util import safe_exp

from beartype.typing import Tuple
from check_and_compile import check_and_compile
from jax import debug, lax, numpy as jnp, tree
from jaxtyping import Array, Float, PyTree


@check_and_compile()
def rms_loss(
    rms: Float[Array, "*batch"],
    xs: Float[Array, "*batch"],
    epsilon: Float[Array, ""],
) -> Float[Array, ""]:
    """
    Loss that, over time, precisely computes root-mean-squared.
    Assumes a mean of zero.
    """
    actual_mean_squared = jnp.square(xs)  # zero-mean
    mean_squared = jnp.square(rms)
    squared_rms_error = jnp.square(actual_mean_squared - mean_squared)
    relative_error = squared_rms_error / (epsilon + lax.stop_gradient(mean_squared))
    return jnp.sum(relative_error)


@check_and_compile()
def rms_grad_exact(
    rms: Float[Array, "*batch"],
    xs: Float[Array, "*batch"],
    epsilon: Float[Array, ""],
) -> Float[Array, "*batch"]:
    """
    Directly compute the gradient of the above function in closed form.
    """
    actual_mean_squared = jnp.square(xs)  # zero-mean
    mean_squared = jnp.square(rms)

    # # Loss w.r.t. mean_squared is easy:
    # dLdv = 2 * (mean_squared - actual_mean_squared) / (epsilon + lax.stop_gradient(mean_squared))
    # return 2 * rms * dLdv

    return (
        # 4 *
        rms
        * (mean_squared - actual_mean_squared)
        / (epsilon + lax.stop_gradient(mean_squared))
    )


@check_and_compile()
def rms_grad_approx(
    rms: Float[Array, "*batch"],
    xs: Float[Array, "*batch"],
    epsilon: Float[Array, ""],
) -> Float[Array, "*batch"]:
    """
    Directly compute the gradient of the above function in closed form.
    """
    actual_mean_squared = jnp.square(xs)  # zero-mean
    mean_squared = jnp.square(rms)

    # # Loss w.r.t. mean_squared is easy:
    # dLdv = 2 * (mean_squared - actual_mean_squared) / (epsilon + lax.stop_gradient(mean_squared))
    # return 2 * rms * dLdv

    return (mean_squared - actual_mean_squared) / (
        epsilon + jnp.abs(lax.stop_gradient(rms))
    )


@check_and_compile()
def normalize(
    xs: Float[Array, "*batch"],
    rms: Float[Array, "*batch"],
    epsilon: Float[Array, ""],
) -> Float[Array, "*batch"]:
    return xs / (epsilon + rms)


@check_and_compile()
def update_tensor(
    parameters: Float[Array, "*batch"],
    current_grad: Float[Array, "*batch"],
    previous_grad: Float[Array, "*batch"],
    lr: Float[Array, "*batch"],
    rms: Float[Array, "*batch"],
    ideal_dot_prod: Float[Array, ""],
    sensitivity: Float[Array, ""],
    rms_update: Float[Array, ""],
    weight_decay: Float[Array, ""],
    epsilon: Float[Array, ""],
) -> Tuple[
    Float[Array, "*batch"],
    Float[Array, "*batch"],
    Float[Array, "*batch"],
    Float[Array, "*batch"],
]:
    # Compute root-mean-squared error (but don't update it yet):
    # dLds = grad(rms_loss)(rms, current_grad, epsilon)
    dLds = rms_grad_approx(rms, current_grad, epsilon)

    # Normalize gradients by their root-mean-squared:
    current_grad = normalize(current_grad, rms, epsilon)
    # PREVIOUS GRADIENT IS ALREADY NORMALIZED: DO NOT DIVIDE TWICE

    # Update root-mean-squared (now that we've used the previous one for outliers):
    rms = rms - rms_update * dLds

    # If the dot product of this gradient and the last is negative,
    # that means we're oscillating, and the more negative, the worse:
    dot_prod = current_grad * previous_grad  # already normalized

    # Adjust learning rate to keep dot product approximately `ideal_dot_prod`:
    dot_prod = jnp.minimum(1, dot_prod)
    error = dot_prod - ideal_dot_prod
    exponent = sensitivity * error
    growth = safe_exp(exponent)
    lr = lr * growth
    # NOTE: Why exponentiate above?
    # If we have a normal distribution of dot products (as we'd expect),
    # then we want the learning rate to stay constant.
    # The only way you can multiply by some function of +1 then -1
    # and have the result not change is to exponentiate, since e.g. (e^1 e^-1) = 1:
    # addition to zero in exponential space cancels out.

    # Just in case:
    lr = jnp.maximum(lr, epsilon)

    # Update parameters and return:
    return weight_decay * parameters - lr * current_grad, current_grad, lr, rms


@check_and_compile()
def update(
    parameters: PyTree[Float[Array, "..."]],
    current_grad: PyTree[Float[Array, "..."]],
    previous_grad: PyTree[Float[Array, "..."]],
    lr: PyTree[Float[Array, "..."]],
    rms: PyTree[Float[Array, "..."]],
    ideal_dot_prod: Float[Array, ""],
    sensitivity: Float[Array, ""],
    rms_update: Float[Array, ""],
    weight_decay: Float[Array, ""],
    epsilon: Float[Array, ""],
) -> Tuple[
    PyTree[Float[Array, "..."]],
    PyTree[Float[Array, "..."]],
    PyTree[Float[Array, "..."]],
    PyTree[Float[Array, "..."]],
]:
    return tree.transpose(
        outer_treedef=tree.structure(parameters),
        inner_treedef=tree.structure(("*", "*", "*", "*")),
        pytree_to_transpose=tree.map(
            lambda *args: update_tensor(
                *args,
                ideal_dot_prod,
                sensitivity,
                rms_update,
                weight_decay,
                epsilon,
            ),
            parameters,
            current_grad,
            previous_grad,
            lr,
            rms,
        ),
    )
