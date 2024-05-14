from damper.util import safe_exp

from beartype.typing import Tuple
from check_and_compile import check_and_compile
from jax import debug, lax, numpy as jnp, tree
from jaxtyping import Array, Float32, Float64, PyTree


@check_and_compile()
def standard_deviation_loss(
    standard_deviations: Float64[Array, "*batch"],
    xs: Float64[Array, "*batch"],
    epsilon: Float32[Array, ""],
) -> Float64[Array, ""]:
    """
    Loss that, over time, precisely computes standard deviation.
    Assumes a mean of zero.
    """
    actual_variances = jnp.square(xs)  # zero-mean
    variances = jnp.square(standard_deviations)
    squared_std_error = jnp.square(actual_variances - variances)
    relative_error = squared_std_error / (epsilon + lax.stop_gradient(variances))
    return jnp.sum(relative_error)


@check_and_compile()
def standard_deviation_grad_exact(
    standard_deviations: Float64[Array, "*batch"],
    xs: Float64[Array, "*batch"],
    epsilon: Float32[Array, ""],
) -> Float64[Array, "*batch"]:
    """
    Directly compute the gradient of the above function in closed form.
    """
    actual_variances = jnp.square(xs)  # zero-mean
    variances = jnp.square(standard_deviations)

    # # Loss w.r.t. variance is easy:
    # dLdv = 2 * (variances - actual_variances) / (epsilon + lax.stop_gradient(variances))
    # return 2 * standard_deviations * dLdv

    return (
        # 4 *
        standard_deviations
        * (variances - actual_variances)
        / (epsilon + lax.stop_gradient(variances))
    )


@check_and_compile()
def standard_deviation_grad_approx(
    standard_deviations: Float64[Array, "*batch"],
    xs: Float64[Array, "*batch"],
    epsilon: Float32[Array, ""],
) -> Float64[Array, "*batch"]:
    """
    Directly compute the gradient of the above function in closed form.
    """
    actual_variances = jnp.square(xs)  # zero-mean
    variances = jnp.square(standard_deviations)

    # # Loss w.r.t. variance is easy:
    # dLdv = 2 * (variances - actual_variances) / (epsilon + lax.stop_gradient(variances))
    # return 2 * standard_deviations * dLdv

    return (variances - actual_variances) / (
        epsilon + jnp.abs(lax.stop_gradient(standard_deviations))
    )


@check_and_compile()
def normalize(
    xs: Float64[Array, "*batch"],
    standard_deviations: Float64[Array, "*batch"],
    epsilon: Float32[Array, ""],
) -> Float64[Array, "*batch"]:
    return xs / (epsilon + standard_deviations)


@check_and_compile()
def normalized_dot_product(
    a: Float64[Array, "*batch"],
    b: Float64[Array, "*batch"],
    standard_deviations: Float64[Array, "*batch"],
    epsilon: Float32[Array, ""],
) -> Float64[Array, "*batch"]:
    # TODO: variance instead of dividing by std twice
    a = normalize(a, standard_deviations, epsilon)
    b = normalize(b, standard_deviations, epsilon)
    return a * b  # no sum: treat each entry independently


@check_and_compile()
def update_tensor(
    parameters: Float64[Array, "*batch"],
    current_grad: Float64[Array, "*batch"],
    previous_grad: Float64[Array, "*batch"],
    lr: Float64[Array, "*batch"],
    stds: Float64[Array, "*batch"],
    sensitivity: Float32[Array, ""],
    std_update: Float32[Array, ""],
    epsilon: Float32[Array, ""],
) -> Tuple[
    Float64[Array, "*batch"],
    Float64[Array, "*batch"],
    Float64[Array, "*batch"],
]:
    # If the dot product of this gradient and the last is negative,
    # that means we're oscillating, and the more negative, the worse:
    dot_prod = normalized_dot_product(
        current_grad,
        previous_grad,
        stds,
        epsilon,
    )

    # Compute standard deviation error (but don't update it yet):
    # dLds = grad(standard_deviation_loss)(stds, current_grad, epsilon)
    dLds = standard_deviation_grad_approx(stds, current_grad, epsilon)

    # # Toss out outliers:
    # normalized = current_grad / (epsilon + stds)
    # sigma = jnp.abs(normalized)
    # current_grad = jnp.where(sigma < 3, current_grad, 0)

    # Update standard deviation (now that we've used the previous one for outliers):
    stds = stds - std_update * dLds

    # Adjust learning rate to keep dot product approximately `ideal_covariance`:
    lr = lr * safe_exp(sensitivity * dot_prod)
    # NOTE: Why exponentiate above?
    # If we have a normal distribution of dot products (as we'd expect),
    # then we want the learning rate to stay constant.
    # The only way you can multiply by some function of +1 then -1
    # and have the result not change is to exponentiate, since e.g. (e^1 e^-1) = 1:
    # addition to zero in exponential space cancels out.

    # Just in case:
    lr = jnp.maximum(lr, epsilon)

    # Update parameters and return:
    return parameters - lr * current_grad, lr, stds


@check_and_compile()
def update(
    parameters: PyTree[Float64[Array, "..."]],
    current_grad: PyTree[Float64[Array, "..."]],
    previous_grad: PyTree[Float64[Array, "..."]],
    lr: PyTree[Float64[Array, "..."]],
    stds: PyTree[Float64[Array, "..."]],
    sensitivity: Float32[Array, ""],
    std_update: Float32[Array, ""],
    epsilon: Float32[Array, ""],
) -> Tuple[
    PyTree[Float64[Array, "..."]],
    PyTree[Float64[Array, "..."]],
    PyTree[Float64[Array, "..."]],
]:
    return tree.transpose(
        outer_treedef=tree.structure(parameters),
        inner_treedef=tree.structure(("*", "*", "*")),
        pytree_to_transpose=tree.map(
            lambda *args: update_tensor(
                *args,
                sensitivity,
                std_update,
                epsilon,
            ),
            parameters,
            current_grad,
            previous_grad,
            lr,
            stds,
        ),
    )
