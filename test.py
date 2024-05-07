from damper import damper

from hypothesis import given
from hypothesis.extra import numpy as hnp
from jax import grad, numpy as jnp, random as jrnd
import numpy as np


key = jrnd.PRNGKey(42)
BATCHES = [1, 2, 3]


def prop_standard_deviation_loss_gradient_exact(stds, xs, eps):
    f = damper.standard_deviation_grad_exact
    g = grad(damper.standard_deviation_loss)
    args = stds, xs, eps
    args = tuple([jnp.array(arg) for arg in args])
    y = g(*args)
    z = 4 * f(*args)
    roundoff = jnp.maximum(jnp.maximum(jnp.square(stds), jnp.square(xs)) * 1e-5, 1e-8)
    if jnp.all(jnp.isfinite(y)) and jnp.all(jnp.isfinite(z)):
        assert jnp.all(
            jnp.abs(y - jnp.sign(stds) * z) < roundoff
        ), f"\n\n{y}\n=/=\n{z}\n"


def test_standard_deviation_loss_gradient_exact_1():
    prop_standard_deviation_loss_gradient_exact(
        jnp.zeros(BATCHES, dtype=jnp.float64),
        jnp.zeros(BATCHES, dtype=jnp.float64),
        jnp.zeros([], dtype=jnp.float32),
    )


def test_standard_deviation_loss_gradient_exact_2():
    prop_standard_deviation_loss_gradient_exact(
        jnp.ones(BATCHES, dtype=jnp.float64),
        jnp.zeros(BATCHES, dtype=jnp.float64),
        jnp.zeros([], dtype=jnp.float32),
    )


@given(
    hnp.arrays(np.float64, BATCHES),  # type: ignore[call-overload]
    hnp.arrays(np.float64, BATCHES),  # type: ignore[call-overload]
    hnp.arrays(np.float32, []),  # type: ignore[call-overload]
)
def test_standard_deviation_loss_gradient_exact_prop(stds, xs, eps):
    prop_standard_deviation_loss_gradient_exact(stds, xs, eps)


def prop_standard_deviation_loss_gradient_approx(stds, xs, eps):
    f = damper.standard_deviation_grad_approx
    g = grad(damper.standard_deviation_loss)
    args = stds, xs, eps
    args = tuple([jnp.array(arg) for arg in args])
    y = g(*args)
    z = 4 * f(*args)
    roundoff = jnp.maximum(jnp.maximum(jnp.square(stds), jnp.square(xs)) * 1e-5, 1e-8)
    if jnp.all(jnp.isfinite(y)) and jnp.all(jnp.isfinite(z)):
        assert jnp.all(
            jnp.abs(y - jnp.sign(stds) * z) < roundoff
        ), f"\n\n{y}\n=/=\n{z}\n"


def test_standard_deviation_loss_gradient_approx_1():
    prop_standard_deviation_loss_gradient_approx(
        jnp.zeros(BATCHES, dtype=jnp.float64),
        jnp.zeros(BATCHES, dtype=jnp.float64),
        jnp.zeros([], dtype=jnp.float32),
    )


def test_standard_deviation_loss_gradient_approx_2():
    prop_standard_deviation_loss_gradient_approx(
        jnp.ones(BATCHES, dtype=jnp.float64),
        jnp.zeros(BATCHES, dtype=jnp.float64),
        jnp.zeros([], dtype=jnp.float32),
    )


@given(
    hnp.arrays(np.float64, BATCHES),  # type: ignore[call-overload]
    hnp.arrays(np.float64, BATCHES),  # type: ignore[call-overload]
)
def test_standard_deviation_loss_gradient_approx_prop(stds, xs):
    prop_standard_deviation_loss_gradient_approx(
        stds,
        xs,
        jnp.zeros([], dtype=jnp.float32),
    )


def test_run_simple():
    p = jrnd.normal(key, BATCHES)
    loss = lambda x: jnp.sum(jnp.square(x))
    init_loss = loss(p)

    lr = jnp.full_like(p, 0.01)
    stds = jnp.ones_like(p)
    sensitivity = jnp.array(0.05, dtype=jnp.float32)
    epsilon = jnp.array(1e-8, dtype=jnp.float32)

    dLd = grad(loss)

    dLdp = dLd(p)
    p, lr, stds = damper.update(
        p,
        dLdp,
        jnp.zeros_like(p),
        lr,
        stds,
        sensitivity,
        epsilon,
    )
    assert loss(p) < init_loss

    last_dLdp = dLdp
    dLdp = dLd(p)
    p, lr, stds = damper.update(
        p,
        dLdp,
        last_dLdp,
        lr,
        stds,
        sensitivity,
        epsilon,
    )
    assert loss(p) < init_loss
