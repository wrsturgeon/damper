from damper import damper

from hypothesis import given
from hypothesis.extra import numpy as hnp
from jax import grad, numpy as jnp, random as jrnd
import numpy as np


key = jrnd.PRNGKey(42)
BATCHES = [1, 2, 3]


def prop_rms_loss_gradient_approx(rms, xs, eps):
    f = damper.rms_grad_approx
    g = grad(damper.rms_loss)
    args = rms, xs, eps
    args = tuple([jnp.array(arg) for arg in args])
    y = g(*args)
    z = 4 * f(*args)
    roundoff = jnp.maximum(jnp.maximum(jnp.square(rms), jnp.square(xs)) * 1e-5, 1e-8)
    if jnp.all(jnp.isfinite(y)) and jnp.all(jnp.isfinite(z)):
        assert jnp.all(
            jnp.abs(y - jnp.sign(rms) * z) < roundoff
        ), f"\n\n{y}\n=/=\n{z}\n"


def test_rms_loss_gradient_approx_1():
    prop_rms_loss_gradient_approx(
        jnp.zeros(BATCHES, dtype=jnp.float64),
        jnp.zeros(BATCHES, dtype=jnp.float64),
        jnp.zeros([], dtype=jnp.float32),
    )


def test_rms_loss_gradient_approx_2():
    prop_rms_loss_gradient_approx(
        jnp.ones(BATCHES, dtype=jnp.float64),
        jnp.zeros(BATCHES, dtype=jnp.float64),
        jnp.zeros([], dtype=jnp.float32),
    )


@given(
    hnp.arrays(np.float64, BATCHES),  # type: ignore[call-overload]
    hnp.arrays(np.float64, BATCHES),  # type: ignore[call-overload]
)
def test_rms_loss_gradient_approx_prop(rms, xs):
    prop_rms_loss_gradient_approx(
        rms,
        xs,
        jnp.zeros([], dtype=jnp.float32),
    )


def test_run_simple():
    p = jrnd.normal(key, BATCHES)
    loss = lambda x: jnp.sum(jnp.square(x))
    init_loss = loss(p)

    lr = jnp.full_like(p, 0.01)
    rms = jnp.ones_like(p)
    ideal_dot_prod = jnp.array(0.0, dtype=jnp.float32)
    sensitivity = jnp.array(0.01, dtype=jnp.float32)
    rms_update = jnp.array(0.001, dtype=jnp.float32)
    weight_decay = jnp.array(0.999, dtype=jnp.float32)
    epsilon = jnp.array(1e-5, dtype=jnp.float32)

    dLd = grad(loss)

    dLdp = dLd(p)
    p, last_dLdp, lr, rms = damper.update(
        p,
        dLdp,
        jnp.zeros_like(p),
        lr,
        rms,
        ideal_dot_prod,
        sensitivity,
        rms_update,
        weight_decay,
        epsilon,
    )
    assert loss(p) < init_loss

    for _ in range(100):
        dLdp = dLd(p)
        p, last_dLdp, lr, rms = damper.update(
            p,
            dLdp,
            last_dLdp,
            lr,
            rms,
            ideal_dot_prod,
            sensitivity,
            rms_update,
            weight_decay,
            epsilon,
        )

    assert loss(p) < init_loss
