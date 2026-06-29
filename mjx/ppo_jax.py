import jax
import jax.numpy as jp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from functools import partial

GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
ENT_COEF = 0.02
MAX_GRAD = 0.5
LOGSTD_MIN = -2.0


class Actor(nn.Module):
    @nn.compact
    def __call__(self, x):
        log_std = self.param("log_std", lambda k: jp.full((13,), -0.5))
        for _ in range(4):
            x = nn.relu(nn.Dense(256)(x))
        # zero-init last layer so the initial policy outputs near zero (home pose delta)
        mean = nn.tanh(nn.Dense(13, kernel_init=nn.initializers.zeros,
                                bias_init=nn.initializers.zeros)(x))
        return mean, log_std


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        for _ in range(4):
            x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(64)(x))
        return nn.Dense(1)(x).squeeze(-1)


def _logprob(action, mean, std):
    return jp.sum(-0.5 * ((action - mean) / std) ** 2 - jp.log(std) - 0.5 * jp.log(2 * jp.pi), axis=-1)


def _entropy(std):
    return jp.sum(jp.log(std) + 0.5 * jp.log(2 * jp.pi * jp.e))


def init_states(rng, obs_dim, lr_schedule):
    ka, kc = jax.random.split(rng)
    sample = jp.zeros((obs_dim,))
    tx = lambda: optax.chain(optax.clip_by_global_norm(MAX_GRAD), optax.adam(lr_schedule))
    actor = TrainState.create(apply_fn=Actor().apply, params=Actor().init(ka, sample), tx=tx())
    critic = TrainState.create(apply_fn=Critic().apply, params=Critic().init(kc, sample), tx=tx())
    return actor, critic


def act(actor, obs, key):
    """Sample an action and its log prob. obs is already normalized."""
    mean, log_std = actor.apply_fn(actor.params, obs)
    std = jp.exp(jp.clip(log_std, min=LOGSTD_MIN))
    action = mean + std * jax.random.normal(key, mean.shape)
    return action, _logprob(action, mean, std)


def value(critic, obs):
    return critic.apply_fn(critic.params, obs)


def compute_gae(rewards, values, dones):
    """GAE over (T, N) arrays; bootstraps 0 past the rollout, matching the torch version."""
    def step(carry, x):
        gae, next_value = carry
        reward, val, done = x
        nonterminal = 1.0 - done
        delta = reward + GAMMA * next_value * nonterminal - val
        gae = delta + GAMMA * LAM * nonterminal * gae
        return (gae, val), gae
    init = (jp.zeros(rewards.shape[1]), jp.zeros(rewards.shape[1]))
    _, adv = jax.lax.scan(step, init, (rewards, values, dones), reverse=True)
    return adv, adv + values


def _actor_loss(params, apply_fn, obs, actions, old_lp, adv):
    mean, log_std = apply_fn(params, obs)
    std = jp.exp(jp.clip(log_std, min=LOGSTD_MIN))
    new_lp = _logprob(actions, mean, std)
    ratio = jp.exp(new_lp - old_lp)
    clipped = jp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv
    return -jp.mean(jp.minimum(ratio * adv, clipped)) - ENT_COEF * _entropy(std)


def _critic_loss(params, apply_fn, obs, returns):
    return jp.mean((apply_fn(params, obs) - returns) ** 2)


@partial(jax.jit, static_argnames=("epochs", "n_minibatch"))
def update(actor, critic, obs, actions, old_lp, adv, returns, key, epochs, n_minibatch):
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    B = obs.shape[0]
    mb = B // n_minibatch

    def epoch_body(carry, ekey):
        actor, critic = carry
        perm = jax.random.permutation(ekey, B)

        def mb_body(carry, m):
            actor, critic = carry
            idx = jax.lax.dynamic_slice_in_dim(perm, m * mb, mb)
            o = obs[idx]
            al, ag = jax.value_and_grad(_actor_loss)(actor.params, actor.apply_fn, o, actions[idx], old_lp[idx], adv[idx])
            cl, cg = jax.value_and_grad(_critic_loss)(critic.params, critic.apply_fn, o, returns[idx])
            return (actor.apply_gradients(grads=ag), critic.apply_gradients(grads=cg)), (al, cl)

        (actor, critic), losses = jax.lax.scan(mb_body, (actor, critic), jp.arange(n_minibatch))
        return (actor, critic), losses

    (actor, critic), losses = jax.lax.scan(epoch_body, (actor, critic), jax.random.split(key, epochs))
    al, cl = losses
    return actor, critic, jp.mean(al), jp.mean(cl)
