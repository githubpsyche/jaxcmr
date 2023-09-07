from evosax import Strategies, ParameterReshaper, EvoState
from jax_tqdm import loop_tqdm
from functools import partial
from jax import numpy as jnp, random, lax, jit, tree_util
from jax.nn import sigmoid
from beartype.typing import Any
from jaxcmr.helpers import PRNGKeyArray

__all__ = ["FittingStrategy"]


def scale_params(params: dict, bounds: dict[str, list[float]]):
    return tree_util.tree_map(
        lambda x, y: y[0] + sigmoid(x) * (y[1] - y[0]), params, bounds
    )


class FittingStrategy:
    def __init__(
        self,
        objective_fn,
        parameters,
        strategy_name="DE",
        maxiter=1000,
        popsize=15,
        mutation=1.0,
        recombination=0.7,
        disp=False,
    ):
        self.reshape_params = ParameterReshaper(
            {key: parameters["fixed"][key] for key in parameters["free"]}
        ).reshape
        self.strategy = Strategies[strategy_name](
            popsize=popsize, num_dims=len(parameters["free"])
        )
        self.es_params = self.strategy.default_params.replace(
            cross_over_rate=recombination, diff_w=mutation
        )
        self.objective_fn = objective_fn
        self.parameters = parameters
        self.maxiter = maxiter
        if disp:
            self._strategy_step = loop_tqdm(self.maxiter)(self._strategy_step)

    def _strategy_step(self, tmp: Any, state_input: EvoState):
        """Helper es step to iterate through."""
        rng, state = state_input
        rng, rng_iter = random.split(rng)
        x, state = self.strategy.ask(rng_iter, state, self.es_params)
        x_dict = scale_params(self.reshape_params(x), self.parameters["free"])
        fitness = self.objective_fn(x_dict)
        state = self.strategy.tell(x, fitness, state, self.es_params)
        return [rng, state]

    @partial(jit, static_argnums=(0,))
    def __call__(self, rng) -> EvoState:

        state = self.strategy.initialize(rng, self.es_params)
        return lax.fori_loop(0, self.maxiter, self._strategy_step, [rng, state])[1]

    def format_fit_result(self, evosax_result):
        fitted_parameters = dict(
            self.parameters, likelihood=float(evosax_result.best_fitness)
        )
        best_member = scale_params(
            self.reshape_params(jnp.expand_dims(evosax_result.best_member, 0)),
            self.parameters["free"],
        )

        for key in self.parameters["free"]:
            fitted_parameters["fixed"][key] = float(best_member[key])

        return fitted_parameters
