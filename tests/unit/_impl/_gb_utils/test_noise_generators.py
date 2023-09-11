from unittest.mock import MagicMock, PropertyMock

import numpy as np

from pygb import GBNoiseGenerator, GoalBabblingContext


def dummy_context() -> MagicMock:
    context_mock = MagicMock(spec=GoalBabblingContext)
    context_mock.current_parameters = PropertyMock(dim_act=3, dim_obs=2, sigma=0.1, sigma_delta=0.01)

    return context_mock


def test_init() -> None:
    context_mock = dummy_context()
    rng = np.random.default_rng(seed=42)
    generator = GBNoiseGenerator(context_mock, rng)

    assert generator.sigma.shape == (1, 3)
    assert generator.sigma_delta.shape == (1, 3)
    assert generator.A.shape == (3, 2)
    assert generator.b.shape == (3, 1)


def test_generate() -> None:
    context_mock = dummy_context()
    rng = np.random.default_rng(seed=42)
    generator = GBNoiseGenerator(context_mock, rng)

    noise = generator.generate(np.array([4.0, 2.0]))

    assert noise.shape == (3,)


def test_update() -> None:
    context_mock = dummy_context()
    rng = np.random.default_rng(seed=42)
    generator = GBNoiseGenerator(context_mock, rng)

    prev_A = generator.A.copy()
    prev_b = generator.b.copy()

    generator.update()

    assert np.all(generator.A != prev_A)
    assert np.all(generator.b != prev_b)

    assert generator.A.shape == (3, 2)
    assert generator.b.shape == (3, 1)
