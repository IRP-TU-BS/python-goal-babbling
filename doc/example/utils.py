import numpy as np
from master_utils.learner.llm import LLM
from pydctr.discrete_shape_estimation.PrintedTube import PrintedTube
from pydctr.discrete_shape_estimation.TorsionallyRigidCTRNew import TorsionalRigidCTR
from spatialmath import SE3

from pygb.interfaces import AbstractForwardModel, AbstractInverseEstimator


class ForwardModel(AbstractForwardModel):
    def __init__(self, joint_limits: np.ndarray) -> None:
        tubes = [
            PrintedTube(
                len=110,
                arc_len=100,
                u_0=4.37,
                flex_rigidity=0.29343457132232925,
                tor_rigidity=0.056429725254294086,
                theta=0,
                rho=0,
                mass=0,
            ),
            PrintedTube(
                len=165,
                arc_len=100,
                u_0=12.4,
                flex_rigidity=0.04084070449666733,
                tor_rigidity=0.007853981633974485,
                theta=0,
                rho=0,
                mass=0,
            ),
            PrintedTube(
                len=210,
                arc_len=41,
                u_0=28.0,
                flex_rigidity=0.0036226490286707306,
                tor_rigidity=0.0006966632747443713,
                theta=0,
                rho=0,
                mass=0,
            ),
        ]
        self.ctcr = TorsionalRigidCTR(base_pose=SE3().Rz(theta=270, unit="deg").A, tubes=tubes)
        self.joint_limit = joint_limits

    def forward(self, action: np.ndarray) -> np.ndarray:
        # action formatted as [a1, a2, a3, b1, b2, b3] in rad and m respectively
        self.ctcr.move_robot(thetas=action[:3], rhos=action[3:] * 1000)

        _, p_m = self.ctcr.get_deformed_shape(wrench=np.zeros(6), link_len=1)

        return p_m[-1]

    def forward_batch(self, action_batch: np.ndarray) -> np.ndarray:
        batch = [self.forward(action) for action in action_batch]

        return np.asarray(batch)

    def clip(self, action: np.ndarray) -> np.ndarray:
        _action = action.copy()

        if squeeze := action.ndim == 2:
            _action = _action.squeeze()

        _action = _action.clip(min=self.joint_limit[0, :], max=self.joint_limit[1, :])

        n_tubes = 3
        for tube_index in range(1, n_tubes):
            if _action[n_tubes + tube_index] < _action[n_tubes + tube_index - 1]:
                _action[n_tubes + tube_index] = _action[n_tubes + tube_index - 1]

        if squeeze:
            return _action.reshape(1, -1)

        return _action

    def clip_batch(self, action_batch: np.ndarray) -> np.ndarray:
        batch = [self.clip(action) for action in action_batch]
        return np.asarray(batch)


class InverseEstimator(AbstractInverseEstimator):
    def __init__(self, observation0: np.ndarray, action0: np.ndarray, radius: float, learning_rate=float) -> None:
        self.llm = LLM(x0=observation0, y0=action0, radius=radius, learning_rate=learning_rate)

    def fit(self, observation: np.ndarray, action: np.ndarray, weight: float) -> float:
        self.llm.fit_sample(observation, action, weight)

        rmse = np.sqrt(np.mean(action - self.llm.predict_sample(observation)) ** 2)
        return rmse

    def fit_batch(self, observation_batch: np.ndarray, action_batch: np.ndarray, weights: np.ndarray) -> float:
        raise NotImplementedError

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self.llm.predict(observation)

    def predict_batch(self, observation_batch: np.ndarray) -> np.ndarray:
        batch = [self.predict(observation) for observation in observation_batch]

        return np.asarray(batch)
