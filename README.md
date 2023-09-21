# Python Goal Babbling

This package implements the *Goal Babbling* method developed at the **IRP** (Intitut für Robotik und Prozessinformatik/
Institute of Robotics and Process Control) / **Technische Universität Braunschweig**.

It is licensed under the EUPL (compatible to e.g. GPLv3).

:warning: This project is still work in progress. :warning:

## What is Goal Babbling?

See

* Rolf et al., 2010: "Goal Babbling Permits Direct Learning of Inverse Kinematics" (*IEEE Transactions on Autonomous Mental Deployment*)
* Rolf et al., 2011: "Online Goal Babbling for rapid bootstrapping of inverse models in high dimensions" (*IEEE International Conference on Development and Learning*)

for publications on Goal Babbling.

Goal Babbling (GB) is an online learning strategy for directly learning an Inverse Kinematics mapping of a robot. The
robot is used during the training process (hence it is called an *online* learning method) to explore the robot's
workspace while simultaneously and incrementally updating an Inverse Kinematics estimate, e.g. a Neural Network. While
the robot explores its workspace, a weighting mechanism tries to filter the gathered data in a way that only
goal-directed, efficient movements are learned.

## Getting Started

### Installation

For now this package can be installed locally:

```shell
cd python-goal-babbling
pip install .
```

If you intend to plot the state machine which is used under the hood:
```shell
pip install .[plot]
```

If you intend to build the documentation:
```shell
pip install .[doc]
```

(You can combine all optional dependencies like so: `pip install .[plot,doc,dev]`.)

### Documentation

You can build a Sphinx documentation by running `tox` in the root directory of the package:

```shell
tox
```

This will remove any previously built documentation artifacts and rebuild it using `sphinx-autodoc` and `sphinx-build`.
The documentation will be located under `docs/build/html/index.html`.

### Quick Start

```Python
from pygb import (
    EventSystem,
    GBParameters,
    GoalBabblingContext,
    GoalSet,
    RandomGoalSelector,
    TargetPerformanceStop,
    TimeBudgetStop,
    observes,
    setup_logging,
    vanilla_goal_babbling,
)
from pygb.interfaces import (
    AbstractForwardModel,
    AbstractInverseEstimate,
    AbstractEstimateCache,
)

class MyForwardModel(AbstractForwardModel):
  # You need to implement the Forward Model yourself
  ...

class MyInverseEstimate(AbstractInverseEstimate):
  # You need to wrap your learner, e.g. your NN
  ...

class MyCache(AbstractEstimateCache):
  # You need to implement a model cache which caches 'best' models somewhere (e.g. in 
  # your RAM/on your disk)
  ...

# Your paramters (sigma, number of epochs, etc.) go here
parameters = GBParameters(
  sigma=..., 
  sigma_delta=...,
  dim_act=...,
  dim_obs=3, # in case of a kinematics learing problem with x, y, z as the observation
  len_sequence=20,
  len_epoch=30,
  len_epoch_set=1,
  go_home_chance=0.1,
  home_action= ...,
  home_observation= ...,
)

# You MUST provide training and test goals in form of numpy arrays
goal_set = GoalSet(train=..., test=...)

state_machine = vanilla_goal_babbling(
  parameters=parameters,
  goal_sets=goal_set,
  forward_model=MyForwardModel(),
  inverse_estimate=MyInverseEstimate(),
  estimate_cache=MyCache(),
  goal_selector=RandomGoalSelector()
)

# This is optional but it is nice to see the current performance:
@observes("epoch-complete")
def log_progress(context):
  msg = f"Epoch {context.runtime_data.epoch_index}: "
  msg += f"RMSE {context.runtime_data.performance_error * 1000}mm"
  print(msg)

# Run the training for the full amount of epochs specified in the GBParameters 
state_machine.run()

# Now you have a trained inverse estimate which you can use to predict actions to reach 
# a targeted observation. Either use the 'most recent' inverse estimate, which might 
# not be the best one:
trained_estimate = context.inverse_estimate
print(trained_estimate.predict(observation=...))

# ... or load the best performing estimate from the most recent epoch set. As we only
# trained one epoch_set, we use '0' as the epoch set index:
trained_estimate = context.model_store.load(epoch_set=0)
print(trained_estimate.predict(observation=...))
```

If you want to train your estimate in distinct sets, e.g. in order to change parameters during the course of training or
change/extend your workspace, simply provide a list of `GBParameters` (or `GBParameterIncrement`s) and a list of
`GoalSet`s instead of a single instance each. The length of both lists must be equal. It determines the number of epoch
sets trained.

## Terminology

| Term             | Description                                                                                                                                                                                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Action           | An action a can be seen as the **cause of an observation**. In kinematics terms, an action represents a robot's joint configuration.                                                                                                                                |
| Observation      | An observation o represents the **observable state of a robot**. In kinematics terms, an observation is the robot's end effector position (or pose).                                                                                                                |
| Forward Model    | The Forward Model **executes an action a and yields an observation o**, i.e. represents the mapping f(a) = o. Thus, the forward model represents the robot.                                                                                                         |
| Inverse Estimate | The Inverse Estimate is an **estimation of the inverse mapping g(o) = a**. As it is an estimation it is usually expressed using an asterisk: g*(o) = a*. It is implemented as a machine learning model, i.e. a Neural Network or something like a Local Linear Map. |
| (Global) Goal    | Pre-recorded **observation from the robot's workspace**. Used to generate sequences (e.g. linear paths for IK learning) along which the inverse estimate is trained. Represents the training input.                                                                 |
| Local Goal       | Observation which is **generated during training as part of a sequence** between two (global) goals.                                                                                                                                                                |
| Sequence         | A **collection of local goals** between two (global) goals. In kinematics terms, a sequence is a linear path of positions in task space along which the robot moves ('reaching motions') and along which training samples are generated.                            |
| Epoch            | A **collection of multiple sequences**. The performance error on the provided test goals (+ optional additional goals) is calculated after a completed epoch.                                                                                                       |
| Epoch Set        | A **collection of multiple epochs**. Allows e.g. changing Goal Babbling parameters or training goals during training. The best estimate per epoch set is recorded in a model store.                                                                                 |

## Contributors

* Nico Weil
* Heiko Donat

This package is based on the work of M. Rolf et al.