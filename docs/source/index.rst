.. Python Goal Babbling documentation master file, created by
   sphinx-quickstart on Thu Sep  7 19:57:15 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to python-goal-gabbling's documentation!
================================================

This framework implements the *Goal Babbling* method developed at IRP (Institute of Robotics and Process Control,
Technische Universität Braunschweig):

* Rolf et al., 2010: "Goal Babbling Permits Direct Learning of Inverse Kinematics" (*IEEE  Transactions on Autonomous Mental Deployment*)
* Rolf et al., 2011: "Online Goal Babbling for rapid bootstrapping of inverse models in high dimensions" (*IEEE Interational Conference on Development and Learning*)

Goal Babbling (GB) is an online learning strategy for directly learning an Inverse Kinematics mapping of a robot. The
robot is used during the training process (hence it is called an *online* learning method) to explore the robot's
workspace while simultaneously and incrementally updating an Inverse Kinematics estimate, e.g. a Neural Network. While
the robot explores its workspace, a weighting mechanism tries to filter the gathered data in a way that only
goal-directed, efficient movements are learned.

This framework takes the original idea a little further: It is not restricted to inverse kinematics learning, but allows
to use Goal Babbling in any sort of action/observation-related online learning process. Every component can be replaced
by a different implementation, so as long as the use case is suitable for Goal Babbling, `pygb` **should** do the job.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   how_to
   extending
   pygb



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
