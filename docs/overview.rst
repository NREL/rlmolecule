========
Overview
========

This is a general-purpose `Reinforcement Learning <https://en.wikipedia.org/wiki/Reinforcement_learning>`_ (RL) library based primarily on the AlphaZero (AZ) style of RL, with `TensorFlow <https://www.tensorflow.org/>`_ as the ML backend. 

RLMolecule Overview
===================

As with any RL method, the goal of the agent is to learn which actions to take in a given environment to optimize a specific reward function (e.g., win the game). Our general workflow is as follows: 

#. Many agents begin "playing games" by performing MCTS rollouts (on CPUs) to explore the action space. These games and the corresponding rewards are stored in a database. Since the reward calculation can be expensive, the reward for a given state is calculated only once, and then retrieved from the database if seen again.
#. Once enough games have been played, the policy begins training (on a GPU) on the games and corresponding rewards, which are retrieved from the database. 
#. All agents use the most recent policy to continue playing games, and the process is repeated until terminated.

TODO include a diagram.


Software
========

*[TODO add a description of the software. Something like this:]* This software package aims to simplify the process of implementing any RL-based problem. The user simply needs to define the environment/state, an action space, and a reward function, and the software handles the rest. See the  `Example Problems`_ section below for problems we have already implemented.

*[TODO Update this section. Do we want to outline this from the gym perspective? Or the molecule perspective? From scratch? Or all three?]*

Problem
*******
A ``Problem`` class which implements the following four functions is required:

#. ``get_initial_state()``: defines the starting state for each game
#.  ``get_rewards(state)``: for a given state, calculates the reward
#. ``policy_model()``: defines the policy model (see examples)
#. ``get_policy_inputs(state)``: passes the desired inputs of a given state to train the policy model

The Environment/State
*********************
The user must define an environment/state class.


The Action Space
****************
The user must define an action space.



Example Problems
================

*[TODO For a couple of these problems, outline the problem, the implementation, how to solve it, and monitor/analyze the results.]*

* Hallway
* GridWorld
* Molecule Building

  * QED optimization
  * Stable radical optimization



Resources
=========

Below are some resources for learning about MCTS and AZ:

**AlphaZero**

* `AlphaZero <https://science.sciencemag.org/content/362/6419/1140>`_, *Science* 2018
* Single player AZ: `Ranked Reward: Enabling Self-Play Reinforcement Learning for Combinatorial Optimization <https://arxiv.org/abs/1807.01672>`_, *arXiv* 2018

**Monte Carlo Tree Search**

* `MCTS For Every Data Science Enthusiast <https://towardsdatascience.com/monte-carlo-tree-search-158a917a8baa>`_, towards data science, 2018
* `MCTS explanation video <https://www.youtube.com/watch?v=UXW2yZndl7U>`_
