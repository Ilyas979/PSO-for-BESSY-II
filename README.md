# PSO-for-BESSY-II
Particle Swarm Optimization Method written for BESSY II (Berlin)

The file pso.py comprises two different global optimization algorithms.

The first one is "sopso", which means it can be used for single objective opimization. Apart from the standard parameters like swarmsize, maximum number of iterations, optimized function and its boundaries, the user can choose from several options for "initial distribution" of the particles, "weight decrease strategy" and "mutation rate". "Bait" may be used if the user has some guesses about the position of the optimimum. The code can be safely interrupted with "ctrl+C" and continued from the same point with "continue_flag".

The second one is "mopso", which means it can be used to optimize several objectives at once. In addition to parameters that can be adjusted in single objective version, one can change the number of elements in external repository in multi-objective algorithm.
