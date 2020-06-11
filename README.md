# Multilayer Perceptron

A (hopefully!) simple two-layer perceptron with backpropagation of errors.

# Inspiration

Two crucial elements in this code were greatly influenced by other's wonderful work:
1. module `diff.jl` is my variation of https://github.com/Roger-luo/YAAD.jl from Roger Luo:
the main idea is Roger's, I've adjusted some parts to my taste. This module covers enough constructions to allow teaching CNN;
2. module `dual.jl` started based on Jarrett Revels' ForwardDiff and Mike Iness' diff-zoo (https://github.com/MikeInnes/diff-zoo);
3. module `optimizers.jl` came from a great book: Mykel Kochenderfer and Tim Wheeler, "Algorithms for Optimization", MIT Press, 2019.

# Motivation

Did it as a way of learning implementation details and challanges of training neural networks.
This simple example allows to explore the relationship between neural networks a simple least mean square approximation.
