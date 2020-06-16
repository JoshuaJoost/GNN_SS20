# kernel imputs

# own imputs
from constants import activationFunction, activationFunctionDerived_1

# main
h1I = 0.65239106 + (0.72097573 * -0.95020846) + (1.55695014 * 0.5414358)
print(h1I)
h2I = -0.96525123 + (0.72097573 * 0.53384854) + (1.55695014 * -0.38728552)
print(h2I)

h1O = activationFunction(h1I)
print(h1O)
h2O = activationFunction(h2I)
print(h2O)

o1I = -0.43790297 + (0.692174 * -0.81445714) + (0.234451 * 0.48348276)
print(o1I)
o1O = activationFunction(o1I)
print(o1O)

