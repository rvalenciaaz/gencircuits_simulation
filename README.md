# gencircuits_simulation

**The genetic repressilator and repressive circuits**

One of the simplest genetic circuits is the repressilator, a three-gene system that exhibit oscillations for some of the system parameters.

To compile the C code, use

`gcc -shared -o libRepressilator.so -fPIC repressilator.c -lgsl -lgslcblas -lm`