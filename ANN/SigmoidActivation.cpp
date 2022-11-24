#include "SigmoidActivation.h"
#include <math.h>

double SigmoidActivation::activate(double x) {
	return 1 / (1 + exp(-x));
}

double SigmoidActivation::differential_activate(double x) {
	return activate(x) * (1 - activate(x));
}