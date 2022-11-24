#pragma once
#include "ActivationFunction.h"

class SigmoidActivation : public ActivationFunction
{
public : 
	double activate(double x);
	double differential_activate(double x);
};

