#pragma once
class ActivationFunction
{
public : 
	virtual double activate(double x) = 0;
	virtual double differential_activate(double x) = 0;
};

