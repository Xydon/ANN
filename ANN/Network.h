#pragma once
#include "index.h"


class Network
{
private : 
	bool hasAssembled; 
public : 
	void train(vector<DOUBLE> data, DOUBLE result); 
	void assemble(); 
};

