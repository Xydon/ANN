#pragma once
#include "index.h"
#include "Aggregrator.h"
#include "SigmoidActivation.h"
#include "Utils.h"

class Neuron {
public:
	DOUBLE weight;
	Aggregrator* aggregrator = new Aggregrator();
	ActivationFunction* activationFunction = new SigmoidActivation();
	double bias = ((double)rand()) / RAND_MAX;
	double previousIn = 0;
	DOUBLE previousPattern;
	double updateFactor;

	void initialize(size_t number_of_inputs) {
		weight.resize(number_of_inputs);

		for (int i = 0; i < number_of_inputs; ++i) {
			double randVal = ((double)rand()) / RAND_MAX;
			if (getRandomBool()) {
				randVal = -randVal;
			}
			weight.at(i) = randVal;
		}
	}

	void initialize(size_t number_of_inputs, double value) {
		weight.resize(number_of_inputs);

		for (int i = 0; i < number_of_inputs; ++i) {
			weight.at(i) = value;
		}
	}

	void initialize(size_t number_of_inputs, double (*getInitializeValue)(int)) {
		weight.resize(number_of_inputs);

		for (int i = 0; i < number_of_inputs; ++i) {
			weight.at(i) = getInitializeValue(i);
		}
	}

	double getOuput(DOUBLE input) {
		double product = aggregrator->aggregrate(input, weight);
		double in_value = product + bias;
		previousIn = in_value;
		previousPattern = input;
		return activationFunction->activate(in_value);
	}

	virtual void updateWeight(double factor, double learning_rate) {

		double updateFactor = factor * activationFunction->differential_activate(previousIn);
		this->updateFactor = updateFactor;

		size_t n = weight.size();
		for (int i = 0; i < n; ++i) {
			weight.at(i) = weight.at(i) + learning_rate * updateFactor * previousPattern.at(i);
		}

		bias = bias + learning_rate * updateFactor;
	}

	DOUBLE getFactor() {
		DOUBLE factor;

		for (auto val : weight) {
			factor.push_back(val * updateFactor);
		}

		return factor;
	}
};