#pragma once
#include "Neuron.h"
#include "index.h"

class NeuronLayer {
private:

public:
	vector<Neuron*> layerNeurons;
	double learning_rate;

	NeuronLayer(double learning_rate, size_t dimension, size_t input_size) : learning_rate(learning_rate) {
		layerNeurons.resize(dimension);

		for (int i = 0; i < dimension; ++i) {
			Neuron* newNeuron = new Neuron();
			newNeuron->initialize(input_size);

			layerNeurons.at(i) = newNeuron;
		}
	}

	// feed forward 
	DOUBLE getOutput(DOUBLE input) {

		size_t size = layerNeurons.size();
		DOUBLE output(size);

		for (int i = 0; i < size; ++i) {
			output.at(i) = layerNeurons.at(i)->getOuput(input);
		}

		return output;
	}

	// back propagation of error 
	vector<double> getFactors() {
		vector<double> factors;
		size_t input_size = 0;

		for (Neuron* neuron : layerNeurons) {
			auto neuronFactor = neuron->getFactor();

			if (factors.size() == 0) {
				factors = neuronFactor;
				input_size = factors.size();
				continue;
			}

			for (int i = 0; i < input_size; ++i) {
				factors.at(i) += neuronFactor.at(i);
			}
		}

		return factors;
	}

	void updateWeight(vector<double> factors) {
		if (factors.size() != layerNeurons.size())
			throw invalid_argument("factor dimension don't match layerNeurons dimension");

		size_t n = factors.size();

		for (int i = 0; i < n; ++i) {
			Neuron* neuron = layerNeurons.at(i);
			neuron->updateWeight(factors.at(i), learning_rate);
		}
	}
};

