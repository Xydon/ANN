#include "index.h" 


void printVector(vector<double> v) {
	for (auto i : v) {
		cout << i << ' ';
	}
	cout << endl;
}

class Aggregrator {
public : 
	virtual double aggregrate(DOUBLE pattern, DOUBLE weight) {
		if (pattern.size() != weight.size())
			throw invalid_argument("pattern and weight size must match to aggregrate"); 

		size_t n = pattern.size(); 

		double aggregrate_value = 0; 
		for (int i = 0; i < n; ++i) {
			aggregrate_value += pattern.at(i) * weight.at(i); 
		}

		return aggregrate_value; 
	}
};

class ActivationFunction {
public: 
	virtual double activate(double x) {
		return 1 / (1 + exp(-x)); 
	}

	virtual double differential_activate(double x) {
		return activate(x)*(1 - activate(x)); 
	}
};


class Neuron {
public : 
	DOUBLE weight; 
	Aggregrator* aggregrator = new Aggregrator(); 
	ActivationFunction* activationFunction = new ActivationFunction();
	double bias = ((double)rand())/RAND_MAX;
	double previousIn = 0; 
	DOUBLE previousPattern; 
	double updateFactor; 

	void initialize(size_t number_of_inputs) {
		weight.resize(number_of_inputs); 

		for (int i = 0; i < number_of_inputs; ++i) {
			double randVal = ((double)rand()) / RAND_MAX;
			if (randVal > 0.6) {
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

class NeuronLayer {
private : 

public : 
	vector<Neuron *> layerNeurons; 
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

		for (Neuron* neuron: layerNeurons) {
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

int getIndexOfMax(DOUBLE output) {
	int indx = -1; 
	for (int i = 0; i < output.size(); ++i) {
		if (indx == -1) {
			indx = 0; 
		}

		if (output.at(indx) < output.at(i)) {
			indx = i; 
		}
	}
	return indx; 
}

DOUBLE getErrorVector(DOUBLE vec1, DOUBLE vec2) {
	DOUBLE ans; 

	if (vec1.size() != vec2.size()) {
		throw invalid_argument("size of first and the second vector don't match for calculation of error"); 
	}

	size_t size = vec1.size(); 
	for (int i = 0; i < size; ++i) {
		ans.push_back(vec1.at(i) - vec2.at(i)); 
	}

	return ans; 
}

DOUBLE createVec(size_t size, int pos) {
	DOUBLE vec(size, 0);
	vec.at(pos) = 1; 
	return vec; 
}


double norm(DOUBLE vec) {
	double sum = 0; 
	for (auto i : vec) {
		sum += i * i; 
	}
	return sqrt(sum); 
}

int main() {
	vector<int> input; 
	vector<int> output; 

	for (int i = 0; i < 10; ++i) {
		input.push_back(i); 
		output.push_back(i); 
		input.push_back(i);
		output.push_back(i);
		input.push_back(i);
		output.push_back(i);
		input.push_back(i);
		output.push_back(i);
		input.push_back(i);
		output.push_back(i);
		input.push_back(i);
		output.push_back(i);
		input.push_back(i);
		output.push_back(i);
		input.push_back(i);
		output.push_back(i);
		input.push_back(i);
		output.push_back(i);
	}

	NeuronLayer hiddenLayer(0.001, 80, 1); 
	NeuronLayer outputLayer(0.001, 10, 80); 

	size_t n = input.size(); 

	for (int k = 0; k < 2000; ++k)
	for (int i = 0; i < n; ++i) {
		auto hiddenOutput = hiddenLayer.getOutput({ (double)input.at(i) });
		auto finalOutput = outputLayer.getOutput(hiddenOutput);

		auto errVec = getErrorVector(createVec(finalOutput.size(), output.at(i)), finalOutput); 
		if (norm(errVec) < 0.2) break; 
		outputLayer.updateWeight(errVec); 
		auto outputLayerFactors = outputLayer.getFactors();

		hiddenLayer.updateWeight(outputLayerFactors); 
	}

	cout << "testing" << endl; 
	for (int i = 0; i < n; ++i) {
		auto hiddenOutput = hiddenLayer.getOutput({ (double)input.at(i) });
		auto finalOutput = outputLayer.getOutput(hiddenOutput); 
		cout << input.at(i) << ' ' << getIndexOfMax(finalOutput) << endl;
	}
	


	return 0; 
}