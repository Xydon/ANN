#include "index.h" 
#include "NeuronLayer.h"

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