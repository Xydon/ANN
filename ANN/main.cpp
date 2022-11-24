#include "index.h" 
#include "Network.h"
#include "ActivationFunction.h" 
#include "SigmoidActivation.h"
#include <utility>

#include <fstream>

constexpr const double lp = 0.3;

SigmoidActivation activation; 

ifstream dataset_file("data_banknote_authentication.txt");	
ifstream test("test.txt"); 

class NetworkLayer {
public : 
	MatrixXd layerMatrix, layerBias, layerDel, layerIn; 
	size_t previous_layer_size, current_layer_size; 
	ActivationFunction* activationFunction = nullptr; 
	double learning_rate = 0.01;

	NetworkLayer(size_t previous_layer_size, size_t current_layer_size, double learning_rate, ActivationFunction * activationFunction = new SigmoidActivation()) : 
		previous_layer_size(previous_layer_size), 
		current_layer_size(current_layer_size), 
		activationFunction(activationFunction), 
		learning_rate(learning_rate) {

		layerMatrix = MatrixXd::Random(current_layer_size, previous_layer_size); 
		layerBias = MatrixXd::Random(1, current_layer_size); 
		layerDel = MatrixXd::Random(1, current_layer_size); 
		layerIn = MatrixXd::Random(1, current_layer_size); 
	}

	MatrixXd getOutput(MatrixXd input) {
		MatrixXd layerOutput(1, current_layer_size); 

		for (int i = 0; i < current_layer_size; ++i) {
			MatrixXd weight = layerMatrix.row(i).transpose(); 
			layerOutput(i) = (input * weight).sum() + layerBias(i); 
			layerIn(i) = layerOutput(i); 
			layerOutput(i) = activationFunction->activate(layerOutput(i)); 
		}

		return layerOutput; 
	}

	void updateWeights(double del_in, MatrixXd input) {
		for (int i = 0; i < current_layer_size; ++i) {
			double del_j = del_in * activation.differential_activate(layerIn(i)); 
			layerDel(i) = del_j; 
			
			for (int k = 0; k < previous_layer_size; ++k) {
				layerMatrix(i, k) += lp * del_j * input(k); 
			}

			layerBias(i) += lp * del_j; 
		}
	}

	void updateWeights(NetworkLayer frontLayer, MatrixXd input) {
		for (int i = 0; i < current_layer_size; ++i) {
			double del_in = 0;
			for (int k = 0; k < frontLayer.current_layer_size; ++k) {
				double layerDel_k = frontLayer.layerDel(k);
				double connectionWeight = frontLayer.layerMatrix(i, k);

				del_in += connectionWeight * layerDel_k;
			}
			double del_j = del_in * activation.differential_activate(layerIn(i));
			layerDel(i) = del_j; 

			for (int k = 0; k < previous_layer_size; ++k) {
				layerMatrix(i, k) += learning_rate * del_j * input(k);
			}

			layerBias(i) += learning_rate * del_j; 
		}

	}

	MatrixXd getDels() {
		return layerDel; 
	}
};


class Loader {
private:
	string filename = "data_banknote_authentication.txt"; 

	MatrixXd append(MatrixXd& matrix, MatrixXd data) {
		matrix.conservativeResize(matrix.rows() + 1, matrix.cols()); 
		matrix.row(matrix.rows() - 1) = data; 
		return matrix; 
	}
	const int cols = 5; 

public : 

	MatrixXd getDataMatrix() { 
		string inp; 
		
		MatrixXd dataMatrix(0, cols);

		if (!dataset_file) {
			cout << "failed to open dataset" << endl; 
		}


		while (getline(dataset_file, inp)) {
			MatrixXd dataRow(1, cols);

			int currCol = 0; 

			string val = ""; 

			for (auto ch : inp) {
				if (ch == ',') {
					dataRow(currCol) = stod(val); 
					currCol++; 
					val = ""; 
				}
				else val.push_back(ch); 
			}
			
			dataRow(currCol) = stod(val); 

			append(dataMatrix, dataRow); 
		}

		return dataMatrix; 
	}

	MatrixXd randomize(MatrixXd matrix) {
		size_t rows = matrix.rows(); 
		for (int i = 0; i < 1000; ++i) {
			int i1 = (rand() % rows); 
			int i2 = (rand() % rows); 
			MatrixXd temp = matrix.row(i1); 
			matrix.row(i1) = matrix.row(i2); 
			matrix.row(i2) = temp; 
		}
		return matrix; 
	}

	pair<pair<MatrixXd, MatrixXd>, pair<MatrixXd, MatrixXd>> train_test_split(double ratio = 0.3) { 
		
		auto dataMatrix = randomize(getDataMatrix()); 

		if (ratio > 1) ratio = 0.3; 
		ratio = min(0.5, ratio); 

		int matrixCols = dataMatrix.cols(); 

		int trainRowCount = round(dataMatrix.rows() * (1-ratio));
		MatrixXd trainData(trainRowCount, matrixCols - 1);
		MatrixXd trainResults(trainRowCount, 1);

		for (int i = 0; i <= trainRowCount; ++i) {
			MatrixXd data = dataMatrix.row(i);
			data.conservativeResize(1, static_cast<Eigen::Index>(matrixCols) - 1);
			trainData.row(i) = data;
			trainResults(i) = dataMatrix.row(i)(matrixCols - 1);
		}

		int testRowCount = (int)dataMatrix.rows() - trainRowCount;
		MatrixXd testData(testRowCount, matrixCols - 1);
		MatrixXd testResults(testRowCount, 1);

		int offset = min((int)dataMatrix.rows(), trainRowCount + 1); 

		for (int i = offset; i < dataMatrix.rows(); ++i) {
			MatrixXd data = dataMatrix.row(i);
			data.conservativeResize(1, matrixCols - 1);
			testData.row(i-offset) = data; 
			testResults(i-offset) = dataMatrix.row(i)(matrixCols - 1); 
		}
		
		return {
			{trainData, trainResults},
			{testData, testResults}
		}; 
	}
};

void test2() {
	NetworkLayer hiddenLayer1(4, 7, 0.01);
	NetworkLayer hiddenLayer2(7, 3, 0.01);
	NetworkLayer outputLayer(3, 1, 0.01);

	MatrixXd dataset(6, 4);
	dataset << 1, 2, 3, 4,
		1, 2, 5, 2,
		1, 6, 3, 2,
		-1, -2, -3, -4,
		-5, -4, -10, -39,
		-2, -20, -3, -5;

	MatrixXd dataset2(3, 4);
	dataset2 << 1, 1, 1, 1,
		2, 2, 2, 2,
		-1, -1, -1, -1;


	vector<int> results = {
		1, 1, 1, 0, 0, 0
	};


	for (int iter = 0; iter < 1000; ++iter)
		for (int i = 0; i < 6; ++i) {
			auto o1 = hiddenLayer1.getOutput(dataset.row(i));
			auto o2 = hiddenLayer2.getOutput(o1);
			auto o3 = outputLayer.getOutput(o2);

			const double error = results[i] - o3.sum();

			outputLayer.updateWeights(error, o2);
			hiddenLayer2.updateWeights(outputLayer, o1);
			hiddenLayer1.updateWeights(hiddenLayer2, dataset.row(i));
		}

	for (int i = 0; i < 3; ++i) {
		auto o1 = hiddenLayer1.getOutput(dataset2.row(i));
		auto o2 = hiddenLayer2.getOutput(o1);
		auto o3 = outputLayer.getOutput(o2);

		// cout << round(o3.sum()) << endl; 
	}
}

int main() {

	Loader dataLoader; 
	auto dataset = dataLoader.train_test_split(); 

	constexpr const double learning_rate = 0.01; 

	// the layers
	NetworkLayer first(4, 100, learning_rate),
		second(100, 30, learning_rate),
		third(30, 5, learning_rate),
		output(5, 1, learning_rate); 

	// training part 
	auto trainData = dataset.first.first;
	auto trainResults = dataset.first.second; 
	
	int rows = trainResults.rows(); 

	for (int i = 0; i < rows; ++i) {

	}

	return 0; 
}