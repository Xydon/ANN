#include "index.h" 
#include "Network.h"
#include "ActivationFunction.h" 
#include "SigmoidActivation.h"
#include <map>

constexpr const double lp = 0.3;

SigmoidActivation activation; 

ifstream dataset_file("data_banknote_authentication.txt");	
ofstream error_file("error.txt"); 
ifstream iris_file("iris.txt"); 
ifstream transfusion_file("transfusion.txt"); 

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

	void updateWeights(MatrixXd errorVec, MatrixXd input) {
		for (int i = 0; i < current_layer_size; ++i) {
			double del_j = errorVec(i) * activation.differential_activate(layerIn(i));
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

class LoaderUtils {
public : 
	MatrixXd append(MatrixXd& matrix, MatrixXd data) {
		matrix.conservativeResize(matrix.rows() + 1, matrix.cols());
		matrix.row(matrix.rows() - 1) = data;
		return matrix;
	}

	pair<MatrixXd, MatrixXd> extractLabel(MatrixXd matrix) {
		const int row = matrix.rows(), cols = matrix.cols();

		MatrixXd result(matrix.rows(), 1);
		MatrixXd data(matrix.rows(), matrix.cols() - 1);
		for (int i = 0; i < matrix.rows(); ++i) {
			MatrixXd matrixRow = matrix.row(i);
			result(i) = matrixRow(cols - 1);
			matrixRow.conservativeResize(1, cols - 1);
			data.row(i) = matrixRow;
		}

		return { data, result };
	}

	void randomize(pair<MatrixXd, MatrixXd> &dataset) {
		srand(time(NULL)); 
		int rows = dataset.first.rows(); 
		int iter = (double(rows) * 0.8);
		
		for (int i = 0; i < iter; ++i) {
			int i1 = rand() % rows; 
			int i2 = rand() % rows; 


			// swapping the rows 
			MatrixXd temp = dataset.first.row(i1); 
			dataset.first.row(i1) = dataset.first.row(i2); 
			dataset.first.row(i2) = temp; 

			// swapping the values in the col 
			double tempval = dataset.second(i1); 
			dataset.second(i1) = dataset.second(i2); 
			dataset.second(i2) = tempval; 
		}
	}

	pair<pair<MatrixXd, MatrixXd>, pair<MatrixXd, MatrixXd>> train_test_split(pair<MatrixXd, MatrixXd> dataset) {
	
		int col = dataset.first.cols(); 
		int row = dataset.first.rows(); 

		int trainRow = ((double)row * 0.7); 
		int testRow = row - trainRow; 

		MatrixXd data = dataset.first, results = dataset.second; 

		MatrixXd trainData(trainRow, col), trainResult(trainRow, 1); 
		MatrixXd testData(testRow, col), testResult(testRow, 1); 

		for (int i = 0; i < trainRow; ++i) {
			trainData.row(i) = data.row(i); 
			trainResult(i) = results(i); 
		}

		for (int i = trainRow; i < row; ++i) {
			testData.row(i - trainRow) = data.row(i); 
			testResult(i - trainRow) = results(i); 
		}

		return {
			{trainData, trainResult}, 
			{testData, testResult}
		};
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
		srand(time(NULL)); 
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


class BankNotesModel {
public: 
	void execute() {
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

		int rows_train = trainResults.rows();

		for (int k = 0; k < 100; ++k) {
			clock_t startTime = clock();

			double err_sum = 0;

			for (int i = 0; i < rows_train; ++i) {
				MatrixXd data = trainData.row(i);

				MatrixXd o1 = first.getOutput(data);
				MatrixXd o2 = second.getOutput(o1);
				MatrixXd o3 = third.getOutput(o2);
				MatrixXd op = output.getOutput(o3);

				const double error = trainResults(i) - op.sum();

				err_sum += abs(error);

				output.updateWeights(error, o3);
				third.updateWeights(output, o2);
				second.updateWeights(third, o1);
				first.updateWeights(second, data);
			}
			error_file << (err_sum / rows_train) << endl;
			cout << "COMPLETED EPOCH " << (k + 1) << " in " << (clock() - startTime) / (double)CLOCKS_PER_SEC << " seconds" << endl;
		}


		// testing part 
		auto testData = dataset.second.first;
		auto testResults = dataset.second.second;

		int rows_test = testResults.rows();

		int correct_count = 0;

		for (int i = 0; i < rows_test; ++i) {
			MatrixXd data = testData.row(i);

			MatrixXd o1 = first.getOutput(data);
			MatrixXd o2 = second.getOutput(o1);
			MatrixXd o3 = third.getOutput(o2);
			MatrixXd op = output.getOutput(o3);

			if (round(op.sum()) == testResults(i)) ++correct_count;

		}

		cout << "accuracy achieved ";
		cout << (((double)correct_count / (double)rows_test) * 100) << "%" << endl;

	}
};

class ErrorUtils {
public : 
	static MatrixXd classErrorVector(size_t classNumber, size_t classCount) {
		MatrixXd vec(1, classCount); 
		vec = MatrixXd::Zero(1, classCount); 
		vec(classNumber) = 1; 
		return vec; 
	}

	static int getIndeOfMax(MatrixXd vec) {
		
		MatrixXd copy = vec; 
		double sum = 0; 
		for (int i = 0; i < vec.cols(); ++i) {
			vec(0, i) = exp(vec(0, i)); 
			sum += vec(0, i); 
		}
		vec = vec / sum; 

		int ans = 0;
		for (int i = 0; i < vec.cols(); ++i) {
			if (vec(0, i) > vec(0, ans)) {
				ans = i;
			}
		}
		return ans; 

	}
};

class IrisModel  : public LoaderUtils {
public :
	map<string, int> classMap; 
	vector<string> reverseMap; 
	int cols = 5; 
	
	IrisModel() {
		classMap["Iris-setosa"] = 0; 
		classMap["Iris-versicolor"] = 1; 
		classMap["Iris-virginica"] = 2; 

		reverseMap = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" }; 
	}

	MatrixXd getMatrix() {
		MatrixXd dataMatrix(0, 5); 
		string inp; 

		while (getline(iris_file, inp)) {
			MatrixXd dataRow(1, cols); 
			int currCol = 0; 
			string val = ""; 

			for (auto ch : inp) {
				if (ch == ',') {
					dataRow(currCol) = stod(val); 
					currCol++; 
					val = ""; 
				}
				else { 
					val.push_back(ch); 
				}
			}

			dataRow(currCol) = classMap[val]; 

			append(dataMatrix, dataRow); 
		}

		return dataMatrix; 
	}

	void eval() {
		pair<MatrixXd, MatrixXd> dataset_raw = extractLabel(getMatrix()); 
		randomize(dataset_raw); 

		auto dataset = train_test_split(dataset_raw); 

		const double learning_rate = 0.0001;

		NetworkLayer first(4, 100, learning_rate);
		NetworkLayer second(100, 50, learning_rate);
		NetworkLayer third(50, 10, learning_rate);
		NetworkLayer finalLayer(10, 3, learning_rate);

		// training
		auto trainData = dataset.first.first;
		auto trainResults = dataset.first.second;

		int train_row_count = trainData.rows();

		/*
			EPOCH COUNT = 1000 
		*/
		for(int k = 0; k < 1000; ++k)
		for (int i = 0; i < train_row_count; ++i) {
			auto data = trainData.row(i);

			auto o1 = first.getOutput(data);
			auto o2 = second.getOutput(o1);
			auto o3 = third.getOutput(o2);
			auto op = finalLayer.getOutput(o3);


			// calculating error 
			auto errorVec = ErrorUtils::classErrorVector(trainResults(i), 3) - op;


			finalLayer.updateWeights(errorVec, o3); 
			third.updateWeights(finalLayer, o2); 
			second.updateWeights(third, o1); 
			first.updateWeights(second, data); 
		}

		// testing 
		auto testData = dataset.second.first; 
		auto testResults = dataset.second.second; 

		int test_row_count = testData.rows(); 


		int correct_count = 0; 
		for (int i = 0; i < test_row_count; ++i) {
			auto data = testData.row(i); 

			auto o1 = first.getOutput(data);
			auto o2 = second.getOutput(o1);
			auto o3 = third.getOutput(o2);
			auto op = finalLayer.getOutput(o3);

			if (ErrorUtils::getIndeOfMax(op) == testResults(i)) ++correct_count; 
		}

		cout << "accuracy obtained --- " << ((correct_count / (double)test_row_count) * 100) << " %" << endl; 
	}
};

class TransfusionModel : public LoaderUtils {
public : 

	int cols = 5; 

	MatrixXd getMatrix() {
		string inp;

		MatrixXd dataMatrix(0, cols);

		if (!transfusion_file) {
			cout << "failed to open dataset" << endl;
		}


		while (getline(transfusion_file, inp)) {
			MatrixXd dataRow(1, cols);

			int currCol = 0;

			string val = "";

			for (auto ch : inp) {
				if (ch == ' ') continue; 
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

	void eval() {
		auto dataset_raw = extractLabel(getMatrix());
		// randomize(dataset_raw);

		auto dataset = train_test_split(dataset_raw);

		const double learning_rate = 0.001;

		NetworkLayer first(4,25 , learning_rate);
		NetworkLayer second(25, 10, learning_rate);
		NetworkLayer third(10, 5, learning_rate);
		NetworkLayer finalLayer(5, 1, learning_rate);

		// training
		auto trainData = dataset.first.first;
		auto trainResults = dataset.first.second;

		int train_row_count = trainData.rows();

		for (int k = 0; k < 1000; ++k) {
			double error_sum = 0; 
			for (int i = 0; i < train_row_count; ++i) {
				auto data = trainData.row(i);
				auto o1 = first.getOutput(data);
				auto o2 = second.getOutput(o1);
				auto o3 = third.getOutput(o2);
				auto op = finalLayer.getOutput(o3);


				// calculating error 
				double error = trainResults(i) - op.sum();

				error_sum += abs(error); 

				finalLayer.updateWeights(error, o3);
				third.updateWeights(finalLayer, o2);
				second.updateWeights(third, o1);
				first.updateWeights(second, data);
			}
			error_file << (error_sum / (double)train_row_count) << endl; 
		}

		// testing 
		auto testData = dataset.second.first;
		auto testResults = dataset.second.second;

		int test_row_count = testData.rows();


		int correct_count = 0;
		for (int i = 0; i < test_row_count; ++i) {
			auto data = testData.row(i);

			auto o1 = first.getOutput(data);
			auto o2 = second.getOutput(o1);
			auto o3 = third.getOutput(o2);
			auto op = finalLayer.getOutput(o3);

			if (round(op.sum()) == testResults(i)) ++correct_count;
		}

		cout << "accuracy obtained --- " << ((correct_count / (double)test_row_count) * 100) << " %" << endl;
	}

	void reduceError() {
		auto dataset_raw = extractLabel(getMatrix());
		randomize(dataset_raw);

		auto dataset = train_test_split(dataset_raw);

		const double learning_rate = 0.001;

		NetworkLayer first(4, 20, learning_rate); 
		NetworkLayer second(20, 30, learning_rate); 
		NetworkLayer finalLayer(30, 1, learning_rate);

		// training
		auto trainData = dataset.first.first;
		auto trainResults = dataset.first.second;

		int train_row_count = trainData.rows();

		for (int k = 0; k < 100; ++k) {
			double error_sum = 0;
			for (int i = 0; i < train_row_count; ++i) {
				auto data = trainData.row(i);

				auto o1 = first.getOutput(data);
				auto o2 = second.getOutput(o1);
				auto op = finalLayer.getOutput(o2);


				// calculating error 
				double error = trainResults(i) - op.sum();

				error_sum += abs(error);

				finalLayer.updateWeights(error, o2);
				second.updateWeights(finalLayer, o1); 
				first.updateWeights(second, data);
			}
			error_file << (error_sum / (double)train_row_count) << endl;
		}

		// testing 
		auto testData = dataset.second.first;
		auto testResults = dataset.second.second;

		int test_row_count = testData.rows();

		int correct_count = 0;
		for (int i = 0; i < test_row_count; ++i) {
			auto data = testData.row(i);

			auto o1 = first.getOutput(data);
			auto o2 = second.getOutput(o1); 
			auto op = finalLayer.getOutput(o2);

			if (round(op.sum()) == testResults(i)) ++correct_count; 
		}

		cout << "accuracy obtained --- " << ((correct_count / (double)test_row_count) * 100) << " %" << endl;
	}
};

int main() {

	TransfusionModel model; 
	model.eval(); 
	
	return 0; 
}
