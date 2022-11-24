#include "Utils.h"

void printVector(vector<double> v) {
	for (auto i : v) {
		cout << i << ' ';
	}
	cout << endl;
}

bool getRandomBool() {
	int randVal = rand();
	if (randVal > 10) return false;
	else return true;
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