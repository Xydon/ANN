#include "Aggregrator.h"

double Aggregrator::aggregrate(DOUBLE pattern, DOUBLE weight) {
	if (pattern.size() != weight.size())
		throw invalid_argument("pattern and weight size must match to aggregrate");

	size_t n = pattern.size();

	double aggregrate_value = 0;
	for (int i = 0; i < n; ++i) {
		aggregrate_value += pattern.at(i) * weight.at(i);
	}

	return aggregrate_value;
}