#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <random>
#include <iomanip>

using namespace std;
const double e = 2.718281828459045, momentum = 0.8;
double learnrate = 0.001;
const int batch_size = 100, smpl = 10000, a2_neurons = 50, test_smpl = 100;
const double beta1 = 0.9, beta2 = 0.999;
double t = 0;
double correct = 0, cost = 0;
vector<vector<double>>a, a2, a3, w2, w3, m2, m3, v2, v3;
vector<int>labels(smpl + test_smpl), nnanswer(batch_size);
void set(int images), read_images(int smpls), read_labels(int smpls), nn(int batch), bp(int batch), record(int batch), test(int n), print_all();
void clean(), print(int batch), shuffle();
default_random_engine setran(time(NULL));
normal_distribution<double> distribution_w2(0.0, sqrt(2.0/784));
normal_distribution<double> distribution_w3(0.0, sqrt(1.0 / a2_neurons));
uniform_int_distribution<int>sh(0, smpl);

int main()
{
	set(smpl + test_smpl);
	for (int epoch = 0; epoch < 6; epoch++) {
		for (int batc = 0; batc < smpl / batch_size; batc++) {
			nn(batc);
			bp(batc);
			record(batc);
			print(batc);
			clean();
		}
		test(test_smpl);
		shuffle();
	}
	print_all();
    return 0;
}

void set(int images) {
	read_labels(images);
	read_images(images);
	srand(time(NULL));
	//a2
	vector<double>temp(a2_neurons);
	for (int i = 0; i < batch_size; i++) {
		a2.push_back(temp);
	}
	//a3
	vector<double>temp2(10);
	for (int i = 0; i < batch_size; i++) {
		a3.push_back(temp2);
	}
	//w2
	for (int i = 0; i < a2_neurons; i++) {
		vector<double>temp3;
		for (int j = 0; j < 784; j++)
			temp3.push_back(distribution_w2(setran));
		w2.push_back(temp3);
	}
	//w3
	for (int i = 0; i < 10; i++) {
		vector<double>temp4;
		for (int j = 0; j < a2_neurons; j++)
			temp4.push_back(distribution_w3(setran));
		w3.push_back(temp4);
	}
	//v2
	vector<double>temp5;
	for (int i = 0; i < 784; i++)temp5.push_back(0);
	for (int i = 0; i < a2_neurons; i++) { 
		v2.push_back(temp5);
		m2.push_back(temp5);
	}
	//v3
	vector<double>temp6;
	for (int i = 0; i < a2_neurons; i++)temp6.push_back(0);
	for (int i = 0; i < 10; i++) {
		v3.push_back(temp6);
		m3.push_back(temp6);
	}
	cout << "set" << endl << endl;
}

void shuffle() {
	int s;
	for (int i = 0; i < smpl; i++) {
		s = sh(setran);
		swap(a[i], a[s]);
		swap(labels[i], labels[s]);
	}
}

void nn(int batch){
#pragma omp parallel for//parallel 
	for (int sampl = 0; sampl < batch_size; sampl++) {
		//first layer
		int pic = batch * batch_size + sampl;
		for (int i = 0; i < a2_neurons; i++) {
			double temp = 0;
			for (int j = 0; j < 784; j++)
				temp += w2[i][j] * a[pic][j];
			a2[sampl][i] = (temp > 0 ? temp : 0);
		}
		//second layer

		//multiply
		for (int i = 0; i < 10; i++)
			for (int j = 0; j < a2_neurons; j++)
				a3[sampl][i] += a2[sampl][j] * w3[i][j];

		//softmax
		double a3max_value = a3[sampl][0], a3sum = 0;
		nnanswer[sampl] = 0;
		for (int i = 1; i < 10; i++) {
			if (a3[sampl][i] > a3max_value) {
				a3max_value = a3[sampl][i];
				nnanswer[sampl] = i;
			}
		}

		for (int i = 0; i < 10; i++) {
			a3[sampl][i] = pow(e, a3[sampl][i] - a3max_value);
			a3sum += a3[sampl][i];
		}
		for (int i = 0; i < 10; i++)
			a3[sampl][i] /= a3sum + 0.00000001;
	}
}

void bp(int batch) {
	t++;
	//dw3
#pragma omp parallel for
	for (int i = 0; i < 10; i++) {
		double m, v;
		for (int j = 0; j < a2_neurons; j++) {
			double temp = 0;
			for (int k = 0; k < batch_size; k++)
				if (i == labels[batch * batch_size + k])
					temp += a2[k][j] * (a3[k][i] - 1);
				else
					temp += a2[k][j] * a3[k][i];

			//adam
			m3[i][j] = beta1 * m3[i][j] + (1 - beta1)*temp;
			v3[i][j] = beta2 * v3[i][j] + (1 - beta2)*temp*temp;
			m = m3[i][j] / (1 - pow(beta1, t));
			v = v3[i][j] / (1 - pow(beta2, t));
			w3[i][j] -= learnrate * m / (sqrt(v) + 0.00000001);
		}
	}
	//dw2
#pragma omp parallel for
	for (int i = 0; i < a2_neurons; i++) {
		double m, v, temp = 0, dada = 0;
		for (int j = 0; j < 784; j++) {
			temp = 0;
			for (int k = 0; k < batch_size; k++) {
				dada = 0;
				int answer = labels[batch*batch_size + k];
				for (int l = 0; l < 10; l++)
					if (l == answer)
						dada += (a3[k][answer] - 1) * w3[l][i];
					else
						dada += a3[k][l] * w3[l][i];
				temp += dada * (bool)a2[k][i] * a[batch*batch_size + k][j];
			}

			//adam
			m2[i][j] = beta1 * m2[i][j] + (1 - beta1)*temp;
			v2[i][j] = beta2 * v2[i][j] + (1 - beta2)*temp*temp;
			m = m2[i][j] / (1 - pow(beta1, t));
			v = v2[i][j] / (1 - pow(beta2, t));
			w2[i][j] -= learnrate * m / (sqrt(v) + 0.00000001);
		}
	}
}

void test(int n) {
	for (int i = smpl / batch_size; i < (smpl + test_smpl) / batch_size; i++) {
		nn(i);
		record(i);
	}
	cout << "Test_accuracy: " << correct / (test_smpl) << " Test_cost: " << cost / (test_smpl) << endl;
	clean();
	cost = 0;
	correct = 0;
}

void print_all() {
	cout << fixed << setprecision(7);
	//w2
	cout << "w2 :" << endl;
	for (int i = 0; i < a2_neurons; i++) {
		for (int j = 0; j < 784; j++) {
			cout << w2[i][j] << ",";
			if (j % 28 == 27)
				cout << endl;
		}
		cout << endl;
	}
	//w3
	cout << "w3 :" << endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < a2_neurons; j++) {
			cout << w3[i][j] << ",";
		}
		cout << endl;
	}
}

void record(int batch) {
	for (int i = 0; i < batch_size; i++) {
		correct += (nnanswer[i] == labels[batch*batch_size + i] ? 1 : 0);
		cost -= log(a3[i][labels[batch*batch_size + i]] + 0.00000001);
	}
}

void print(int batch) {
	if ((batch + 1) % (smpl / batch_size / 10) == 1)
		cout << batch * batch_size + 1 << "-" << (batch + smpl / batch_size / 10)*batch_size << ": {";
	if ((batch + 1) % (max(smpl,10000) / batch_size / 10 / 10) == 0)
		cout << "==";
	if ((batch + 1) % (smpl / batch_size / 10) == 0) {
		cout << "} Accuracy: " << correct / (smpl / 10) << " Cost: " << cost / (smpl / 10) << endl;
		correct = 0;
		cost = 0;
	}
}

void clean() {
	a2.clear();
	vector<double>temp(a2_neurons);
	for (int i = 0; i < batch_size; i++) {
		a2.push_back(temp);
	}
	a3.clear();
	vector<double>temp2(10);
	for (int i = 0; i < batch_size; i++) {
		a3.push_back(temp2);
	}
}

void read_labels(int smpls) {
	ifstream label("train-labels.idx1-ubyte", ios::binary | ios::in);
	label.ignore(8);
	for (int i = 0; i < smpls; i++) {
		unsigned char ch;
		label.read((char*)&ch, 1);
		labels[i] = (int)ch;
	}
}

void read_images(int smpls) {
	ifstream image("train-images.idx3-ubyte", ios::binary | ios::in);
	image.ignore(16);
	for (int i = 0; i < smpls; i++) {
		vector<double>imag;
		for (int row = 0; row < 28; row++) {
			for (int col = 0; col < 28; col++) {
				unsigned char ch;
				image.read((char*)&ch, 1);
				imag.push_back((double)ch / 255);
			}
		}
		a.push_back(imag);
	}
}
