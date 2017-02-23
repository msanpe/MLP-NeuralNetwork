//
// Created by Miguel on 21/12/2016.
//

#include "data.h"

data::data(){
  inp = 0;       //Numero de entradas de la red
  out = 0;      //Numero de salidas de la red;
  file = "";
}

void data::setData(int a, int b, std::string p) {
    inp = a;       //Numero de entradas de la red
    out = b;      //Numero de salidas de la red;
    file = p;
}

data::~data() {
    delete[] Inputs;
    delete[] Outputs;
}

// lee valores de fichero y los almacena hasta que son pedidos
void data::readFile(void) {
  int j = 0;
  std::ifstream in(file.c_str());
  std::string str;
  std::string substr;

  if(!in) {
    std::cout << "error reading file" << std::endl;
    return;
  }

  std::getline(in, str);
  dataSize = atoi(str.c_str());
  createDataContainers();

  for (int i = 0; i < dataSize; i++) {
    std::getline(in, str);
    std::stringstream ss(str);
    for (int j = 0; j < inp; ++j) {
      std::getline( ss, substr, ',' );
      Inputs[j][i] = atof(substr.c_str());
    }
    str = str.substr(inp * 2);
    for (int j = 0; j < out; ++j) {
      std::string substr;
      std::getline( ss, substr, ',' );
      Outputs[j][i] = atof(substr.c_str());
    }
  }
  normalizeData();
}

void data::getTrainingData(int n, double *p, double *s) { // fill containers with training data
    for (int i = 0; i < inp; i++) {
        p[i] = Inputs[i][n];
    }
    for (int i = 0; i < out; i++) {
        s[i] = Outputs[i][n];
    }
}

void data::normalizeData() {
  minVal = maxVal = 0;

  for (int i = 0; i < dataSize; i++) {
    for (int j = 0; j < inp; j++) {
      if (Inputs[j][i] < minVal)
        minVal = Inputs[j][i];
      if (Inputs[j][i] > maxVal)
        maxVal = Inputs[j][i];
    }
  }

  for (int i = 0; i < dataSize; i++) {
    for (int j = 0; j < inp; j++) {
      Inputs[j][i] = (Inputs[j][i] - minVal)/(maxVal- minVal);
    }
  }
}

void data::createDataContainers(void) {
    Inputs = createContainer(inp, dataSize);    // inputs
    Outputs = createContainer(out, dataSize);  // targets
}

int data::setSize(void) {
    return dataSize;
}

double data::getMinVal() {
  return minVal;
}

double data::getMaxVal() {
  return maxVal;
}

double **createContainer(int Row, int Col) {
    double **array = new double *[Row];
    for (int i = 0; i < Row; i++)
        array[i] = new double[Col];
    return array;
}
