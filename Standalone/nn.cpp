//
// Created by Miguel on 21/12/2016.
//
#include "nn.h"

NeuralNetwork::NeuralNetwork() {
    Error = 0;
    Inputs = 0;
    Hidden = 0;
    Outputs = 0;
    Target = 0;
    HBias = 0; // hidden bias
    OBias = 0; //output bias
    Delta = 0;
    HDelta = 0;
    DWeights = 0;
    HDWeights = 0;
    InputWeights = 0;
    HiddenWeights = 0;
}

NeuralNetwork::~NeuralNetwork() {
    delete[] Inputs;
    delete[] Hidden;
    delete[] Outputs;
    delete[] Target;

    delete[] HBias;
    delete[] OBias;

    delete[] Delta;
    delete[] HDelta;

    delete[] DWeights;
    delete[] HDWeights;

    delete[] InputWeights;
    delete[] HiddenWeights;
}

// variable initialization and space allocation
void NeuralNetwork::createNet(void) {
    Inputs = new double[inputNum];
    Hidden = new double[hiddenNum];

    Outputs = new double[outputNum];
    Target = new double[targetNum];

    HBias = new double[hiddenNum];
    OBias = new double[outputNum];

    Delta = new double[outputNum];
    HDelta = new double[hiddenNum];

    DWeights = new double[outputNum];
    HDWeights = new double[hiddenNum];

    InputWeights = createLayer(inputNum, hiddenNum);
    HiddenWeights = createLayer(hiddenNum, outputNum);

    randomWeights();
    zeroDeltas();
    randomBias();
}

// Weights initialization
void NeuralNetwork::randomWeights(void) {
    int i, j;

    for (i = 0; i < inputNum; i++) {
        for (j = 0; j < hiddenNum; j++) {
            InputWeights[i][j] = generateRandom(1, -1);
        }
    }

    for (i = 0; i < hiddenNum; i++) {
        for (j = 0; j < outputNum; j++) {
            HiddenWeights[i][j] = generateRandom(1, -1);
        }
    }
}

// bias initialization
void NeuralNetwork::randomBias(void) {
    int i;

    for (i = 0; i < outputNum; i++) {
        OBias[i] = generateRandom(1, 0);
    }

    for (i = 0; i < hiddenNum; i++) {
        HBias[i] = generateRandom(1, 0);
    }
}

void NeuralNetwork::zeroDeltas(void) {
    int i;

    for (i = 0; i < outputNum; i++) {
        DWeights[i] = 0;
    }

    for (i = 0; i < hiddenNum; i++) {
        HDWeights[i] = 0;
    }
}

// transfer function
double NeuralNetwork::sigmoid(double num) {
    return (1.0 / (1.0 + exp(-num)));
}

void NeuralNetwork::feedForward(void) {
    int i, j;
    double synapseSum = 0.0f; // weights * inputs

    // feed hidden layer
    for (i = 0; i < hiddenNum; i++) {
        for (j = 0; j < inputNum; j++) {
            synapseSum += InputWeights[j][i] * Inputs[j];
        }
        Hidden[i] = sigmoid(synapseSum + HBias[i]);

        synapseSum = 0.0f;
    }

    // feed output layer
    for (i = 0; i < outputNum; i++) {
        for (j = 0; j < hiddenNum; j++) {
            synapseSum += HiddenWeights[j][i] * Hidden[j];
        }
        Outputs[i] = sigmoid(synapseSum + OBias[i]);

        synapseSum = 0.0f;
    }
}

void NeuralNetwork::computeError(void) {
    int i, j;
    double errorSum = 0.0f;
    double sumSquaredVals = regSumSquaredVals();
    Error = 0;

    // output layer error
    for (i = 0; i < outputNum; i++) {
        Err = (Target[i] - Outputs[i]);
        Delta[i] = (1 - Outputs[i]) * Outputs[i] * Err; // partial derivative of the total error with respect to the net input of the neuron
        Error += (0.5f * Err * Err); // error
    }

    // hidden layer error
    for (i = 0; i < hiddenNum; i++) {
        for (j = 0; j < outputNum; j++) {
            errorSum += Delta[j] * HiddenWeights[i][j];
        }
        HDelta[i] = (1 - Hidden[i]) * Hidden[i] * errorSum;

        errorSum = 0.0f;
    }
}

void NeuralNetwork::backpropagate(void) {
    int i, j;

    // output layer
    for (i = 0; i < hiddenNum; i++) {
        for (j = 0; j < outputNum; j++) {
            HiddenWeights[i][j] += LR * ((Delta[j] * Hidden[i]) + (LambdaHO * Hidden[i])) + Alpha * DWeights[j]; // backprop with momentum
            DWeights[j] = LR * Delta[j] * Hidden[i];
        }
    }

    // hidden layer
    for (i = 0; i < inputNum; i++) {
        for (j = 0; j < hiddenNum; j++) {
            InputWeights[i][j] += LR * ((HDelta[j] * Inputs[i]) + (LambdaIH * Inputs[i])) + Alpha * HDWeights[j];
            HDWeights[j] = LR * HDelta[j] * Inputs[i];
        }
    }

    // output bias
    for (i = 0; i < outputNum; i++) {
        OBias[i] += LR * Delta[i];
    }

    // hidden bias
    for (i = 0; i < hiddenNum; i++) {
        HBias[i] += LR * HDelta[i];
    }
}

double NeuralNetwork::regSumSquaredVals() {
  int i, j;
  double sumSquaredVals = 0.0;

  // output layer
  for (i = 0; i < hiddenNum; i++) {
      for (j = 0; j < outputNum; j++) {
          sumSquaredVals += (HiddenWeights[i][j] * HiddenWeights[i][j]);

      }
  }

  // hidden layer
  for (i = 0; i < inputNum; i++) {
      for (j = 0; j < hiddenNum; j++) {
          sumSquaredVals += (InputWeights[i][j] * InputWeights[i][j]);
      }
  }

  return sumSquaredVals;
}

// weights array creator
double **createLayer(int Row, int Col) {
    double **array = new double *[Row];
    for (int i = 0; i < Row; i++)
        array[i] = new double[Col];

    return array;
}

double generateRandom(int High, int Low) {
    srand((unsigned int) time(NULL));
    return ((double) rand() / RAND_MAX) * (High - Low) + Low;
}

void NeuralNetwork::trainNet(void) {
    feedForward();
    computeError();
    backpropagate();
}

void NeuralNetwork::testNet(void) {
    feedForward();
}

void NeuralNetwork::saveNet(char *p) {
    FILE *fw = fopen(p, "w");

    if (!fw) {
        perror(p);
        return;
    }

    int i, j;

    fprintf(fw, "%d\n", inputNum);
    fprintf(fw, "%d\n", hiddenNum);
    fprintf(fw, "%d\n", outputNum);
    fprintf(fw, "%d\n", targetNum);

    // save momentum and LR
    fprintf(fw, "%lf\n", Alpha);
    fprintf(fw, "%lf", LR);
    fprintf(fw, "\n");
    // Save bias output layer
    for (i = 0; i < outputNum; i++) {
        fprintf(fw, "%lf  ", OBias[i]);
    }

    fprintf(fw, "\n\n");

    // save bias hidden layer
    for (i = 0; i < hiddenNum; i++) {
        fprintf(fw, "%lf  ", HBias[i]);
    }

    fprintf(fw, "\n\n");

    // save input weights
    for (i = 0; i < inputNum; i++) {
        for (j = 0; j < hiddenNum; j++) {
            fprintf(fw, "%lf  ", InputWeights[i][j]);
        }
    }

    fprintf(fw, "\n\n\n");

    // save hidden weights
    for (i = 0; i < hiddenNum; i++) {
        for (j = 0; j < outputNum; j++) {
            fprintf(fw, "%lf  ", HiddenWeights[i][j]);
        }
    }

    fprintf(fw, "\n\n");

    // save deltas
    for (i = 0; i < outputNum; i++) {
        fprintf(fw, "%lf  ", DWeights[i]);
    }

    fprintf(fw, "\n\n");

    for (i = 0; i < hiddenNum; i++) {
        fprintf(fw, "%lf  ", HDWeights[i]);
    }

    fflush(fw);
    fclose(fw);
}

void NeuralNetwork::loadNet(char *p) {
    FILE *fw = fopen(p, "r");

    if (!fw) {
        perror(p);
        return;
    }

    int i, j;

    fscanf(fw, "%d", &inputNum);
    fscanf(fw, "%d", &hiddenNum);
    fscanf(fw, "%d", &outputNum);
    fscanf(fw, "%d", &targetNum);

    createNet();

    fscanf(fw, "%lf\n\n", &Alpha);
    fscanf(fw, "%lf", &LR);

    for (i = 0; i < outputNum; i++) {
        fscanf(fw, "%lf", &OBias[i]);
    }
    for (i = 0; i < hiddenNum; i++) {
        fscanf(fw, "%lf", &HBias[i]);
    }

    for (i = 0; i < inputNum; i++) {
        for (j = 0; j < hiddenNum; j++) {
            fscanf(fw, "%lf", &InputWeights[i][j]);
        }
    }

    for (i = 0; i < hiddenNum; i++) {
        for (j = 0; j < outputNum; j++) {
            fscanf(fw, "%lf", &HiddenWeights[i][j]);
        }
    }

    for (i = 0; i < outputNum; i++) {
        fscanf(fw, "%lf  ", &DWeights[i]);
    }

    for (i = 0; i < hiddenNum; i++) {
        fscanf(fw, "%lf  ", &HDWeights[i]);
    }

    fclose(fw);
}

void NeuralNetwork::loadBot(int input, int hidden, int out, double alfa, double learnR,
                            double OBi[], double HBi[],
                            double inputW[], double hiddenW[],
                            double DWeigH[], double HDWeigh[]) {

  inputNum = input;
  hiddenNum = hidden;
  outputNum = out;
  targetNum = out;
  createNet();

  Alpha = alfa;
  LR = learnR;
  int i, j, k;
  k = 0;

  for (i = 0; i < outputNum; i++) {
      OBias[i] = OBi[i];
  }

  for (i = 0; i < hiddenNum; i++) {
      HBias[i] = HBi[i];

  }

  for (i = 0; i < inputNum; i++) {
      for (j = 0; j < hiddenNum; j++) {
          InputWeights[i][j] = inputW[k];
          k++;
      }
  }

  k = 0;

  for (i = 0; i < hiddenNum; i++) {
      for (j = 0; j < outputNum; j++) {
          HiddenWeights[i][j] = hiddenW[k];
          k++;
      }
  }

  for (i = 0; i < outputNum; i++) {
      DWeights[i] = DWeigH[i];
  }

  for (i = 0; i < hiddenNum; i++) {
      HDWeights[i] = HDWeigh[i];
  }
}
