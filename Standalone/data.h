//
// Created by Miguel on 21/12/2016.
//

#ifndef NEURAL_NET_DATA_H
#define NEURAL_NET_DATA_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>

//Leer un archivo con los patrones de entrada y salida usados por la red
class data {
private:
    int dataSize,
            inp,
            out;

    double **Inputs, //Contenedores de los datos de la muestras
            **Outputs;

    double minVal, maxVal;
    std::string file;
    void normalizeData();
public:
    data();
    void setData(int a, int b, std::string p);
    ~data();
    void readFile(void);
    void getTrainingData(int n, double *p, double *s);
    void createDataContainers(void);
    double getMinVal();
    double getMaxVal();
    int setSize(void);
};

double **createContainer(int Row, int Col);

#endif //NEURAL_NET_DATA_H
