#include <iostream>
#include "functions.h"


using namespace std;

int c = 0; // control del loop

int main(int argc, char *argv[]) {
    while (c != 1) {
        c = menu();
    }
}
