# MLP-NeuralNetwork
Standalone version of my neural network for the <a href="https://github.com/msanpe/Machine-Learning-Arcade-Learning-Environment">Arcade Learning Environment machine learning project</a>

**General Information**

**Hyperparameter tuning:**

The network has Learning Rate, Momentum and Regularization Strength. All of them can be adjusted from the functions.cpp file.

**Training options:**

Training can be adjusted to a target error or by number of epochs.

**Data structuring:**

Data normalization is done from the data class once all the dataset is loaded from the file.

Data file structure:

    Number of training samples
    Input11,Input12,Input1N,Output11,Output12,Output1N
    Input21,Input22,Input2N,Output21,Output22,Output2N
    InputN1,InputN2,InputNN,OutputN1,OutputN2,OutputNN
    
File example for an XOR gate:

    4
    0,0,1
    0,1,0
    1,0,0
    1,1,1 

**To compile:**

    g++ main.cpp functions.cpp data.cpp nn.cpp -o net
**To run:**
    
    ./net   
**Use:**

    1. Train new network
    2. Switch between continuos and discrete outputs
    3. Test network
    4. Load network
    5. Exit
    
    1. To train a new network place the inputs.txt in the /data folder, after the training the weights will be saved in the
    same folder as the executable.
    2. This option applies the threshold to convert between discrete and continuous outputs.
    3. Test the network by manually entering the inputs.
    4. Loads networks weights and architecture from a previous training.
