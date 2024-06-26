
# DENL: Diverse Ensemble and Noisy Logits

**Paper:** 
Mina Yazdani, Hamed Karimi, and Reza Samavi. "*DENL: Diverse Ensemble and Noisy Logits for Improved Robustness of Neural Networks*." Accepted in The 15th Asian Conference on Machine Learning (ACML 2023).


## Getting Started


### Prerequisites
The following python modules are required to run the code:
- Numpy (general array manipulations and utilities)
- Tensorflow 2.8.0
- Keras (neural network models)
- Matplotlib (graphing utilities)

### Running the experiments
Run the script ***main_of_mains_cross_validation.py*** for both training time and inference time.

### Files Descriptions
- ***main_of_mains_cross_validation.py***: This code file is the main code to run our proposed method and produce the results. To run the code, you need to 
set/use the flags to set up the details of the experiments. This code uses the following Python files:

- ***train_models_ensemble.py***: This code file contains the functions required for phase 1 training. 

- ***ensemble-training_utils.py***: This code file contains the functions required for phase 2 training.
 
- ***l2_attack.py***: This code file is to attack a network optimizing for $l_2$ distance.

- ***setup_cifar.py***, ***setup_mnist.py***: These code files are to prepare the datasets and CNN models for the experiments.

- ***test_ensemble_func.py***: This code file generates adversarial examples with two attack scenarios (single attack and superimposition attack).

- ***process_results.py***: This code file processes the results of the experiments.



