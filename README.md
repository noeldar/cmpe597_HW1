# cmpe597_HW1

#### Part a ####

* NN.py is the script for train/evaluate implemented CNN.
  * Gradient updates were done with batch gradient descent (batch size = 64, lr = 0.001, epoch = 10)
  * Train/Test datasets' sizes are 1024x28x28 (train/test sizes should be divisible by batch size)
  * Train/prediction are done with batches
  * Can be run by simply **python NN.py** or just open the whole project in an IDE and run NN.py
* Cmpe597_HW1_sanitycheck.ipynb is the script that contains implementation of desired CNN in PyTorch
* There is only 0.07 test accuracy difference between implemented CNN and PyTorch CNN
* NN-Eval-PreTrained.py can be used for upload trained weights and make an evaluation on test split
* NN-Eval-PreTrained.py can be run by simply **python NN-Eval-PreTrained.py** or just open the whole project in an IDE and run NN-Eval-PreTrained.py

#### Part b ####
* Cmpe597_HW1_part_b.ipynb is the script that is used for plotting decision boundaries


* Whole jupyter notebooks can be run by running all cells in order
          
