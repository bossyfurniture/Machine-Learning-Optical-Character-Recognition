# Read-Me for Project 1
*Group Name: ML*

*Group Members: Jason Gibson, Abbas Furniturewalla, Marquita Ali, Omkar Mulekar*

## How to Run:
1) The function to call in your script is **TestModel()** located in *test.py*.  Import it to your script using "from **test** import **TestModel**". Restart the kernel before the first time you use this line.
2) The function uses 2 arguments:
	+ filename: path to the pickle file containing the python list of 2D numpy arrays.
	+ model: use "ab" to use the model trained on a's and b's, use "all" to use the model trained on all letters
3) The function has 1 output: the model predictions on the input data.
4) An example can be seen in *example.py*

## Other Details
### example_test.py
This script is simply an example for the TAs to follow when using our code.
```python
# Import dependancy
from test import TestModel

# Examples of calling the function
ab = TestModel('train_data.pkl','ab')
al = TestModel('train_data.pkl','all')

# Print outputs
print(ab)
print(al)
```

### train.py
This script trains two models and outputs them to .sav files.  One model is for identifying a's and b's, and one model is for identifying all letters.  The models in the output .sav files are what the **TestModel()** function in *test.py* use.

### RUN_TrainModel.py
This script runs the **TestModel()** function in *train.py*.

### test.py
This script contains the **TestModel()** function.