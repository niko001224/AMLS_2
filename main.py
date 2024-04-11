from A.vit_run import train
from A.test import test


acc_A_train = train() # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A_test = test()   # Test model based on the test set.

# ======================================================================================================================
## Print out your results with following format:
print('TA:{},{};'.format(acc_A_train, acc_A_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'