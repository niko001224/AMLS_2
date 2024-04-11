from A.vit_run import train
from A.test import test


acc_A_train = train() 
acc_A_test = test()   


print('TA:{},{};'.format(acc_A_train, acc_A_test))

