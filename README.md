# AMLS_2
Cassava Leaf Disease Classification 
The main function invokes two functions from folder A:  

1、The train() function is a fully qualified path to the training model, sourced from the vit_run.py file.  

This function incorporates three classes:  
class EarlyStopping  :                 provides functionality for early stopping.  

class LabelSmoothingCrossEntropy    : defines a loss function with label smoothing.  

class CustomImageDataset         :     establishes the connection between image folders and labels.  


2、The test() function is a fully qualified path to the testing model, sourced from the test.py file.   

This function includes one class:class CustomTTADataset                establishes the connection between image folders and labels.  


library functions  

torchvision.io  

pandas  

os  

torchvision.transforms  

torchvision   

torch  

matplotlib.pyplot  

math   

torch.optim.lr_scheduler
