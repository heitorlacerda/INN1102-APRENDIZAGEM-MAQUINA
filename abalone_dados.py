from numpy import double
import numpy as np

class DataMgr:
    """
    Class responsible for importing right data from text bases.
    """
    
    def __init__(self):
        with open("../datasets/mfeat-fac.txt") as fac_f:
            self.dataSet_fac = []
            for line in fac_f:
                self.dataSet_fac.append(list(map(int, line.split()))) 
                           
        print("Imported fac")   
        
        with open("../datasets/mfeat-fou.txt") as fou_f:
            self.dataSet_fou = []
            for line in fou_f:
                self.dataSet_fou.append(list(map(double, line.split()))) 
                               
            print("Imported fou")  
            
        with open("../datasets/mfeat-kar.txt") as kar_f:
            self.dataSet_kar = []
            for line in kar_f:
                self.dataSet_kar.append(list(map(double, line.split()))) 
                               
            print("Imported kar")   
            
        with open("../datasets/labels.txt") as labels_f:
            self.dataSet_labels = []
            for line in labels_f:
                self.dataSet_labels.append(int(line.split()[0]))
                               
            print("Imported labels")  
    
    def rawData(self):
        """
        
        """
            
        return np.array(self.dataSet_labels), [np.array(self.dataSet_fac), np.array(self.dataSet_fou), np.array(self.dataSet_kar)]
