from numpy import zeros
from pboost.feature.factory import BaseFeatureFactory

class BMI_Exponential(BaseFeatureFactory):
    
    def blueprint(self,height,weight,power):
        return pow(self.data[:,weight],power)+pow(self.data[:,height],power)
    
    def produce(self):
        height = 1
        weight = 2
        power = 2
        self.make(height,weight,power)
        
class BGL_Time_Period_Av(BaseFeatureFactory):
    
    def blueprint(self,bgl_list):
        s = zeros(self.data.shape[0])
        for bgl in bgl_list:
            s = s + self.data[:,bgl]
        return s/len(bgl_list)
    
    def produce(self):
        bgl_m = list()
        bgl_n = list()
        bgl_e = list()
        for k in range(1,(self.data.shape[1]-4)/3+1):
            bgl_m.append(3*k)
            bgl_n.append(3*k+1)
            bgl_e.append(3*k+2)
        self.make(bgl_m)  # average of all morning BGL levels
        self.make(bgl_n)  # average of all noon BGL levels
        self.make(bgl_e)  # average of all evening BGL levels

class BGL_Day_Av(BaseFeatureFactory):
    
    def blueprint(self,bgl_m,bgl_n,bgl_e):
        s = self.data[:,bgl_m] + self.data[:,bgl_n] + self.data[:,bgl_e]
        return s/3
    
    def produce(self):
        bgl_m = list()
        bgl_n = list()
        bgl_e = list()
        for k in range(1,(self.data.shape[1]-4)/3+1):
            bgl_m = 3*k
            bgl_n = 3*k+1
            bgl_e = 3*k+2
            self.make(bgl_m,bgl_n,bgl_e)
            self.make(bgl_m,bgl_n,bgl_e)  # average of all BGL levels within that day

class BMI(BaseFeatureFactory):
    
    def blueprint(self,height,weight):
        return self.data[:,weight]/pow(self.data[:,height],2)
    
    def produce(self):
        height = 1
        weight = 2
        self.make(height,weight)
