import numpy as np
from pboost.feature.factory import BaseFeatureFactory, BaseFeatureFactoryManager

'''
Attributes of pima dataset
0 : Number of times pregnant
1 : Plasma glucose concentration a 2 hours in an oral glucose tolerance test
2 : Diastolic blood pressure (mm Hg)
3 : Triceps skin fold thickness (mm)
4 : 2-Hour serum insulin (mu U/ml)
5 : Body mass index (weight in kg/(height in m)^2)
6 : Diabetes pedigree function
7 : Age (years)
'''


class DefaultFactoryManager(BaseFeatureFactoryManager):

    def produce(self):
        for attr in range(8):
            self.make(attr)


class DefaultFactory(BaseFeatureFactory):

    def blueprint(self, attr):
        return self.data[:, attr]


class CrossFactoryManager(BaseFeatureFactoryManager):

    def produce(self):
        for attr_1 in range(8):
            for attr_2 in range(8):
                self.make(attr_1, attr_2)


class CrossFactory(BaseFeatureFactory):

    def blueprint(self, attr1, attr2):
        return self.data[:, attr1] * self.data[:, attr2]


class EntropyFactoryManager(BaseFeatureFactoryManager):

    def produce(self):
        for attr in range(8):
            self.make(attr)


class EntropyFactory(BaseFeatureFactory):

    def blueprint(self, attr):
        return self.data[:, attr] * np.log(self.data[:, attr])


class RatioFactoryManager(BaseFeatureFactoryManager):

    def produce(self):
        for attr_1 in range(8):
            for attr_2 in range(8):
                self.make(attr_1, attr_2)


class RatioFactory(BaseFeatureFactory):

    def blueprint(self, attr1, attr2):
        return self.data[:, attr1]*1.0 / self.data[:, attr2]
