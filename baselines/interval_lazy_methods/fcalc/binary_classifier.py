from . import patterns
from .classifier import FcaClassifier
from . import binary_decision_functions
import numpy as np

class BinarizedBinaryClassifier(FcaClassifier):
    '''
    FCA support based classifier for binary classification on binarized data
 
    Attributes
    ----------
    context : list, numpy.ndarray
        Binarized features of objects with a known class labels
    labels : list, numpy.ndarray
        Labels of the objects
    support : None, numpy.ndarray
        Precomputed support or None
    method : str
        Name of classification method
    alpha : float 
        Hyperparameter of the method
    '''
    def __init__(self, context, labels, support=None, method="standard", alpha=0.):
        '''
        Initializes BinarizedBinaryClassifier object
 
        Parameters
        ----------
        context : list, numpy.ndarray: 
            Features of objects with a known class labels
        labels : list, numpy.ndarray 
            Labels of the objects
        support : None, numpy.ndarray
            Precomputed support or None
        method : str
            Name of classification method
        alpha : float
            Hyperparameter of the method
        '''        
        super().__init__(context, labels, support)
        self.context = self.context.astype(dtype=np.bool_)
        self.method = method
        self.alpha = alpha

    def compute_support(self, test):
        '''
        Computes support for the given test objects
 
        Parameters
        ----------
        test : list, numpy.ndarray
            Test objects description (binarized)
        '''
        test = np.asarray(test, dtype=np.bool_)
        train_pos = self.context[self.labels == True]
        train_neg = self.context[self.labels == False]

        positive_support = np.zeros(shape=(len(test), len(train_pos)))
        positive_counter = np.zeros(shape=(len(test), len(train_pos)))
        negative_support = np.zeros(shape=(len(test), len(train_neg)))
        negative_counter = np.zeros(shape=(len(test), len(train_neg)))

        for i in range(len(test)):
            intsec_pos = test[i].reshape(1, -1) & train_pos
            n_support_pos = ((intsec_pos @ (~train_pos.T)) == 0).sum(axis=1)
            n_counter_pos = ((intsec_pos @ (~train_neg.T)) == 0).sum(axis=1)

            intsec_neg = test[i].reshape(1, -1) & train_neg
            n_support_neg = ((intsec_neg @ (~train_neg.T)) == 0).sum(axis=1)
            n_counter_neg = ((intsec_neg @ (~train_pos.T)) == 0).sum(axis=1)

            positive_support[i] = n_support_pos
            positive_counter[i] = n_counter_pos
            negative_support[i] = n_support_neg
            negative_counter[i] = n_counter_neg
        
        self.support = [np.array((positive_support, positive_counter)), 
                        np.array((negative_support, negative_counter))]

    def predict(self, test):
        '''
        Predicts the class labels for the given test objects
 
        Parameters
        ----------
        test : list, numpy.ndarray
            Test objects description (binarized)
        '''
        if not self.support:
            self.compute_support(test)
        
        self.predictions = np.zeros(len(test))

        if self.method == "standard":
            for i in range(len(test)):
                self.predictions[i] = binary_decision_functions.alpha_weak(self.support[0][:,i], 
                                                                           self.support[1][:,i], 
                                                                           self.alpha)
        elif self.method == "standard-support":
            for i in range(len(test)):
                self.predictions[i] = binary_decision_functions.alpha_weak_support(self.support[0][:,i], 
                                                                                   self.support[1][:,i], 
                                                                                   self.alpha)
        elif self.method == "ratio-support":
            for i in range(len(test)):
                self.predictions[i] = binary_decision_functions.ratio_support(self.support[0][:,i], 
                                                                              self.support[1][:,i], 
                                                                              self.alpha)

class PatternBinaryClassifier(FcaClassifier):
    '''
    FCA support based classifier for binary classification using pattern structures
 
    Attributes
    ----------
    context : list, numpy.ndarray
        Binarized features of objects with a known class labels
    labels : list, numpy.ndarray
        Labels of the objects
    support : None, numpy.ndarray
        Precomputed support or None
    categorical : list
        list of indixes of columns with categorical features
    method : str
        Name of classification method
    alpha : float
        Hyperparameter of the method
    '''
    def __init__(self, context, labels, support=None, categorical=None, method="standard", alpha=0.):
        '''
        Initializes PatternBinaryClassifier object
 
        Parameters
        ----------
        context : list, numpy.ndarray
            Binarized features of objects with a known class labels
        labels : list, numpy.ndarray
            Labels of the objects
        support : None, numpy.ndarray
            Precomputed support or None
        categorical : list
            list of indixes of columns with categorical features
        method : str
            Name of classification method
        alpha : float
            Hyperparameter of the method
        '''      

        super().__init__(context, labels, support)
        self.method = method
        self.alpha = alpha
        if categorical is None:
            self.categorical = []
        else: 
            self.categorical = categorical
    
    def compute_support(self, test):
        '''
        Computes support for the given test objects
 
        Parameters
        ----------
        test : list, numpy.ndarray
            Test objects description
        '''

        train_pos = self.context[self.labels == True]
        train_neg = self.context[self.labels == False]

        positive_support = np.zeros(shape=(len(test), len(train_pos)))
        positive_counter = np.zeros(shape=(len(test), len(train_pos)))
        negative_support = np.zeros(shape=(len(test), len(train_neg)))
        negative_counter = np.zeros(shape=(len(test), len(train_neg)))

        if len(self.categorical) == 0:
            for i in range(len(test)):
                for j in range(len(train_pos)):
                    intsec = patterns.IntervalPattern(test[i],train_pos[j])
                    positive_support[i][j] = sum((~((intsec.low <= train_pos) * (train_pos <= intsec.high))).sum(axis=1) == 0)
                    positive_counter[i][j] = sum((~((intsec.low <= train_neg) * (train_neg <= intsec.high))).sum(axis=1) == 0)
                
                for j in range(len(train_neg)):
                    intsec = patterns.IntervalPattern(test[i],train_neg[j])
                    negative_support[i][j] = sum((~((intsec.low <= train_neg) * (train_neg <= intsec.high))).sum(axis=1) == 0)
                    negative_counter[i][j] = sum((~((intsec.low <= train_pos) * (train_pos <= intsec.high))).sum(axis=1) == 0)

        elif len(self.categorical) == test.shape[1]:
            for i in range(len(test)):
                for j in range(len(train_pos)):
                    intsec = patterns.CategoricalPattern(test[i], train_pos[j])
                    positive_support[i][j] = sum((~(train_pos[:,intsec.mask] == intsec.vals)).sum(axis=1)==0)
                    positive_counter[i][j] = sum((~(train_neg[:,intsec.mask] == intsec.vals)).sum(axis=1)==0)
                
                for j in range(len(train_neg)):
                    intsec = patterns.CategoricalPattern(test[i], train_neg[j])
                    negative_support[i][j] = sum((~(train_neg[:,intsec.mask] == intsec.vals)).sum(axis=1)==0)
                    negative_counter[i][j] = sum((~(train_pos[:,intsec.mask] == intsec.vals)).sum(axis=1)==0)

        else:
            train_pos_cat =  train_pos[:,self.categorical]
            train_pos_num = np.delete(train_pos, self.categorical, axis=1)
            train_neg_cat =  train_neg[:,self.categorical]
            train_neg_num = np.delete(train_neg, self.categorical, axis=1)

            for i in range(len(test)):
                for j in range(len(train_pos)):

                    intsec_cat = patterns.CategoricalPattern(test[i][self.categorical], train_pos_cat[j])
                    intsec_num = patterns.IntervalPattern(np.delete(test[i], self.categorical), train_pos_num[j])
                    
                    positive_support[i][j] = sum(((~((intsec_num.low <= train_pos_num) * (train_pos_num <= intsec_num.high))).sum(axis=1) == 0) * 
                                                 ((~(train_pos_cat[:,intsec_cat.mask] == intsec_cat.vals)).sum(axis=1)==0))
                    positive_counter[i][j] = sum(((~((intsec_num.low <= train_neg_num) * (train_neg_num <= intsec_num.high))).sum(axis=1) == 0) * 
                                                 ((~(train_neg_cat[:,intsec_cat.mask] == intsec_cat.vals)).sum(axis=1)==0))
                    
                for j in range(len(train_neg)):
                    intsec_cat = patterns.CategoricalPattern(test[i][self.categorical], train_neg_cat[j])
                    intsec_num = patterns.IntervalPattern(np.delete(test[i], self.categorical), train_neg_num[j])
                    
                    negative_support[i][j] = sum(((~((intsec_num.low <= train_neg_num) * (train_neg_num <= intsec_num.high))).sum(axis=1) == 0) * 
                                                 ((~(train_neg_cat[:,intsec_cat.mask] == intsec_cat.vals)).sum(axis=1)==0))
                    negative_counter[i][j] = sum(((~((intsec_num.low <= train_pos_num) * (train_pos_num <= intsec_num.high))).sum(axis=1) == 0) * 
                                                 ((~(train_pos_cat[:,intsec_cat.mask] == intsec_cat.vals)).sum(axis=1)==0))
                    
        self.support = [np.array((positive_support, positive_counter)), 
                        np.array((negative_support, negative_counter))]

    def predict(self, test):
        '''
        Predicts the class labels for the given test objects
 
        Parameters
        ----------
        test : list, numpy.ndarray
            Test objects description
        '''

        if not self.support:
            self.compute_support(test)
        
        self.predictions = np.zeros(len(test))

        if self.method == "standard":
            for i in range(len(test)):
                self.predictions[i] = binary_decision_functions.alpha_weak(self.support[0][:,i], 
                                                                            self.support[1][:,i],
                                                                            self.alpha)
        elif self.method == "standard-support":
            for i in range(len(test)):
                self.predictions[i] = binary_decision_functions.alpha_weak_support(self.support[0][:,i], 
                                                                                   self.support[1][:,i], 
                                                                                   self.alpha)
                
        elif self.method == "ratio-support":
            for i in range(len(test)):
                self.predictions[i] = binary_decision_functions.ratio_support(self.support[0][:,i], 
                                                                              self.support[1][:,i], 
                                                                              self.alpha)