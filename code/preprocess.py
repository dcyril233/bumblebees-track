from sklearn.preprocessing import StandardScaler


class Preprocess:
    
    
    # create object storing feature
    def __init__(self, training_data):
        self.scaler = StandardScaler().fit(training_data)


    # normalize data
    def normalization(self, data):
        normalized_feature = self.scaler.transform(data)
        return normalized_feature