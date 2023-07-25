import sklearn.preprocessing as skl_pre


class Scaler(object):
    def __init__(self, name):
        self.name = name
        if self.name == 'None':
            self.base_scaler = None
        elif self.name == 'StandardScaler':
            self.base_scaler = skl_pre.StandardScaler()
        elif self.name == 'MinMaxScaler':
            self.base_scaler = skl_pre.MinMaxScaler()
        elif self.name == 'MinMaxScalerPlus1':
            self.base_scaler = skl_pre.MinMaxScaler()
        else:
            raise Exception('Scaler not recognized!')
        return

    def fit(self, x):
        if self.name != 'None':
            self.base_scaler.fit(x)
        return

    def transform(self, x):
        if self.name == 'None':
            return x
        x_scaled = self.base_scaler.transform(x)
        if self.name == 'MinMaxScalerPlus1':
            return x_scaled + 1.
        else:
            return x_scaled

    def inverse_transform(self, x):
        if self.name == 'None':
            return x
        if self.name == 'MinMaxScalerPlus1':
            x_scaled = x - 1.
        else:
            x_scaled = x
        return self.base_scaler.inverse_transform(x_scaled)
