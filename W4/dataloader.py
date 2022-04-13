# data loader for the dataset
# Author: Li Wei
# Date: 2022/04/12


import pandas


class DataLoader:
    def __init__(self, data_path):
        '''
        :param data_path: the path of the data
        '''
        self.data_path = data_path
        self.data = self._load_data()

    def _load_data(self):
        '''
        :return: the data
        '''
        return pandas.read_csv(self.data_path, header=None,index_col=None)

    def _get_data(self,columns):
        '''
        get the data colums according to the columns index list
        :param columns: the index list of the columns
        :return: the data as a list type
        '''
        return self.data[columns].values.tolist()
    
    def _get_col_count(self):
        '''
        get the column count of the data
        :return: the column count
        '''
        return self.data.shape[1]

    def data_split(self,train_number = 200):
        '''
        split the data into train set and test set
        :param train_number: the number of train set
        :return: the train set and test set
        '''
        features = self._get_data(range(0,self._get_col_count()-1))
        labels = self._get_data(range(self._get_col_count()-1,self._get_col_count()))
        # labels should be translated as one-hot type, the label 'g' as 1, 'b' as 0
        labels = [[1 if i == 'g' else 0 for i in j] for j in labels]
        train_features = features[:train_number]
        train_labels = labels[:train_number]

        test_features = features[train_number:]
        test_labels = labels[train_number:]
        
        train_set = (train_features,train_labels)
        test_set = (test_features,test_labels)
        return train_set,test_set

    def get_row_count(self):
        '''
        get the row count of the data
        :return: the row count
        '''
        return self.data.shape[0]

if __name__=="__main__":
    '''
    test the data loader
    '''
    data_path = 'data\Ionosphere+Dataset.csv'
    data_loader = DataLoader(data_path)
    data = data_loader.get_data([1,2])
    cols_count  = data_loader.get_col_count()
    rows_count = data_loader.get_row_count()
    print(cols_count,rows_count)
    print(data)