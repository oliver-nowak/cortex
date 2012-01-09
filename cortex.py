from scipy.io import loadmat
import numpy as np


def sigmoid(matrix):
    print '[+] calculating sigmoid for matrix...'
    return 1.0 / ( 1.0 + np.exp(-matrix) ) 

def predict_labels(Theta1, Theta2, X):
    print '[+] predicting labels...'
    bias_column = np.ones( (len(X), 1) )
    print '[+] created bias column : ', bias_column.shape
    print '[+] X shape : ', X.shape

    X = np.hstack( (bias_column, X) )
    print '[+] adding bias_column to data...', X.shape

    hidden_layer = np.dot(X, np.transpose(Theta1))
    print '[+] hidden_layer.shape : ', hidden_layer.shape

    sigmoid_hidden_layer = sigmoid(hidden_layer)    
    print '[+] sigmoid_hidden_layer.shape : ', sigmoid_hidden_layer.shape
    
    bias_column = np.ones( (len(hidden_layer), 1) )
    print '[+] created bias column for hidden layer...', bias_column.shape

    sigmoid_hidden_layer = np.hstack( (bias_column, sigmoid_hidden_layer) )
    print '[+] sigmoid hidden layer shape : ', sigmoid_hidden_layer.shape
    
    output_layer = np.dot(sigmoid_hidden_layer, np.transpose(Theta2))
    print '[+] output layer.shape : ', output_layer.shape

    sigmoid_output_layer = sigmoid(output_layer)
    print '[+] sigmoid output layer shape : ', sigmoid_output_layer.shape

    # return p vector containing label from 1...K
    p = sigmoid_output_layer.argmax(axis=1)
    
    # matlab labels are from 1...K, but p is from 0...K-1, so add 1 to all labels
    p = p + 1 

    return p
    

def load_weights():
    print '[+] loading matlab-based precomputed weights...'
    data_file = '/Users/nowak/Development/github/cortex/data/mnist_weights.mat'
    data = loadmat(data_file, matlab_compatible=True)
    return data['Theta1'].squeeze(), data['Theta2'].squeeze()

def load_training_data():
    print '[+] loading matlab-based training data...'
    data_file = '/Users/nowak/Development/github/cortex/data/mnist_data.mat'
    data = loadmat(data_file, matlab_compatible=True)
    
    return data['y'].squeeze(), data['X'].squeeze()

def main():
    y_data, X_data = load_training_data()
    
    y = np.array(y_data)
    print '[+] y.shape : ', y.shape
    
    X = np.array(X_data)
    print '[+] X.shape : ', X.shape
    
    m = len(X)
    print '[+] num of training data : ', len(X)

    Theta1_data, Theta2_data = load_weights()
    
    Theta1 = np.array(Theta1_data)
    print '[+] Theta1.shape : ', Theta1.shape

    Theta2 = np.array(Theta2_data)
    print '[+] Theta2.shape : ', Theta2.shape

    pred_data = predict_labels(Theta1, Theta2, X)
    
    count = 0
    for i in range(y.shape[0]):
        if y[i] == pred_data[i]:
            count = count + 1

    prediction = (float(count)/float(m)) * 100.00
    print '[+] Prediction : ', prediction


if __name__ == '__main__':
    main()
