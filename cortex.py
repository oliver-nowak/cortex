from scipy.io import loadmat
import numpy as np


def cost_function(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, nn_lambda, areLabelsZeroIndexed):
    m = len(X)
    J = 0
    Theta1_grad = np.zeros( (len(Theta1), 0) )
    Theta2_grad = np.zeros( (len(Theta2), 0) )
    
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
    
    #create yy array - where column = 1, with value of y[m] (as column index of yy)
    
    yy = np.zeros( (m, num_labels) )

    for i, y_label in enumerate(y):
        #create column index from y label (label is 1...K for now, instead of 0-indexed)
        if areLabelsZeroIndexed == False:
            col_ind = y_label - 1
        else:
            # else create column index with 'offset' 
            col_ind = y_label
        
        # label the correct column
        yy[i,col_ind] = 1

    # calculate Regularization Function
    regFunc = (nn_lambda/(2*m)) * (np.sum(np.sum(np.power(Theta1[:, 1:], 2))) + np.sum(np.sum(np.power(Theta2[:, 1:], 2))) )
    
    # calculate cost fuction

    out_layer = np.log(sigmoid_output_layer)
    print 'out_layer.shape : ', out_layer.shape
    print 'out_layer : ', out_layer[0,:]

    neg_y = yy * -1
    print 'neg_y : ',neg_y.shape
    print 'neg_y : ', neg_y[0,:]
    
    first_term = neg_y * out_layer 
    print 'first_term : ', first_term[0,:]
    print 'first_term shape : ', first_term.shape
    print 'first_term sum : ', first_term.sum(axis=0)

    second_term = (1 - yy) * np.log(1 - sigmoid_output_layer)

    sum_across = (first_term - second_term).sum(axis=0)

    sum_down = sum_across.sum()
    print sum_down
    
    divisor = 1.00/m

    print '1/m : ', divisor

    cost = divisor * sum_down
    print 'cost  : ', cost

    J = divisor * sum_down

    return J



def sigmoid(matrix):
    print '[+] calculating sigmoid for matrix...'
    return 1.0 / ( 1.0 + np.exp(-matrix) ) 

def predict_labels(Theta1, Theta2, X, areLabelsZeroIndexed):
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
    if areLabelsZeroIndexed == False:
        p = p + 1 
   

    return p

def predict(Theta1, Theta2, X, y, m):
    pred_data = predict_labels(Theta1, Theta2, X)
    
    count = 0
    for i in range(y.shape[0]):
        if y[i] == pred_data[i]:
            count = count + 1

    prediction = (float(count)/float(m)) * 100.00
    print '[+] Prediction : ', prediction


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

    input_layer_size  = 400
    hidden_layer_size = 25
    num_labels = 10  
    nn_lambda = 0
    #predict(Theta1, Theta2, X, y, m, areLabelsZeroIndexed=False)

    J = cost_function(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, nn_lambda, areLabelsZeroIndexed=False)

    print '[+] J : ', J

    
if __name__ == '__main__':
    main()
    
