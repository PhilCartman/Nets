from CartmanNet import *

# Hyperparameters
input_nodes = 784
hidden_nodes1 = 200
hidden_nodes2 = 10
output_nodes = 10
epochs = 1
learning_rate = 0.1

# load the mnist training data CSV file into a list
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# instantiate
n = Sequential_Nets(learning_rate,input_nodes, hidden_nodes1, hidden_nodes2, output_nodes)
# train the net
n.train_net(training_data_list,1)

# label
image_data = n.inverse_flow(3)

# load the mnist test data CSV file into a list
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
# evaluate the net
n.evaluate(test_data_list)

# visualization
plt.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')
plt.show()