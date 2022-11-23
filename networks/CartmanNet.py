import numpy
import scipy.special
import weakref
import matplotlib.pyplot as plt
# scipy.special for the sigmoid function expit(), and its inverse logit()


class Sequential_Nets:
    # initialise the neural network
    instances = []
    def __init__(self, learningrate=0.5, *nodelist):
        # 获取实例的方法
        Sequential_Nets.instances.append(weakref.ref(self))

        # set number of nodes
        self.nodes = nodelist
        # 神经网络最大深度256层
        self.w = [None]*256

        # link weight matrices, weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        for i in range(len(self.nodes)-1) :
            # 用正态分布初始化权重，均值为 0，方差为输入节点数的开方分之 1
            self.w[i] = (numpy.random.normal(0.0, pow(self.nodes[i], -0.5), (self.nodes[i+1], self.nodes[i])))
        pass

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        # the function can receive a vector or matrix as input
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)


    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # creat a dict to record hidden outputs, the first element is inputs
        hid_flow = {}; hid_flow[0] = inputs

        # forward process
        for i in range(len(self.nodes) - 1):
            temp_value = numpy.dot(self.w[i], hid_flow[i])
            hid_flow[i+1] = self.activation_function(temp_value)
        pass

        # creat a dict to record errors, this represents inverse process
        errors={}; errors[len(self.nodes) - 2] = targets - hid_flow[len(self.nodes) - 1]
        for i in range(len(self.nodes) - 2, -1, -1):
            errors[i-1] = numpy.dot(self.w[i].T, errors[i])
            self.w[i]  += self.lr * numpy.dot((errors[i] * hid_flow[i + 1] * (1.0 - hid_flow[i + 1])),numpy.transpose(hid_flow[i]))
        pass


    # 接收网络输入返回网络输出
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # creat a dict to record hidden outputs
        hid_flow = {}; hid_flow[0] = inputs

        for i in range(len(self.nodes) - 1):
            temp_value = numpy.dot(self.w[i], hid_flow[i])
            hid_flow[i+1] = self.activation_function(temp_value)
        pass

        return hid_flow[list(hid_flow)[-1]]


    # backquery the neural network

    def inverse_flow(self, ground_truth):
        # one-hot encoding
        vector = numpy.zeros(self.nodes[-1]) + 0.01
        # vector[ground_truth] is the target label for this data
        vector[ground_truth] = 0.99
        print(f'{ground_truth}的向量形式为: {vector}')

        # creat a dict to record hidden outputs in inverse flow
        vector = numpy.array(vector, ndmin=2).T
        hid_flow = {}; hid_flow[len(self.nodes) - 1] = vector

        # inverse process
        for i in range(len(self.nodes) - 2, -1, -1):
            temp = self.inverse_activation_function(hid_flow[i+1])
            hid_flow[i] = numpy.dot(self.w[i].T, temp)
            # scale them back to 0.01 to .99
            hid_flow[i] -= numpy.min(hid_flow[i])
            hid_flow[i] /= numpy.max(hid_flow[i])
            hid_flow[i] *= 0.98
            hid_flow[i] += 0.01
        pass

        return hid_flow[0]


    def train_net(self, train_data, epochs):
        for j in range(epochs):
            for i, item in enumerate(train_data):
                # split the data by the ',' commas
                all_values = item.split(',')
                # scale and shift the inputs
                inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                # create the target output values (all 0.01, except the desired label which is 0.99)
                # 以 one-hot 编码方式进行目标制作
                targets = numpy.zeros(self.nodes[-1]) + 0.01
                # all_values[0] is the target label for this record
                targets[int(all_values[0])] = 0.99

                # 获取已经创建的实例列表
                instances = [ref() for ref in Sequential_Nets.instances if ref() is not None]
                # 选择实例列表中的第一个实例，train it
                instances[0].train(inputs, targets)
                if ((i + 1) % 2000) == 0:
                    print(f'=================================>第{i + 1}个数据')
            pass
        print(f'===>>第{j + 1}次迭代完成<<===')
        pass

    def evaluate(self, testdata):
        # scorecard for how well the network performs, initially empty
        scorecard = []

        # go through all the records in the test data set
        for record in testdata:
            # split the record by the ',' commas
            all_values = record.split(',')
            # correct answer is first value
            correct_label = int(all_values[0])
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # 获取已经创建的实例列表, get the instance list which has been created
            instances = [ref() for ref in Sequential_Nets.instances if ref() is not None]
            # query the network
            outputs = instances[0].query(inputs)
            # the index of the highest value corresponds to the label
            label = numpy.argmax(outputs)
            # append correct or incorrect to list
            if (label == correct_label):
                # network's answer matches correct answer, add 1 to scorecard
                scorecard.append(1)
            else:
                # network's answer doesn't match correct answer, add 0 to scorecard
                scorecard.append(0)
        pass

        # calculate the performance score, the fraction of correct answers
        scorecard_array = numpy.asarray(scorecard)
        print(f"准确率为 {scorecard_array.sum() / scorecard_array.size}")



if __name__ == '__main__':
    """
    这是一个训练的例子
    """
    # Hyperparameters
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    epochs = 1
    learning_rate = 0.1

    # load the mnist training data CSV file into a list
    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # instantiate
    n = Sequential_Nets(learning_rate,input_nodes, hidden_nodes, output_nodes,10 )
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