import tensorflow as tf
from data_process import *
from metric import *
import numpy as np

class Simple_dnn(object):

    def __init__(self, n_input, hidden_factor, learning_rate, epoch, batch_size, random_seed):
        self.n_input = n_input
        self.hidden_factor = hidden_factor
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.random_seed = random_seed

        self._init_graph()

    def _init_graph(self):

        self.graph = tf.Graph()

        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            # Inputs
            self.train_features = tf.placeholder(tf.float32, shape=[None, self.n_input])
            self.train_lables = tf.placeholder(tf.float32, shape=[None,1])

            # Variables
            self.weights = self._initialize_weights()

            # Model
            layer1 = tf.nn.relu(tf.matmul(self.train_features, self.weights['h1']) + self.weights['b1'])
            layer2 = tf.nn.relu(tf.matmul(layer1, self.weights['h2']) + self.weights['b2'])
            self.out_layer = tf.matmul(layer2, self.weights['out']) + self.weights['b_o']

            # Loss
            self.loss  = tf.nn.l2_loss(tf.subtract(self.train_lables, self.out_layer))

            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)

            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)


    def _initialize_weights(self):
         all_weights = dict()
         all_weights['h1'] = tf.Variable(tf.random_normal([self.n_input, self.hidden_factor[0]]))
         all_weights['b1'] = tf.Variable(tf.random_normal([self.hidden_factor[0]]))
         all_weights['h2'] = tf.Variable(tf.random_normal([self.hidden_factor[0],self.hidden_factor[1]]))
         all_weights['b2'] = tf.Variable(tf.random_normal([self.hidden_factor[1]]))
         all_weights['out'] = tf.Variable(tf.random_normal([self.hidden_factor[1],1]))
         all_weights['b_o'] = tf.Variable(tf.random_normal([1]))

         return all_weights


    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X'], self.train_lables: data['Y']}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss


    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X , Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append(data['Y'][i])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}

    # def train(self, Train_Data, Val_Data, Test_Data):
    def train(self, Train_Data, Test_Data):
        for epoch in range(self.epoch):
            total_batch = int(len(Train_Data['X'])/self.batch_size)
            for i in range(total_batch):
                if i % 1000 == 0:
                    print('Epoch {}/{} Batch {}/{}'.format(epoch+1, self.epoch, i+1,
                                                       total_batch))
                batch_xs = self.get_random_block_from_data(Train_Data, self.batch_size)

                loss = self.partial_fit(batch_xs)

            # self.evaluate(epoch, Val_Data)
            # if (epoch+1) % 5 == 0:
            self.predict(epoch, Test_Data)

    def evaluate(self, epoch, data):

        pred = []
        total_batch = int(len(data['X']) / self.batch_size)
        for i in range(total_batch):
            feed_dict = {self.train_features: data['X'][i*self.batch_size:(i+1)*self.batch_size], self.train_lables: data['Y'][i*self.batch_size:(i+1)*self.batch_size]}
            out_layer = self.sess.run((self.out_layer), feed_dict=feed_dict)
            for item in out_layer:
                pred.append(item)

        total_batch = int(len(data['X']) / self.batch_size)
        feed_dict = {self.train_features: data['X'][total_batch*self.batch_size:], self.train_lables: data['Y'][total_batch*self.batch_size:]}
        out_layer = self.sess.run((self.out_layer), feed_dict=feed_dict)
        for item in out_layer:
            pred.append(item)

        truth = data['Y']

        assert(len(pred) == len(truth))

        auc = cal_auc(truth, pred)

        print('VAL Epoch{} AUC{}'.format(epoch+1, auc))

    def predict(self, epoch, data):

        pred = []

        total_batch = int(len(data['X']) / self.batch_size)
        for i in range(total_batch):
            feed_dict = {self.train_features: data['X'][i * self.batch_size:(i + 1) * self.batch_size],
                         self.train_lables: data['Y'][i * self.batch_size:(i + 1) * self.batch_size]}
            out_layer = self.sess.run((self.out_layer), feed_dict=feed_dict)
            for item in out_layer:
                pred.append(item)

        total_batch = int(len(data['X']) / self.batch_size)
        feed_dict = {self.train_features: data['X'][total_batch * self.batch_size:],
                     self.train_lables: data['Y'][total_batch * self.batch_size:]}
        out_layer = self.sess.run((self.out_layer), feed_dict=feed_dict)
        for item in out_layer:
            pred.append(item)

        truth = data['Y']
        sess_id = data['Sess_id']
        assert (len(pred) == len(truth))

        ndcg = []
        truth_i = []
        pred_i = []
        for i in range(len(truth)):
            truth_i.append(truth[i])
            pred_i.append(pred[i])
            if i + 1 == len(truth):
                ndcg.append(cal_ndcg(truth_i, pred_i, 10))
                truth_i = []
                pred_i = []
            elif sess_id[i] != sess_id[i+1]:
                ndcg.append(cal_ndcg(truth_i, pred_i, 10))
                truth_i = []
                pred_i = []
        ndcg = np.mean(np.array(ndcg))

        print('TEST Epoch{} NDCG{}'.format(epoch+1, ndcg))

        # auc = cal_auc(truth, pred)
        #
        # print('TEST Epoch{} AUC{}'.format(epoch+1, auc))

        result_file_name = "1Simple_dnn" + "+learning_rate_" + str(self.learning_rate) + "+epoch_" + str(epoch + 1)

        result_file_path = "./results/Simple_dnn/" + result_file_name

        fp = open(result_file_path, 'w')

        for line_index in range(len(pred)):
            fp.write(str(pred[line_index][0]) + '\t' + str(truth[line_index][0]) + '\n')

        fp.close()

if __name__ == '__main__':
    # Load Data
    print("Loading Data")
#    Train_Data = build_train_simple_dnn(udpairs_dv_uv_train)
#    Val_Data = build_train_simple_dnn(udpairs_dv_uv_val)
#    Test_Data = build_train_simple_dnn(udpairs_dv_uv_test_raw)
    # Train_Data = build_train_simple_dnn(udpairs_dv_uv_small)
    # Val_Data = build_train_simple_dnn(udpairs_dv_uv_small)
    # Test_Data = build_train_simple_dnn(udpairs_dv_uv_small)
    Train_Data = build_train_simple_dnn_session(udpairs_dv_uv_session_train_sample)
    Test_Data = build_train_simple_dnn_session("./data/2018-11-15.standardized.udPairs_dv_uv_session_test.tsv")
    print("Build model")
    # Build model & train
    model  = Simple_dnn(n_input = 180, hidden_factor = [256, 256], learning_rate = 0.001, epoch = 100, batch_size = 1024, random_seed = 2018)
#    model.train(Train_Data, Val_Data, Test_Data)
    model.train(Train_Data, Test_Data)