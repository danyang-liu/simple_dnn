import tensorflow as tf
from data_process import *
from metric import *
import numpy as np
from tensorflow.python.ops import array_ops

class Simple_dnn(object):

    def __init__(self, n_input, hidden_factor, learning_rate, epoch, batch_size, random_seed):
        self.n_input = n_input
        self.hidden_factor = hidden_factor
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.random_seed = random_seed

        self._init_graph()

    def focal_loss(self, predit_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
        r"""Compute focal loss for predictions.
            Multi-labels Focal loss formula:
                FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                     ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
        Args:
         prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
         target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
         weights: A float tensor of shape [batch_size, num_anchors]
         alpha: A scalar tensor for focal loss alpha hyper-parameter
         gamma: A scalar tensor for focal loss gamma hyper-parameter
        Returns:
            loss: A (scalar) tensor representing the value of the loss function
        """
        sigmoid_p = tf.nn.sigmoid(predit_tensor)
        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

        # For poitive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(
            tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
        return tf.reduce_sum(per_entry_cross_ent)

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
            layer1 = tf.nn.leaky_relu(tf.matmul(self.train_features, self.weights['h1']) + self.weights['b1'])
            layer2 = tf.nn.leaky_relu(tf.matmul(layer1, self.weights['h2']) + self.weights['b2'])
            self.out_layer = tf.sigmoid(tf.matmul(layer2, self.weights['out']) + self.weights['b_o'])
 #           self.out_layer = tf.matmul(layer2, self.weights['out']) + self.weights['b_o']

            # Loss
            # vars = tf.trainable_variables()
            # self.loss  = tf.nn.l2_loss(tf.subtract(self.train_lables, self.out_layer)) +  tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * 0.001

            vars = tf.trainable_variables()
            self.loss = tf.reduce_mean(tf.losses.log_loss(self.train_lables, self.out_layer))\
                        +  tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * 0.001

            # vars = tf.trainable_variables()
            # self.loss = self.focal_loss(self.out_layer, self.train_lables)+  tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * 0.001

            # self.loss = tf.reduce_mean(tf.losses.log_loss(self.train_lables, self.out_layer))

            # vars = tf.trainable_variables()
            # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.train_lables, tf.matmul(layer2, self.weights['out']) + self.weights['b_o']))\
            #             +  tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * 0.001
            # Optimizer
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
            #                                         epsilon=1e-8).minimize(self.loss)
            # self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

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
         # all_weights['h1'] = tf.get_variable(name='h1',shape=[self.n_input, self.hidden_factor[0]], initializer=tf.contrib.layers.xavier_initializer())
         # all_weights['b1'] = tf.get_variable(name='b1',shape=[self.hidden_factor[0]], initializer=tf.contrib.layers.xavier_initializer())
         # all_weights['h2'] = tf.get_variable(name='h2',shape=[self.hidden_factor[0],self.hidden_factor[1]], initializer=tf.contrib.layers.xavier_initializer())
         # all_weights['b2'] = tf.get_variable(name='b2',shape=[self.hidden_factor[1]], initializer=tf.contrib.layers.xavier_initializer())
         # all_weights['out'] = tf.get_variable(name='out',shape=[self.hidden_factor[1],1], initializer=tf.contrib.layers.xavier_initializer())
         # all_weights['b_o'] = tf.get_variable(name='b_o',shape=[1], initializer=tf.contrib.layers.xavier_initializer())
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
            total_loss = 0
            total_batch = int(len(Train_Data['X'])/self.batch_size)
            for i in range(total_batch):
                if i % 1000 == 0:
                    print('Epoch {}/{} Batch {}/{}'.format(epoch+1, self.epoch, i+1,
                                                       total_batch))
                batch_xs = self.get_random_block_from_data(Train_Data, self.batch_size)

                loss = self.partial_fit(batch_xs)
                total_loss = total_loss + loss
            print('Epoch {}/{} Loss{}'.format(epoch + 1, self.epoch, total_loss))
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

        auc = cal_auc(truth, pred)
        #
        print('TEST Epoch{} AUC{}'.format(epoch+1, auc))

        result_file_name = "Simple_dnn_11-16_128 " + "+learning_rate_" + str(self.learning_rate) + "+epoch_" + str(epoch + 1)

        result_file_path = "./results/Simple_dnn/" + result_file_name

        fp = open(result_file_path, 'w')

        for line_index in range(len(pred)):
            fp.write(str(pred[line_index][0]) + '\t' + str(truth[line_index][0]) + '\n')

        fp.close()

if __name__ == '__main__':
    # Load Data
    print("dv uv 11-16 128")
    print("Loading Data")
#    Train_Data = build_train_simple_dnn(udpairs_dv_uv_train)
#    Val_Data = build_train_simple_dnn(udpairs_dv_uv_val)
#    Test_Data = build_train_simple_dnn(udpairs_dv_uv_test_raw)
    # Train_Data = build_train_simple_dnn(udpairs_dv_uv_small)
    #Val_Data = build_train_simple_dnn(udpairs_dv_uv_small)
    # Test_Data = build_train_simple_dnn(udpairs_dv_uv_small)
    # Train_Data = build_train_simple_dnn_session("./data/exp_compare/2018-11-15_train_f.tsv")
    # Test_Data = build_train_simple_dnn_session("./data/exp_compare/2018-11-16_test_ff.tsv")
    Train_Data = build_train_simple_dnn_session("./data/exp_debug/2018-11-16_session_dv_uv_1.tsv")
    Test_Data = build_train_simple_dnn_session("./data/exp_debug/2018-11-16_session_dv_uv_2.tsv")
    print("Build model")
    # Build model & train
    model  = Simple_dnn(n_input = 180, hidden_factor = [128, 128], learning_rate = 0.001, epoch = 100, batch_size = 1024, random_seed = 2018)
#    model.train(Train_Data, Val_Data, Test_Data)
    model.train(Train_Data, Test_Data)