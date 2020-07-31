# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

import rnncell as rnn
from utils import bio_to_json

from bert import modeling


class Model(object):
    def __init__(self, config):

        self.config = config
        self.char_dim = config["char_dim"]
        self.seg_dim = config["seg_dim"]
        self.lr = config["lr"]
        self.lstm_dim = config["lstm_dim"]
        self.num_tags = config["num_tags"]
        
        # 设置全局变量
        self.num_units = 2 * int(config["lstm_dim"])
        self.num_heads = 4

        # 初始化参数
        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model
        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids") 
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")  # 真实标签
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")

        used = tf.sign(tf.abs(self.input_ids))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.input_ids)[0]
        self.num_steps = tf.shape(self.input_ids)[-1]

        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        self.filter_width = 3
        self.num_filter = self.lstm_dim
        self.embedding_dim = 768
        self.repeat_times = 4
        self.cnn_output_width = 0

        """
        构造tensor的传递
        """
        # embeddings for chinese character and segmentation representation
        embedding = self.bert_embedding() 
        # apply dropout before feed to lstm layer
        idcnn_inputs = tf.nn.dropout(embedding, self.dropout)

        # idcnn layer
        idcnn_outputs = self.IDCNN_layer(idcnn_inputs)

        # multihead-attention mechanism
        attention_outputs = self.multihead_attention(idcnn_outputs)

        # logits for tags/进行预测，得到对每个字符是每个标签的概率
        self.logits = self.project_layer(attention_outputs)

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        # bert模型参数初始化的地方
        init_checkpoint = "chinese_L-12_H-768_A-12/bert_model.ckpt"
        # 获取模型中所有的训练参数。
        tvars = tf.trainable_variables()
        # 加载BERT模型
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        print("**** Trainable Variables ****")
        # 打印加载模型的参数
        train_vars = []
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            else:
                train_vars.append(var)
            print("  name = %s, shape = %s%s" % (var.name, var.shape,
                  init_string))
        # 设置训练阶段的优化算法
        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adagrad":  # 可选择优化器
                self.opt = tf.train.AdagradOptimizer(self.lr)
            elif optimizer == 'rmsprop':
                self.opt = tf.train.RMSPropOptimizer(self.lr)
            else:
                raise KeyError

            grads = tf.gradients(self.loss, train_vars)
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

            self.train_op = self.opt.apply_gradients(
                zip(grads, train_vars), global_step=self.global_step)
            # capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
            #                     for g, v in grads_vars if g is not None]
            # self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step, )

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def bert_embedding(self):
        # load bert embedding
        bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")
        model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)
        embedding = model.get_sequence_output()
        return embedding

    def IDCNN_layer(self, model_inputs, name=None):
        model_inputs = tf.expand_dims(model_inputs, 1)
        reuse = False
        if self.dropout == 1.0:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            shape = [1, self.filter_width, self.embedding_dim,
                       self.num_filter]

            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)

            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer", use_cudnn_on_gpu=True)
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True
                                           if (reuse or j > 0) else False):
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)

            finalOut = tf.squeeze(finalOut, [1])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut

    def multihead_attention(self, keys, scope='multihead_attention', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            Q = tf.nn.relu(tf.layers.dense(keys, self.num_filter * self.repeat_times, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            K = tf.nn.relu(tf.layers.dense(keys, self.num_filter * self.repeat_times, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            V = tf.nn.relu(tf.layers.dense(keys, self.num_filter * self.repeat_times, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            key_masks = tf.tile(key_masks, [self.num_heads, 1])
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[1], 1])
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
            outputs = tf.nn.softmax(outputs)
            query_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            query_masks = tf.tile(query_masks, [self.num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
            outputs *= query_masks
            outputs = tf.nn.dropout(outputs, keep_prob=self.dropout)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)
            outputs += keys
            outputs = self.normalize(outputs)
        return outputs

    def normalize(self, inputs, epsilon=1e-8, scope="ln", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape), dtype=tf.float32)
            gamma = tf.Variable(tf.ones(params_shape), dtype=tf.float32)
            normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
            outputs = gamma * normalized + beta
        return outputs

    def project_layer(self, multihead_outputs, name=None):
        """
        hidden layer between multihead layer and logits
        :param multihead_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            # 隐层的计算
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[(self.num_filter * self.repeat_times), self.num_filter],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_filter], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(multihead_outputs, shape=[-1, (self.num_filter * self.repeat_times)])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # 得到标签概率
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.num_filter, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        通过CRF层计算loss
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss" if not name else name):
            small = -1000.0
            # pad logits for crf loss 设置计算首个字符的概率，用于在CRF中计算真实首个字符的转移概率
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        创建feed_dict
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        _, segment_ids, chars, mask, tags = batch
        feed_dict = {
            self.input_ids: np.asarray(chars),
            self.input_mask: np.asarray(mask),
            self.segment_ids: np.asarray(segment_ids),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        运行sess
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        通过project_layer层得到的每个字符的标签概率和通过loss层得到的标签转移概率矩阵后，
        利用维特比算法对序列标签进行预测
        :param logits: [batch_size, num_steps, num_tags]float32, logits     对序列中字符标签的预测[batch_size, num_steps, num_tags]
        :param lengths: [batch_size]int32, real length of each sequence     每个序列除去padding字符的真实长度[batch_size]
        :param matrix: transaction matrix for inference       状态转移概率矩阵
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags + [0]])  # start是pad的概率为最大
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])  # 极小化预测出pad的概率
            logits = np.concatenate([score, pad], axis=1)  # 添加每个字符是pad的概率
            logits = np.concatenate([start, logits], axis=0)  # 将start的概率添加到序列前面
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        对一个数据集进行预测
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()  # tensor.eval()作用类似于sess.run()，目的在于运行图获取tensor,返回一个array
        for batch in data_manager.iter_batch():
            strings = batch[0]  # 原语句的字符列表
            labels = batch[-1]  # 原语句的labels列表
            lengths, scores = self.run_step(sess, False, batch)  # 运行sess进行预测，获取对每个字符的预测
            batch_paths = self.decode(scores, lengths, trans)  # 利用维特比算法基于状态概率和状态转移概率进行解码
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = [id_to_tag[int(x)] for x in labels[i][1:lengths[i]]]
                pred = [id_to_tag[int(x)] for x in batch_paths[i][1:lengths[i]]]
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        """
        预测一个电子病历语句
        """
        trans = self.trans.eval(sess)
        lengths, scores = self.run_step(sess, False, inputs)  # 运行sess进行预测，获取对每个字符的预测
        batch_paths = self.decode(scores, lengths, trans)  # 利用维特比算法基于状态概率和状态转移概率进行解码
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        
        return bio_to_json(inputs[0], tags[1:-1])





