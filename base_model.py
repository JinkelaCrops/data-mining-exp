import tensorflow as tf
import pandas as pd
import numpy as np
import time
import pickle
from collections import defaultdict
import os


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)


class Hparams:
    learning_rate = 0.005
    steps_per_stats = 100
    num_keep_ckpts = 2000
    batch_size = 500
    total_step = 10000


class BaseModel:
    global_step = 0

    def __init__(self, trainpd: pd.DataFrame, testpd: pd.DataFrame = None):
        self.data = trainpd
        self.test_data = testpd
        self.sess = None
        self.stats = defaultdict(float)
        self.hparams = Hparams()
        self.last_stats_step = 0
        self.last_eval_step = 0

        self.x = None
        self.y = None
        self.y_pred = None
        self.loss = None
        self.test_loss = tf.constant(0, tf.float32)
        self.step = None
        self.metric = None
        self.test_metric = tf.constant(0, tf.float32)

    @staticmethod
    def mini_batch(data, global_step, batch_size):
        actual_step = (global_step * batch_size) % data.shape[0]
        actual_step_1 = ((global_step + 1) * batch_size) % data.shape[0]
        if actual_step < actual_step_1:
            return data[actual_step: actual_step_1]
        else:
            return pd.concat([data[actual_step:], data[:actual_step_1]], axis=0)

    def model_define(self, xlabels, ylabels):
        self.x = tf.placeholder(tf.float32, shape=[self.hparams.batch_size, len(xlabels)], name="features")
        self.y = tf.placeholder(tf.float32, shape=[self.hparams.batch_size, len(ylabels)], name="target")

        # dense1
        layer1 = tf.layers.Dense(32, name="dense1", activation=tf.nn.sigmoid)
        layer1a = layer1(self.x)
        tf.summary.histogram("dense1", layer1a)

        # dense2
        layer11 = tf.layers.Dense(8, name="dense11", activation=tf.nn.sigmoid)
        layer11a = layer11(layer1a)
        tf.summary.histogram("dense11", layer11a)

        # dense2
        layer2 = tf.layers.Dense(len(ylabels), name="dense2")
        layer2p = layer2(layer11a)
        tf.summary.histogram("dense2", layer2p)
        # activate
        self.y_pred = tf.nn.sigmoid(layer2p, name="sigmoid")
        tf.summary.histogram("sigmoid", self.y_pred)

        # loss
        self.loss = tf.losses.mean_squared_error(self.y, self.y_pred)
        # accuracy
        self.metric = tf.reduce_mean(tf.cast(tf.equal(self.y, tf.round(self.y_pred)), tf.float32))
        optimizer = tf.train.AdamOptimizer(self.hparams.learning_rate)
        # using minimize
        self.step = optimizer.minimize(self.loss)

    def train(self, xlabels, ylabels):
        # delete tmp directory
        path = "tmp3"
        if tf.gfile.Exists(path):
            tf.gfile.DeleteRecursively(path)
        learning_rate = tf.constant(self.hparams.learning_rate)
        config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config_proto)
        # model define
        with self.sess:
            self.model_define(xlabels, ylabels)
        # init variables
        self.sess.run(tf.global_variables_initializer())
        # saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.hparams.num_keep_ckpts)
        summary_train_writer = tf.summary.FileWriter(path + "/train_log", self.sess.graph)
        summary_test_writer = tf.summary.FileWriter(path + "/test_log", self.sess.graph)

        # tf.summary.tensor_summary(tensor=self.sess.graph.get_tensor_by_name("dense/kernel:0"), name="weights")
        # tf.summary.tensor_summary(tensor=self.sess.graph.get_tensor_by_name("dense/bias:0"), name="bias")
        define_merge = tf.summary.merge_all()

        lr_summary = tf.summary.scalar("lr", learning_rate)

        test_loss_raw = tf.placeholder(tf.float32)
        test_metric_raw = tf.placeholder(tf.float32)
        test_summary = tf.summary.merge([tf.summary.scalar("loss", test_loss_raw),
                                         tf.summary.scalar("metric", test_metric_raw)])

        while self.global_step < self.hparams.total_step:
            start_time = time.time()
            mini_data = self.mini_batch(self.data, self.global_step, self.hparams.batch_size)
            x_data = mini_data[xlabels].values
            y_data = mini_data[ylabels].values
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            try:
                _, step_summary_all, lr, train_loss, train_acc = self.sess.run(
                    [self.step,
                     define_merge,
                     learning_rate,
                     self.loss,
                     self.metric],
                    feed_dict={self.x: x_data, self.y: y_data},
                    options=run_options,
                    # run_metadata=run_metadata
                )
            except tf.errors.OutOfRangeError:
                # Finished going through the training dataset.  Go to next epoch.
                self.global_step = 0
                continue

            # summary_train_writer.add_run_metadata(run_metadata, 'step%03d' % self.global_step)
            step_summary_1 = self.sess.run(lr_summary, feed_dict={learning_rate: lr})
            step_summary_2 = self.sess.run(test_summary,
                                           feed_dict={
                                               test_loss_raw: train_loss,
                                               test_metric_raw: train_acc
                                           })
            summary_train_writer.add_summary(step_summary_all, self.global_step)
            summary_train_writer.add_summary(step_summary_1, self.global_step)
            summary_train_writer.add_summary(step_summary_2, self.global_step)

            # Update statistics
            self.stats["global_step"] = self.global_step
            self.stats["loss"] += train_loss
            self.stats["metric"] += train_acc
            self.stats["step_time"] += time.time() - start_time
            self.stats["learning_rate"] = lr
            # Once in a while, we print statistics.
            if self.global_step - self.last_stats_step >= self.hparams.steps_per_stats:
                # mean loss
                self.stats["loss"] = self.stats["loss"] / self.hparams.steps_per_stats
                self.stats["metric"] = self.stats["metric"] / self.hparams.steps_per_stats

                if self.test_data is not None:
                    iter_num = int(np.ceil(self.test_data.shape[0] / self.hparams.batch_size))
                    for test_step in range(iter_num):
                        test_batch = self.mini_batch(self.test_data, test_step, self.hparams.batch_size)
                        test_loss, test_acc = self.sess.run(
                            [self.loss,
                             self.metric],
                            feed_dict={self.x: test_batch[xlabels], self.y: test_batch[ylabels]})

                        self.stats["test_loss"] += test_loss
                        self.stats["test_metric"] += test_acc

                    self.stats["test_loss"] = self.stats["test_loss"] / iter_num
                    self.stats["test_metric"] = self.stats["test_metric"] / iter_num
                    test_step_summary = self.sess.run(test_summary,
                                                      feed_dict={
                                                          test_loss_raw: self.stats["test_loss"],
                                                          test_metric_raw: self.stats["test_metric"]
                                                      })
                    summary_test_writer.add_summary(test_step_summary, self.global_step)

                    self.last_stats_step = self.global_step
                    print(dict(self.stats))
                    for i in self.stats.keys():
                        self.stats[i] = 0.0

            if self.global_step - self.last_eval_step >= self.hparams.steps_per_stats * 10:
                self.last_eval_step = self.global_step
                print("# Save eval, global step %d" % self.global_step)
                # Save checkpoint
                saver.save(self.sess, path + "/task.ckpt", global_step=self.global_step)

            self.global_step += 1

        print("done")
        summary_train_writer.close()


with open("data/my_data.x", "rb") as f:
    data = pickle.load(f)
print(list(data.columns))
# y: ["y"], x: ["user_level", "prize_level"]
train_data = data[data["tag"] == "train"]
test_data = data[data["tag"] == "test"]
x_label = ["prize_level", "prize_level", "user_prize", "channel_prize", "message_prize",
           "user_level", "user_level__1", "user_level__2", "user_level_2", "user_level_3",
           "channel_level", "channel_level__1", "channel_level__2", "channel_level_2", "channel_level_3",
           "message_level", "message_level__1", "message_level__2", "message_level_2", "message_level_3"
           ]
y_label = ["y"]

base_model = BaseModel(train_data, test_data)
self = base_model
base_model.train(xlabels=x_label, ylabels=y_label)
