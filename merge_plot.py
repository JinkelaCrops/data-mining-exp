import tensorflow as tf
from numpy import random

writer_1 = tf.summary.FileWriter("logs/plot_1")
writer_2 = tf.summary.FileWriter("logs/plot_2")

log_var = tf.Variable(0.0)
tf.summary.scalar("loss", log_var)

write_op = tf.summary.merge([tf.summary.scalar("loss", log_var)])

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

for i in range(100):
    # for writer 1
    summary = session.run(write_op, {log_var: random.rand()})
    writer_1.add_summary(summary, i)
    writer_1.flush()

    # for writer 2
    summary = session.run(write_op, {log_var: random.rand()})
    writer_2.add_summary(summary, i)
    writer_2.flush()
