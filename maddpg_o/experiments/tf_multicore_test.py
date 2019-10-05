import tensorflow as tf
import numpy as np
import multiprocessing
import time

class Runner(multiprocessing.Process):
    def __init__(self, index, ready_queue, input_queue):
        super().__init__()
        self.index = index
        self.ready_queue = ready_queue
        self.input_queue = input_queue

    def run(self):
        with tf.Session(servers[self.index].target) as sess:
            with tf.variable_scope("scope_{}".format(self.index)):
                a = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                b = tf.contrib.layers.fully_connected(a, 10000)
                c = tf.contrib.layers.fully_connected(b, 10000)
                d = tf.contrib.layers.fully_connected(c, 1)
                sess.run(tf.global_variables_initializer())
                self.ready_queue.put(None)

                while True:
                    inpt = self.input_queue.get()
                    if inpt is not None:
                        self.ready_queue.put(sess.run(d, feed_dict={a: inpt}))
                    else:
                        break

n = 2
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
servers = [tf.train.Server(cluster, job_name="local", task_index=i) for i in range(n)]
ready_queues = [multiprocessing.Queue() for _ in range(n)]
input_queues = [multiprocessing.Queue() for _ in range(n)]
runners = [Runner(i, ready_queues[i], input_queues[i]) for i in range(n)]

for i in range(n):
    runners[i].start()

for i in range(n):
    ready_queues[i].get()

start_t = time.time()
for j in range(100):
    for i in range(n):
        input_queues[i].put([np.random.random(1)])
    for i in range(n):
        answer = ready_queues[i].get()
        print(j, i, answer)
print(time.time() - start_t)

for i in range(n):
    input_queues[i].put(None)

for i in range(n):
    runners[i].join()