import tensorflow as tf
import argparse
from utils import load_data


def get_next_batch(batch_size, data):
    st = 0
    ed = batch_size
    while st <= data.shape[0]:
        yield data[st:ed, :]
        st = ed
        ed = ed + batch_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Prometheus neural net using TensorFlow')
    parser.add_argument('--email', nargs='*', default='')
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--data_file', default='training.csv')
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--batch_size', default=100, type=int)

    args = parser.parse_args()

    train_data, test_data, train_weights, train_labels, test_labels = load_data(args.data_file, args.test_size, one_of_k=True)

    x = tf.placeholder('float', [None, 30])
    W = tf.Variable(tf.zeros([30, 2]))
    b = tf.Variable(tf.zeros([2]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder('float', [None, 2])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    batch_train_data = get_next_batch(100, train_data)
    batch_train_labels = get_next_batch(100, train_labels)
    for batchx, batchy in zip(batch_train_data, batch_train_labels):
        sess.run(train_step, feed_dict={x: batchx, y_: batchy})
        break

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    print(sess.run(y, feed_dict={x: test_data}))
    print(sess.run(correct_prediction, feed_dict={x: test_data, y_: test_labels}))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print(sess.run(accuracy, feed_dict={x: test_data, y_: test_labels}))


