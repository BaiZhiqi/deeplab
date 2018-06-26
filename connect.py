import tensorflow as tf


def main():
    a = tf.Variable(3.0, dtype=tf.float32, name='a')
    b = tf.constant(4.0, dtype=tf.float32, name='b')
    c = a + b

    with tf.Session("grpc://172.16.10.50:2222") as sess:
        #sess.run(tf.global_variables_initializer())
        result = sess.run(c)
        print(result)


if __name__ == '__main__':
    main()
