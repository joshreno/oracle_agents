import tflearn
import tensorflow as tf
import numpy as np

from data import Action, Block, Decoration, retrieve_dataset


def main():
    (X_train, Y_train), (X_test, Y_test) = retrieve_dataset()

    action_size = len(Action)
    network_input = tflearn.input_data(shape=[None, 40, 15, len(Block) + len(Decoration)])
    conv = tflearn.conv_2d(network_input, 8, 4, activation='leaky_relu')
    conv2 = tflearn.conv_2d(conv, 16, 3, activation='leaky_relu')
    conv3 = tflearn.conv_2d(conv2, 32, 3, activation='leaky_relu')
    fc = tflearn.fully_connected(conv3, 40 * 15 * action_size, activation='leaky_relu')
    map_shape = tf.reshape(fc, [-1, 40 , 15, action_size])
    network = tflearn.regression(map_shape, optimizer='adam', metric='accuracy', loss='mean_square')
    model = tflearn.DNN(network)
    model.fit(X_train,
              Y_targets=Y_train,
              n_epoch=100,
              shuffle=True,
              show_metric=True,
              snapshot_epoch=False,
              batch_size=16,
              run_id='cocreativeTest')
    X_test = np.reshape(X_test, (-1, 40, 15, len(Block) + len(Decoration)))
    Y_test = np.reshape(Y_test, (-1, 40, 15, action_size))
    Y_pred = model.predict(X_test)
    score = 0
    for t in range(0, len(Y_pred)):
        predictedChunk = Y_pred[t]
        trueChunk = Y_test[t]
        for x in range(0, 40):
            for y in range(0, 15):
                for z in range(0, action_size):
                    score += predictedChunk[x][y][z] * trueChunk[x][y][z]
    print ("Final Score: "+ str(score))

if __name__ == '__main__':
    main()
