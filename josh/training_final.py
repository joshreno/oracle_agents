import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tflearn
import pickle
import csv, random, glob, random
from tflearn import conv_2d, max_pool_2d, local_response_normalization, batch_normalization, fully_connected, regression, input_data, dropout, custom_layer, flatten, reshape, embedding,conv_2d_transpose
import tensorflow as tf
from tqdm import tqdm
from enum import Enum
import os
tf.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

allNames = ['Ground', 'Stair', 'Treetop', 'Block', 'Bar', 'Koopa', 'Koopa 2', 'PipeBody', 'Pipe', 'Question', 'Coin', 'Goomba', 'CannonBody', 'Cannon', 'Lakitu', 'Bridge', 'Hard Shell', 'SmallCannon', 'Plant', 'Waves', 'Hill', 'Castle', 'Snow Tree 2', 'Cloud 2', 'Cloud', 'Bush', 'Tree 2', 'Bush 2', 'Tree', 'Snow Tree', 'Fence', 'Bark', 'Flag', 'Mario']
actions = ['Ground', 'Stair', 'Treetop', 'Block', 'Bar', 'Koopa', 'Koopa 2', 'PipeBody', 'Pipe', 'Question', 'Coin', 'Goomba', 'CannonBody', 'Cannon', 'Lakitu', 'Bridge', 'Hard Shell', 'SmallCannon', 'Plant', 'Waves', 'Hill', 'Castle', 'Snow Tree 2', 'Cloud 2', 'Cloud', 'Bush', 'Tree 2', 'Bush 2', 'Tree', 'Snow Tree', 'Fence', 'Bark', 'Nothing', 'Nothing']

trainX = np.load("smallX.npy")
print(trainX.shape)
half = trainX.shape[0]/10
trainX = trainX[0:int(half)]
print(trainX.shape)
# trainY = np.load("y.npy")
# print(trainY.shape)

humanX = np.load("humanX.npy")
humanY = np.load("humanY.npy")
# humanDataX = humanX[0:humanX.shape[0]*0.8, :, :, :]
# humanDataY = humanX[0:humanX.shape[0]*0.8, :, :, :]
humanValX = humanX[int(humanX.shape[0] * 0.8):int(humanX.shape[0] * 0.9), :, :, :]
humanValY = humanY[int(humanY.shape[0] * 0.8):int(humanY.shape[0] * 0.9), :, :, :]
# humanTestX = humanX[humanX.shape[0] * 0.9:humanX.shape[0], :, :, :]
# humanTestY = humanX[humanX.shape[0] * 0.9:humanX.shape[0], :, :, :]

class Cases(Enum):
	TRAIN_FROM_SCRATCH = 1
	TRAIN_FROM_LEFT_OFF = 2
	TRAIN_ORACLE_THEN_HUMAN = 3

#case = Cases.TRAIN_FROM_SCRATCH
case = Cases.TRAIN_ORACLE_THEN_HUMAN
stop = False
num_training = 0
threshold = 0.0
difference = float('inf')

# Matthew's original model
networkInput = tflearn.input_data(shape=[None, 40, 15, len(allNames)]) # 40 x 15 x 34
conv = conv_2d(networkInput, 8,4, activation='leaky_relu')
conv2 = conv_2d(conv, 16,3, activation='leaky_relu')
conv3 = conv_2d(conv2, 32,3, activation='leaky_relu')
fc = tflearn.fully_connected(conv3, 40*15*33, activation='leaky_relu')
mapShape = tf.reshape(fc, [-1,40,15,33])
network = tflearn.regression(mapShape, optimizer='adam', metric='accuracy', loss='mean_square')
model = tflearn.DNN(network)
modelString = "fullModel/testFull.tflearn"
model.load(modelString)

while not stop and num_training < 5:
	print("TRAINING ITERATION" + str(num_training))

	qTableOfActions = model.predict(trainX) # qtable of actions from Matthew's model or subsequently trained models
	model = None # clearing model ... not sure if this works the way I'm thinking
	print("q-table", qTableOfActions.shape)
	np.save("qTableOfActions" + str(num_training) + ".npy", qTableOfActions)

	qTableOfActions = np.resize(qTableOfActions, (qTableOfActions.shape[0], qTableOfActions.shape[1], qTableOfActions.shape[2], 34))
	print("q-tabled resized", qTableOfActions.shape)

	tf.reset_default_graph()

	maxValuesBelowThreshold = 0
	maxValues = 0
	oracleInput = np.zeros((qTableOfActions.shape[0], 40, 15, 34, 2))
	for hi in tqdm(range(qTableOfActions.shape[0])):
		for i in range(40):
			for j in range(15):
				maximum_value = float('-inf')
				maximum_index = 0
				for k in range(33):
					oracleInput[hi][i][j][k][0] = trainX[hi][i][j][k] # states
					if qTableOfActions[hi][i][j][k] > maximum_value:
						maximum_value = qTableOfActions[hi][i][j][k]
						maximum_index = k
				if maximum_value > threshold:
					oracleInput[hi][i][j][maximum_index][1] = maximum_value # actions
				if maximum_value < 0: maxValuesBelowThreshold += 1
				maxValues += 1
	print("Oracle Input Shape", oracleInput.shape)
	print("Num below threshold", maxValuesBelowThreshold)
	print("Num values total", maxValues)
	np.save('oracleInput' + str(num_training) + '.npy', oracleInput)

	network_input =  tflearn.input_data(shape=[None, 40, 15, 34, 2])
	conv = tflearn.conv_3d(network_input, 5, 5, activation = 'leaky_relu')
	fc = tflearn.fully_connected(conv, 1, activation = 'leaky_relu')
	network = tflearn.regression(fc, metric = 'accuracy', loss = 'mean_square')
	oracle = tflearn.DNN(network)
	oracle.load('newModel/model.tfl')

	oracleResult = oracle.predict(oracleInput)
	np.save('oracleResult' + str(num_training) + '.npy', oracleResult)
	oracle = None # again, not sure if this fully clears the model

	for hi in tqdm(range(oracleResult.shape[0])):
		for i in range(40):
			for j in range(15):
				for k in range(34):
					oracleInput[hi, i, j, k, 1] *= oracleResult[hi] # multiplying by value from oracle
	np.save('oracleResultMultiplied' + str(num_training) + '.npy', oracleInput)

	state = oracleInput[:, :, :, :, 0] 
	if case == Cases.TRAIN_FROM_LEFT_OFF:
	    action = oracleInput[:, :, :, 0:33, 1]
	elif case == Cases.TRAIN_FROM_SCRATCH:
	    action = oracleInput[:, :, :, :, 1]
	else:
	    if num_training == 0:
		    action = oracleInput[:, :, :, 0:33, 1]
	    else:
		    action = oracleInput[:, :, :, :, 1]
	    
	print(state.shape)
	print(action.shape)
	
	tf.reset_default_graph()
	networkInput = tflearn.input_data(shape=[None, 40, 15, len(allNames)])
	conv = conv_2d(networkInput, 8,4, activation='leaky_relu')
	conv2 = conv_2d(conv, 16,3, activation='leaky_relu')
	conv3 = conv_2d(conv2, 32,3, activation='leaky_relu')
	if case == Cases.TRAIN_FROM_SCRATCH or (case == Cases.TRAIN_ORACLE_THEN_HUMAN and num_training != 0):
            print('training based on size of 34')
            fc = tflearn.fully_connected(conv3, 40*15*len(actions), activation='leaky_relu')
            mapShape = tf.reshape(fc, [-1,40,15,len(actions)])
	elif case == Cases.TRAIN_FROM_LEFT_OFF or (case == Cases.TRAIN_ORACLE_THEN_HUMAN and num_training == 0):
	    print('training based on original size of 33')
	    fc = tflearn.fully_connected(conv3, 40*15*(len(actions) - 1), activation='leaky_relu')
	    mapShape = tf.reshape(fc, [-1, 40, 15, len(actions)-1])
	network = tflearn.regression(mapShape, optimizer='adam', metric='accuracy', loss='mean_square')
	model = tflearn.DNN(network)
	if case == Cases.TRAIN_FROM_LEFT_OFF or (case == Cases.TRAIN_ORACLE_THEN_HUMAN and num_training == 0):
		print('training based on original model', modelString)
		# if we're training from where previous model left off, then load the previous model using modelString
		model.load(modelString)
	print("state", state.shape)
	print("action", action.shape)
	model.fit(state, 
 		Y_targets=action, 
 		n_epoch=100, 
		shuffle=True, 
 		show_metric=True, 
 		snapshot_epoch=False,
 		batch_size=16)
	#	run_id='oracle1')
	modelString = 'fullModel' + str(num_training) + '.tflearn'
	model.save(modelString)

	humanValYPrediction = model.predict(humanValX)
	a,b,c,d = humanValYPrediction.shape
	humYPred = np.zeros_like(humanValY)
	humYPred[:a, :b, :c, :d] = humanValYPrediction
	diff = humYPred - humanValY
	diff = np.average(diff)
	diff = np.average(diff)
	diff = np.average(diff)
	diff = np.average(diff)
	print("Difference", diff)
	if diff >= difference:
		stop = True

	print("Next iteration ...\n")
	trainX = state


	num_training += 1
