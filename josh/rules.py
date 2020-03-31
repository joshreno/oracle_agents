import numpy as np
import random
from enum import IntEnum

class Actions(IntEnum):
	GROUND = 0
	STAIR = 1
	TREETOP = 2
	BLOCK = 3
	BAR = 4
	KOOPA = 5 # ENEMY
	KOOPA2 = 6 # ENEMY
	PIPEBODY = 7
	PIPE = 8
	QUESTION = 9
	COIN = 10
	GOOMBA = 11 # ENEMY
	CANNONBODY = 12
	CANNON = 13
	LAKITU = 14 # ENEMY
	BRIDGE = 15
	HARDSHELL = 16
	SMALLCANNON = 17
	PLANT = 18
	WAVES = 19
	HILL = 20
	CASTLE = 21
	SNOWTREE2 = 22
	CLOUD2 = 23
	CLOUD = 24
	BUSH = 25
	TREE2 = 26
	BUSH2 = 27
	TREE = 28
	SNOWTREE = 29
	FENCE = 30
	BARK = 31
	NOTHING = 32
	NOTHING2 = 33

def shift_by_index(inp, move_up, move_right):
	#assuming inp is np array [40 x 15 x 34]
	rows = len(inp)
	columns = len(inp[0])
	new_inp = np.empty([40, 15, 34])
	for i in range(rows):
		for j in range(columns):
			new_inp[(i + move_up) % rows][(j + move_right) % columns] = inp[i][j]
	return new_inp

def random_selection(inp):
	rows = len(inp)
	columns = len(inp[0])
	for i in range(rows):
		for j in range(columns):
			if inp[i][j][-1] == "Nothing": 
				continue
			else:
				for i in inp[i][j]:
					i = 0
				inp[i][j] = [0] * len(inp[i][j])
				inp[i][j][random.randint(0, 34)] = 1


def strategy(inp):
	# print("STRATEGY")
	# print(inp.shape)
	inp = getRidOfEnemies(inp)
	# print(inp.shape)
	inp = swapGroundAndBlocks(inp)
	# print(inp.shape)
	inp = addCoins(inp)
	# print(inp.shape)
	inp = addTree(inp)
	# print(inp.shape)	
	return inp

#Strat 1
def getRidOfEnemies(inp):
	#assuming inp is np array [100 x 15 x 14]
	for i in range(len(inp)):
		for j in range(len(inp[0])):
			sumation = np.sum(inp[i, j]) == 0
			if inp[i, j, Actions.KOOPA2] == 1:
				if random.randint(0, 10) > 5:
					inp[i, j, Actions.KOOPA2] = 0
			else:
				if random.randint(0, 100) > 1 and sumation:
					inp[i, j, Actions.KOOPA2] = 1
			if inp[i, j, Actions.KOOPA] == 1:
				if random.randint(0, 10) > 5:
					inp[i, j, Actions.KOOPA] = 0
			else:
				if random.randint(0, 100) > 1 and sumation:
					inp[i, j, Actions.KOOPA] = 1
			if inp[i, j, Actions.LAKITU] == 1:
				if random.randint(0, 10) > 5:
					inp[i, j, Actions.LAKITU] = 0
			else:
				if random.randint(0, 100) > 1 and sumation:
					inp[i, j, Actions.LAKITU] = 1
			if inp[i, j, Actions.GOOMBA] == 1:
				if random.randint(0, 10) > 5:
					inp[i, j, Actions.GOOMBA] = 0
			else:
				if random.randint(0, 100) > 1 and sumation:
					inp[i, j, Actions.GOOMBA] = 1
	return inp

#Strat 2
def swapGroundAndBlocks(inp):
	for i in range(len(inp)):
		for j in range(len(inp[0])):
			if inp[i, j, Actions.GROUND] == 1:
				inp[i, j, Actions.GROUND] = 0
				inp[i, j, Actions.BLOCK] = 1
	return inp

#Strat 3
def addCoins(inp):
	for i in range(len(inp)):
		groundBlock = False
		for j in range(len(inp[0])):
			if inp[i, j, Actions.GROUND] == 1:
				groundBlock = True
			else:
				if groundBlock and random.randint(0, 100) > 5 and np.sum(inp[i, j]) == 0:
					inp[i, j, Actions.COIN] = 1
	return inp

#Strat 4
def addTree(inp):
	for i in range(len(inp)):
		groundBlock = -1
		for j in range(len(inp[0])):
			if inp[i, j, Actions.GROUND] == 1:
				groundBlock = j
		if groundBlock + 1 < 15 and np.sum(inp[i, groundBlock + 1]) == 0:
			rand = random.randint(0, 100)
			if rand < 3:
				inp[i, groundBlock + 1, Actions.SNOWTREE] = 1
			elif rand < 6:
				inp[i, groundBlock + 1, Actions.SNOWTREE2] = 1
			elif rand < 13:
				inp[i, groundBlock + 1, Actions.TREE] = 1
			elif rand < 20:
				inp[i, groundBlock + 1, Actions.TREE2] = 1
			else:
				pass
	return inp
