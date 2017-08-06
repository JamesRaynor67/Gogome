import numpy as np
from scipy import signal

class Game(object):
	def __init__(self):
		self.board = np.zeros([15,15]).astype(int)
		self.nextTrun = 'B' # Black on board denote by 1, Write by -1
		self.winner = None
		self.record = [] # [[x0,y0,c0],[x1,y1,c1],...[xn,yn,cn]]
		self.state = np.zeros([9, 15, 15]).astype(int)
		# 0: thisTrunColor (mySide)
		# 1: nextTrunColor (opponent)
		# 2: empty
		# 3: myside two sequence
		# 4: opponent two sequence
		# 5: myside three seq
		# 6: opponent three seq
		# 7: myside four seq
		# 8: opponent four seq

	def printBoard(self):
		print self.board

	def putStone(self, color, x, y):
		assert self.nextTrun == color
		if color == 'B':
			self.board[x, y] = 1
			self.nextTrun = 'W'
		else:
			self.board[x, y] = -1
			self.nextTrun = 'B'

	def checkPosLegal(self, x, y):
		return True

	def checkGameEnd(self):
		x, y, color = self.record[-1]

	def generateState(self, color):
		# color denotes which is 'myside'
		if color == 'B':
			self.state[0] = np.array(self.board == 1).astype(int)
			self.state[1] = np.array(self.board == -1).astype(int)
		else:
			self.state[0] = np.array(self.board == -1).astype(int)
			self.state[1] = np.array(self.board == 1).astype(int)

		atLeast2SeqBlack = ((self._seq2HoriProcess(self.state[0]) + self._seq2VerProcess(self.state[0]) + 
							self._seq2Diagonal(self.state[0]) + self._seq2ViceDiagonal(self.state[0])) != 0).astype(int)
		np.clip(atLeast2SeqBlack, a_min=None, a_max=1)
		
		atLeast2SeqWhite = ((self._seq2HoriProcess(self.state[1]) + self._seq2VerProcess(self.state[1]) + 
							self._seq2Diagonal(self.state[1]) + self._seq2ViceDiagonal(self.state[1])) != 0).astype(int)
		np.clip(atLeast2SeqWhite, a_min=None, a_max=1)

		atLeast3SeqBlack = ((self._seq3HoriProcess(self.state[0]) + self._seq3VerProcess(self.state[0]) + 
							self._seq3Diagonal(self.state[0]) + self._seq3ViceDiagonal(self.state[0])) != 0).astype(int)
		np.clip(atLeast3SeqBlack, a_min=None, a_max=1)
		
		atLeast3SeqWhite = ((self._seq3HoriProcess(self.state[1]) + self._seq3VerProcess(self.state[1]) + 
							self._seq3Diagonal(self.state[1]) + self._seq3ViceDiagonal(self.state[1])) != 0).astype(int)
		np.clip(atLeast3SeqWhite, a_min=None, a_max=1)

		atLeast4SeqBlack = ((self._seq4HoriProcess(self.state[0]) + self._seq4VerProcess(self.state[0]) + 
							self._seq4Diagonal(self.state[0]) + self._seq4ViceDiagonal(self.state[0])) != 0).astype(int)
		np.clip(atLeast4SeqBlack, a_min=None, a_max=1)
		
		atLeast4SeqWhite = ((self._seq4HoriProcess(self.state[1]) + self._seq4VerProcess(self.state[1]) + 
							self._seq4Diagonal(self.state[1]) + self._seq4ViceDiagonal(self.state[1])) != 0).astype(int)
		np.clip(atLeast4SeqWhite, a_min=None, a_max=1)

		if color == 'B':
			self.state[0] = np.array(self.board == 1).astype(int)
			self.state[1] = np.array(self.board == -1).astype(int)
			self.state[3] = atLeast2SeqBlack
			self.state[4] = atLeast2SeqWhite
			self.state[5] = atLeast3SeqBlack
			self.state[6] = atLeast3SeqWhite
			self.state[7] = atLeast4SeqBlack
			self.state[8] = atLeast4SeqWhite
		else:
			self.state[0] = np.array(self.board == -1).astype(int)
			self.state[1] = np.array(self.board == 1).astype(int)
			self.state[3] = atLeast2SeqWhite
			self.state[4] = atLeast2SeqBlack
			self.state[5] = atLeast3SeqWhite
			self.state[6] = atLeast3SeqBlack
			self.state[7] = atLeast4SeqWhite
			self.state[8] = atLeast4SeqBlack
		self.state[2] = np.array(self.board == 0).astype(int)


	def _seq2VerProcess(self, oneSideOnlyBoard):
		kernel = [[1],[1]]
		# do convelution on horizontal direction
		con = (signal.convolve2d(oneSideOnlyBoard, kernel, boundary='fill', mode='same') == 2).astype(int) # fillvalue == 0
		# shift and find all 2 seqence stones
		shiftedCon = np.roll(con, shift=-1, axis=0)
		# Notice that the last line must be all 0, so x|y directly
		return con + shiftedCon

	def _seq2HoriProcess(self, oneSideOnlyBoard):
		kernel = [[1, 1]]
		con = (signal.convolve2d(oneSideOnlyBoard, kernel, boundary='fill', mode='same') == 2).astype(int)
		shiftedCon = np.roll(con, shift=-1, axis=1)
		return con + shiftedCon

	def _seq2Diagonal(self, oneSideOnlyBoard):
		kernel = [[1,0],[0,1]]
		con = (signal.convolve2d(oneSideOnlyBoard, kernel, boundary='fill', mode='same') == 2).astype(int)
		shiftedCon = np.roll((np.roll(con, shift=-1, axis = 0)), shift=-1, axis = 1)
		return con + shiftedCon

	def _seq2ViceDiagonal(self, oneSideOnlyBoard):
		kernel = [[0,1],[1,0]]
		con = (signal.convolve2d(oneSideOnlyBoard, kernel, boundary='fill', mode='same') == 2).astype(int)
		# print (con == 2).astype(int)
		shiftedCon1 = np.roll(con, shift=-1, axis = 0)
		shiftedCon2 = np.roll(con, shift=-1, axis = 1)
		return shiftedCon1 + shiftedCon2

	def _seq3VerProcess(self, oneSideOnlyBoard):
		kernel = [[1],[1],[1]]
		con = (signal.convolve2d(oneSideOnlyBoard, kernel, boundary='fill', mode='same') == 3).astype(int)
		shiftedCon1 = np.roll(con, shift=-1, axis=0)
		shiftedCon2 = np.roll(con, shift=1, axis=0)
		return con + shiftedCon1 + shiftedCon2

	def _seq3HoriProcess(self, oneSideOnlyBoard):
		kernel = [[1, 1, 1]]
		con = (signal.convolve2d(oneSideOnlyBoard, kernel, boundary='fill', mode='same') == 3).astype(int)
		shiftedCon1 = np.roll(con, shift=-1, axis=1)
		shiftedCon2 = np.roll(con, shift=1, axis=1)
		return con + shiftedCon1 + shiftedCon2

	def _seq3Diagonal(self, oneSideOnlyBoard):
		kernel = [[1,0,0],[0,1,0],[0,0,1]]
		con = (signal.convolve2d(oneSideOnlyBoard, kernel, boundary='fill', mode='same') == 3).astype(int)
		shiftedCon1 = np.roll((np.roll(con, shift=-1, axis = 0)), shift=-1, axis = 1)
		shiftedCon2 = np.roll((np.roll(con, shift=1, axis = 0)), shift=1, axis = 1)
		return con + shiftedCon1 + shiftedCon2

	def _seq3ViceDiagonal(self, oneSideOnlyBoard):
		kernel = [[0,0,1],[0,1,0],[1,0,0]]
		con = (signal.convolve2d(oneSideOnlyBoard, kernel, boundary='fill', mode='same') == 3).astype(int)
		shiftedCon1 = np.roll((np.roll(con, shift=-1, axis = 0)), shift=1, axis = 1)
		shiftedCon2 = np.roll((np.roll(con, shift=1, axis = 0)), shift=-1, axis = 1)
		return con + shiftedCon1 + shiftedCon2

	def _seq4VerProcess(self, oneSideOnlyBoard):
		kernel = [[1],[1],[1],[1]]
		con = (signal.convolve2d(oneSideOnlyBoard, kernel, boundary='fill', mode='same') == 4).astype(int)
		shiftedCon1 = np.roll(con, shift=-2, axis=0)
		shiftedCon2 = np.roll(con, shift=-1, axis=0)
		shiftedCon3 = np.roll(con, shift=1, axis=0)
		return con + shiftedCon1 + shiftedCon2 + shiftedCon3

	def _seq4HoriProcess(self, oneSideOnlyBoard):
		kernel = [[1, 1, 1, 1]]
		con = (signal.convolve2d(oneSideOnlyBoard, kernel, boundary='fill', mode='same') == 4).astype(int)
		shiftedCon1 = np.roll(con, shift=-2, axis=1)
		shiftedCon2 = np.roll(con, shift=-1, axis=1)
		shiftedCon3 = np.roll(con, shift=1, axis=1)
		return con + shiftedCon1 + shiftedCon2 + shiftedCon3

	def _seq4Diagonal(self, oneSideOnlyBoard):
		kernel = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
		con = (signal.convolve2d(oneSideOnlyBoard, kernel, boundary='fill', mode='same') == 4).astype(int)
		shiftedCon1 = np.roll((np.roll(con, shift=-2, axis = 0)), shift=-2, axis = 1)
		shiftedCon2 = np.roll((np.roll(con, shift=-1, axis = 0)), shift=-1, axis = 1)
		shiftedCon3 = np.roll((np.roll(con, shift=1, axis = 0)), shift=1, axis = 1)
		return con + shiftedCon1 + shiftedCon2 + shiftedCon3

	def _seq4ViceDiagonal(self, oneSideOnlyBoard):
		kernel = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
		con = (signal.convolve2d(oneSideOnlyBoard, kernel, boundary='fill', mode='same') == 4).astype(int)
		shiftedCon1 = np.roll((np.roll(con, shift=1, axis = 0)), shift=-2, axis = 1)
		shiftedCon2 = np.roll(con, shift=-1, axis = 1)
		shiftedCon3 = np.roll(con, shift=-1, axis = 0)
		shiftedCon4 = np.roll((np.roll(con, shift=-2, axis = 0)), shift=1, axis = 1)
		return shiftedCon1 + shiftedCon2 + shiftedCon3 + shiftedCon4

game = Game()

if __name__ == '__main__':
	game.board = np.random.randint(-1,2,[15,15])
	game.generateState('B')
	print game.board
	for s in game.state:
		print s
