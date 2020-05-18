from game2048.agents import Agent

class myAgent(Agent):
    
    def __init__(self, game, display=None):

        super().__init__(game, display)
        self.model = load_model('model1.h5')

    def step(self):
        transformed_board = transform_board(self.game.board)
        y = self.model.predict(transformed_board)
        for i in range(4):
            if y[0,i]>0.5:
                return i

    def transform_board(board):
    	transformed_board = np.zeros((1,4,4,12))
    	for p in range(4):
        	for q in range(4):
            	if board[p,q] == 0:
                	transformed_board[0,p, q, 0] = 1
            	else:
                	transformed_board[0,p, q, int(np.log2(board[p,q]))] = 1
    	return transformed_board