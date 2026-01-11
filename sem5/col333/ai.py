import time
from collections import defaultdict
import copy
import numpy as np
from helper import *

# ------------------------------------------------------------------------------------------------------------------------------------------------

dimension = 4
my_player = 1
my_timer = None

# ------------------------------------------------------------------------------------------------------------------------------------------------

def count_filled_positions(board: np.array) -> int:
    return np.sum((board == 1) | (board == 2))

def count_unfilled_positions(board: np.array) -> int:
    return np.sum(board == 0)

# ------------------------------------------------------------------------------------------------------------------------------------------------

# from strategies.mcts4 import *
# from strategies.mcts6 import *

def heuristic_rollout_policy_4(possible_moves, board, player_num):
    corners = get_all_corners(board.shape[0])
    edges = get_all_edges(board.shape[0])
    
    best_move = None
    max_score = -float('inf')
    
    for move in possible_moves:
        score = 0
    
        if move in corners:
            score += 3  
        elif move in edges:
            score += 2  
    
        neighbours = get_neighbours(board.shape[0], move)
        for nx, ny in neighbours:
            if board[nx, ny] == player_num:
                score += 4

        if score > max_score:
            best_move = move
            max_score = score
    
    return best_move if best_move else possible_moves[0] 

def heuristic_rollout_policy_6(possible_moves, board, player_num):
    opponent = 3 - player_num
    corners = get_all_corners(board.shape[0])
    edges = get_all_edges(board.shape[0])
    
    best_move = None
    max_score = -float('inf')
    
    for move in possible_moves:
        score = 0
        (x, y) = move
        
        if move in corners:
            score += 0  
        elif move in edges:
            score += 2  
        else:
            score += 1
    
        neighbours = get_neighbours(board.shape[0], move)
        for nx, ny in neighbours:
            if board[nx, ny] == player_num:
                score += 4 

        if score > max_score:
            best_move = move
            max_score = score
    
    return best_move if best_move else possible_moves[0]  

class MonteCarloTreeSearchNode():
    def __init__(self, state, player_num, parent=None, parent_action=None):
        self.state = state
        self.player_num = player_num
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[0] = 0
        self._results[-1] = 0
        self._untried_actions = self.untried_actions()
        return
    
    def untried_actions(self):
        self._untried_actions = self.get_legal_actions()
        return self._untried_actions
    
    def q(self):
        return self._results[1] - self._results[-1]
    
    def n(self):
        return self._number_of_visits
    
    def expand(self):
        while(len(self._untried_actions)):
            action = self._untried_actions.pop()
            next_state = self.move(action)
            next_player_number = 3-self.player_num
            child_node = MonteCarloTreeSearchNode(next_state, parent=self, player_num = next_player_number, parent_action=action)
            
            for _ in range(100):
                reward = child_node.rollout()
                child_node.backpropagate(reward)
                
        
            self.children.append(child_node)
        
        return self.best_child() 
    
    
    def is_terminal_node(self):
        return self.is_game_over(self.player_num,self.parent_action)
    
    def rollout(self):
        store_state = copy.deepcopy(self.state)
        
        player = self.player_num
        action = self.parent_action
        while not self.is_game_over(player,action):
            
            possible_moves = self.get_legal_actions()
            
            action = self.rollout_policy(possible_moves)
            self.rollout_move(action,player)
            player = 3-player

            
            
        outcome = self.game_result()
        self.state = store_state
        
        return outcome
    
    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)
    
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    
    def best_child(self, c_param=1.4):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def rollout_policy(self, possible_moves):
        return heuristic_rollout_policy_4(possible_moves, self.state, self.player_num)
    
    
    def _tree_policy(self):
        current_node = self
                    
        if not current_node.is_fully_expanded():
            return current_node.expand()
        else:
            current_node = current_node.best_child()
        return current_node

    def best_action(self):
        c = 5
        start_time = time.time()
        
        simulation_no = 100000
        
        for i in range(simulation_no):
            if time.time()-start_time >= 10:
                break
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        best_child_node = self.best_child(c_param=1.4)
        
        return best_child_node.parent_action
    
    def get_legal_actions(self): 
        l = get_valid_actions(self.state,self.player_num)
        return l
        
    def is_game_over(self,player,action):
        valid_actions = get_valid_actions(self.state, player)

        if len(valid_actions) == 0:
            return True
        else:
            if(action == None): return False
            win, way = check_win(self.state,action,(3-player))
            if(win): return True
            else: return False

    def game_result(self):
        player = self.player_num
    
        if(my_player == player):
            (our_player,_) = check_win(self.state,self.parent_action,(3-player))
            if(our_player):
                return 1
            else:
                return 0
        else:
            (opp_player,_) = check_win(self.state,self.parent_action,(3-player))
            if(opp_player):
                return -1
            else:
                return 0

    
    def rollout_move(self,action,player_id):
        (x,y) = action
        self.state[x][y] = player_id
        
        
    def move(self,action):
        new_state = copy.deepcopy(self.state)
        
        (x,y) = action
        new_state[x][y] = self.player_num
        
        return new_state

def make_move_4(initial_state,player):
    root = MonteCarloTreeSearchNode(state = initial_state, player_num = player)
    selected_node = root.best_action()

    return selected_node

def make_move_6(initial_state,player):
    root = MonteCarloTreeSearchNode(state = initial_state, player_num = player)
    selected_node = root.best_action()

    return selected_node

# ------------------------------------------------------------------------------------------------------------------------------------------------
# from utils.brute import *

def check_for_win(board, player_num):
    valid_moves = get_valid_actions(board)
    for move in valid_moves:
        temp_board = board.copy()
        temp_board[move] = player_num
        win, _ = check_win(temp_board, move, player_num)  
        if win:
            return True, move  
    return False, None

def check_for_loss(board, player_num):
    opponent = 3 - player_num  
    valid_moves = get_valid_actions(board)
    for move in valid_moves:
        temp_board = board.copy()
        temp_board[move] = opponent
        win, _ = check_win(temp_board, move, opponent)  
        if win:
            return True, move  
    return False, None

def check_mate_in_2(board, player_num):
    valid_moves = get_valid_actions(board)
    for move in valid_moves:
        temp_board = board.copy()
        temp_board[move] = player_num
        win, _ = check_win(temp_board, move, player_num)
        if win:
            return True, move

        opponent = 3 - player_num
        opponent_moves = get_valid_actions(temp_board)
        for opp_move in opponent_moves:
            temp_board_opp = temp_board.copy()
            temp_board_opp[opp_move] = opponent

            player_second_moves = get_valid_actions(temp_board_opp)
            for second_move in player_second_moves:
                temp_board_player = temp_board_opp.copy()
                temp_board_player[second_move] = player_num
                win, _ = check_win(temp_board_player, second_move, player_num)
                if win:
                    return True, move
    return False, None

def check_loss_in_2(board, player_num):
    opponent = 3 - player_num
    valid_moves = get_valid_actions(board)
    for move in valid_moves:
        temp_board = board.copy()
        temp_board[move] = opponent
        win, _ = check_win(temp_board, move, opponent)
        if win:
            return True, move

        player_moves = get_valid_actions(temp_board)
        for player_move in player_moves:
            temp_board_player = temp_board.copy()
            temp_board_player[player_move] = player_num

            opponent_second_moves = get_valid_actions(temp_board_player)
            for opp_second_move in opponent_second_moves:
                temp_board_opp_second = temp_board_player.copy()
                temp_board_opp_second[opp_second_move] = opponent
                win, _ = check_win(temp_board_opp_second, opp_second_move, opponent)
                if win:
                    return True, move
    return False, None

def check_mate_in_3(board, player_num):
    valid_moves = get_valid_actions(board)
    for move in valid_moves:
        temp_board = board.copy()
        temp_board[move] = player_num
        win, _ = check_win(temp_board, move, player_num)
        if win:
            return True, move

        opponent = 3 - player_num
        opponent_moves = get_valid_actions(temp_board)
        for opp_move in opponent_moves:
            temp_board_opp = temp_board.copy()
            temp_board_opp[opp_move] = opponent

            player_second_moves = get_valid_actions(temp_board_opp)
            for second_move in player_second_moves:
                temp_board_player = temp_board_opp.copy()
                temp_board_player[second_move] = player_num
                win, _ = check_win(temp_board_player, second_move, player_num)
                if win:
                    return True, move

                opponent_second_moves = get_valid_actions(temp_board_player)
                for opp_second_move in opponent_second_moves:
                    temp_board_opp_second = temp_board_player.copy()
                    temp_board_opp_second[opp_second_move] = opponent

                    player_third_moves = get_valid_actions(temp_board_opp_second)
                    for third_move in player_third_moves:
                        temp_board_player_third = temp_board_opp_second.copy()
                        temp_board_player_third[third_move] = player_num
                        win, _ = check_win(temp_board_player_third, third_move, player_num)
                        if win:
                            return True, move
    return False, None

def check_loss_in_3(board, player_num):
    opponent = 3 - player_num
    valid_moves = get_valid_actions(board)
    for move in valid_moves:
        temp_board = board.copy()
        temp_board[move] = opponent
        win, _ = check_win(temp_board, move, opponent)
        if win:
            return True, move

        player_moves = get_valid_actions(temp_board)
        for player_move in player_moves:
            temp_board_player = temp_board.copy()
            temp_board_player[player_move] = player_num

            opponent_second_moves = get_valid_actions(temp_board_player)
            for opp_second_move in opponent_second_moves:
                temp_board_opp_second = temp_board_player.copy()
                temp_board_opp_second[opp_second_move] = opponent
                win, _ = check_win(temp_board_opp_second, opp_second_move, opponent)
                if win:
                    return True, move

                player_third_moves = get_valid_actions(temp_board_opp_second)
                for player_third_move in player_third_moves:
                    temp_board_player_third = temp_board_opp_second.copy()
                    temp_board_player_third[player_third_move] = player_num

                    opponent_third_moves = get_valid_actions(temp_board_player_third)
                    for opp_third_move in opponent_third_moves:
                        temp_board_opp_third = temp_board_player_third.copy()
                        temp_board_opp_third[opp_third_move] = opponent
                        win, _ = check_win(temp_board_opp_third, opp_third_move, opponent)
                        if win:
                            return True, move
    return False, None

def first_move_strategy(board: np.array, player_num: int) -> Tuple[int, int]:
    dim = board.shape[0]
    corners = get_all_corners(dim)

    if player_num == 1 and np.sum((board == 1) | (board == 2)) == 0:
        return corners[0]
    
    if player_num == 2:
        opponent_moves = np.argwhere(board == 1)
        if len(opponent_moves) > 0:
            opponent_move = tuple(opponent_moves[0])

            if opponent_move in corners:
                corner_edges = {
                    corners[0]: [corners[1], corners[5]],  
                    corners[1]: [corners[0], corners[2]],  
                    corners[2]: [corners[1], corners[3]],
                    corners[3]: [corners[2], corners[4]],  
                    corners[4]: [corners[3], corners[5]],  
                    corners[5]: [corners[0], corners[4]]
                }
                
                
                return corner_edges[opponent_move][0] 
            else:
                return corners[0]  


    valid_moves = get_valid_actions(board)
    return valid_moves[0] if valid_moves else None

# ------------------------------------------------------------------------------------------------------------------------------------------------
# from utils.dim4 import if_dim_is_4

def if_dim_is_4(board,player_number):
    if  count_filled_positions(board) <=1:
        return first_move_strategy(board,player_number)
    
    win,move1 = check_for_win(board,player_number)
    if win:
        return move1
    
    lose,move2 = check_for_loss(board,player_number)
    if lose:
        return move2
    
    win_in_2, move3 = check_mate_in_2(board, player_number)
    if win_in_2:
        return move3

    lose_in_2, move4 = check_loss_in_2(board, player_number)
    if lose_in_2:
        return move4
    
    if count_filled_positions(board) >= 10:
        win_in_3, move5 = check_mate_in_3(board, player_number)
        if win_in_3:
            return move5

        # check if opponent can win in 3 moves
        lose_in_3, move6 = check_loss_in_3(board, player_number)
        if lose_in_3:
            return move6

# ------------------------------------------------------------------------------------------------------------------------------------------------
# from utils.dim6 import if_dim_is_6

def if_dim_is_6(board,player_number):
    
    if count_filled_positions(board) >= 10:
        win,move1 = check_for_win(board,player_number)
        if win:
            return move1
        
        lose,move2 = check_for_loss(board,player_number)
        if lose:
            return move2
    
    if count_unfilled_positions(board) <= 60:
        win_in_2, move3 = check_mate_in_2(board, player_number)
        if win_in_2:
            return move3

        lose_in_2, move4 = check_loss_in_2(board, player_number) 
        
        if lose_in_2:
            return move4
    
    if count_unfilled_positions(board) <= 25:
        win_in_3, move5 = check_mate_in_3(board, player_number)
        if win_in_3:
            return move5

        # check if opponent can win in 3 moves
        lose_in_3, move6 = check_loss_in_3(board, player_number)
        if lose_in_3:
            return move6

# ------------------------------------------------------------------------------------------------------------------------------------------------

''' opening theory for dim 6 '''

d = {
	1:[[(1,9),(0,10)],[(2,9),(2,10)],[(4,9),(3,10)],[(5,9),(5,10)]],
	2:[[(1,9),(0,10)],[(1,8),(0,8)],[(1,6),(0,7)],[(1,5),(0,5)]],
	3:[[(5,9),(5,10)],[(4,9),(3,10)],[(2,9),(2,10)],[(1,9),(0,10)]],
	4:[[(5,9),(5,10)],[(6,8),(7,8)],[(8,6),(8,7)],[(9,5),(10,5)]],
	5:[[(10,5),(9,5)],[(8,6),(8,7)],[(6,7),(6,8)],[(4,9),(5,9)]],
	6:[[(10,5),(9,5)],[(8,3),(8,4)],[(6,2),(7,2)],[(5,0),(5,1)]],
	7:[[(5,0),(5,1)],[(6,2),(7,2)],[(8,3),(8,4)],[(9,5),(10,5)]],
	8:[[(5,0),(5,1)],[(3,0),(4,1)],[(2,0),(2,1)],[(0,0),(1,1)]],
	9:[[(0,0),(1,1)],[(2,0),(2,1)],[(3,0),(4,1)],[(5,0),(5,1)]],
	10:[[(0,0),(1,1)],[(0,2),(1,2)],[(0,3),(1,4)],[(0,5),(1,5)]],
	11:[[(0,5),(1,5)],[(0,3),(1,4)],[(0,2),(1,2)],[(0,0),(1,1)]],
	12:[[(0,5),(1,5)],[(1,6),(0,7)],[(0,8),(1,8)],[(1,9),(0,10)]],
	13:[[(6,8),(5,9)],[(4,8),(4,9)],[(2,9),(2,10)],[(1,9),(0,10)]],
	14:[[(1,4),(1,5)],[(1,6),(2,6)],[(0,8),(1,8)],[(1,9),(0,10)]],
	15:[[(1,8),(1,9)],[(3,8),(2,9)],[(4,9),(3,10)],[(5,9),(5,10)]],
	16:[[(8,4),(9,5)],[(7,6),(8,6)],[(6,8),(7,8)],[(5,9),(5,10)]],
	17:[[(5,9),(5,10)],[(6,8),(7,8)],[(8,6),(8,7)],[(9,5),(10,5)]],
	18:[[(4,1),(5,1)],[(6,2),(6,3)],[(8,3),(8,4)],[(9,5),(10,5)]],
	19:[[(9,5),(8,6)],[(8,4),(7,4)],[(7,2),(6,2)],[(5,0),(5,1)]],
	20:[[(1,1),(1,2)],[(2,1),(3,2)],[(3,0),(4,1)],[(5,0),(5,1)]],
	21:[[(5,1),(6,2)],[(4,1),(4,2)],[(2,0),(2,1)],[(0,0),(1,1)]],
	22:[[(0,0),(1,1)],[(0,2),(1,2)],[(1,4),(2,4)],[(1,5),(1,6)]],
	23:[[(1,1),(2,1)],[(1,2),(2,3)],[(0,3),(1,4)],[(0,5),(1,5)]],
	24:[[(0,5),(1,5)],[(1,6),(0,7)],[(2,7),(1,8)],[(2,9),(1,9)]]
}
		
l = [[(0,9),(1,10),(6,9),(3,9),(4,10)],
     [(0,9),(1,10),(0,4),(1,7),(0,6)],
     [(4,10),(6,9),(0,9),(3,9),(1,10)],
     [(4,10),(6,9),(9,4),(7,7),(9,6)],
     [(9,6),(9,4),(4,10),(7,7),(5,8)],
     [(9,6),(9,4),(4,0),(7,3),(6,1)],
     [(4,0),(6,1),(9,6),(7,3),(9,4)],
     [(4,0),(6,1),(0,1),(3,1),(1,0)],
     [(0,1),(1,0),(6,1),(3,1),(4,0)],
     [(0,1),(1,0),(0,6),(1,3),(0,4)],
     [(0,4),(0,6),(1,0),(1,3),(0,1)],
     [(0,4),(0,6),(1,10),(1,7),(0,9)],
     [(0,9),(1,10),(6,9),(3,9),(5,8)],
     [(0,9),(1,10),(0,4),(1,7),(2,5)],
     [(4,10),(6,9),(0,9),(3,9),(2,8)],
     [(4,10),(6,9),(9,4),(7,7),(8,5)],
     [(9,6),(9,4),(4,10),(7,7),(6,9)],
     [(9,6),(9,4),(4,0),(7,3),(5,2)],
     [(4,0),(6,1),(9,6),(7,3),(8,5)],
     [(4,0),(6,1),(0,1),(3,1),(2,2)],
     [(0,1),(1,0),(6,1),(3,1),(5,2)],
     [(0,1),(1,0),(0,6),(1,3),(2,5)],
     [(0,4),(0,6),(1,0),(1,3),(2,2)],
     [(0,4),(0,6),(1,10),(1,7),(2,8)]]

def strategy(board,player_num):
	best_list_index = -1
	max_occ = -1
	for i,li in enumerate(l):
		occ = 0
		for j in li:
			if(board[j]==(3-player_num)):
				occ = -1
				break
			elif(board[j]==player_num):
				occ += 1
		if(occ>max_occ):
			max_occ = occ
			best_list_index = i
	if(max_occ==-1):
		return False,None
	if((max_occ)==5):
		return mate_in_4(board,best_list_index+1,player_num)
	else:
		li = l[best_list_index]
		for move in li:
			if(board[move]==0):
				return True,move

def mate_in_4(board,index,player_num):
		moves = d[index]
		cnt = 0
		
		move = None
		for list_moves in moves:
			
			if((board[list_moves[0]]==player_num) or (board[list_moves[1]]==player_num)):
				continue
			elif(board[list_moves[0]]==(3-player_num) and (board[list_moves[1]]==(3-player_num))):
				return False,None
			else:
				curr = 0
				if(board[list_moves[0]]==0):
					move = list_moves[0]
					curr += 1
				if(board[list_moves[1]]==0):
					move = list_moves[1]
					curr += 1
				
				if(curr == 1):
					return True,move
		return True,move
	

# ------------------------------------------------------------------------------------------------------------------------------------------------

''' opening theory for dim 4'''


d1 = {
	(0,0) :[(1,2),(2,1)],
	(3,0):[(2,1),(4,2)],
	(6,3):[(4,2),(4,4)],
	(3,6):[(4,4),(2,5)],
	(0,6):[(2,5),(1,4)],
	(0,3):[(1,4),(1,2)]
}

def inbetween_marked(board,cell,corner,player_num):
	nbr_list = get_neighbours(dim=7,vertex=corner)
	nbr_list2 = get_neighbours(dim=7,vertex=cell)
	for nbr in nbr_list:
		cnt = 0
		if((nbr in nbr_list2)):
			if((board[nbr]==player_num)):
				cnt+=1
	if(board[corner]==player_num):
		cnt+=1
	if(cnt==0):
		return False,corner
	else:
		return True,None
		
def check_for_strat_opp(board,player_num):
	opp = (3-player_num)
	corners = get_all_corners(dim=7)
	
	prev_corner = corners[-1]
	for corner in corners:
		if((board[prev_corner]==opp) and(board[corner]==opp)):
			for nbr in d1[prev_corner]:
				if((nbr in d1[corner])):
					if(board[nbr]==player_num):
						print("already saved")
						continue
					elif(board[nbr]==0):
						print('saviour is here')
						a1,b1 = inbetween_marked(board,nbr,corner,player_num)
						a2,b2 = inbetween_marked(board,nbr,prev_corner,player_num)
						if(not a1 and not a2):
							return True,nbr
		prev_corner = corner
	prev_corner = corners[-1]

	for corner in corners:
		for cell in d1[prev_corner]:
			if((cell in d1[corner])):
				if(board[prev_corner]==opp and board[cell]==opp):
					a,b = inbetween_marked(board,cell,corner,player_num)
					if(not a): return True,b
				elif(board[corner]==opp and board[cell]==opp):
					a,b = inbetween_marked(board,cell,prev_corner,player_num)
					if(not a): return True,b
		prev_corner =corner
	return False,None


# ------------------------------------------------------------------------------------------------------------------------------------------------

class AIPlayer:

    def __init__(self, player_number: int, timer):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        

    def get_move(self, board: np.array) -> Tuple[int, int]:
        
        dimension = (board.shape[0] + 1) // 2    
        my_player = self.player_number  
        my_timer = self.timer
        
        if dimension == 4:
            a = if_dim_is_4(board,self.player_number)
            if a:
                (x,y) = a
                move = (int(x),int(y))
                return move 
            
        if dimension == 4:
            a,b = check_for_strat_opp(board, self.player_number)
            if(a):
                (x,y) = b
                move = (int(x),int(y))
                return move    
        
        if dimension == 4:
            a = make_move_4(board,self.player_number)
            (x,y) = a
            move = (int(x),int(y))
            return move
            
        if dimension == 6:
            a = if_dim_is_6(board,self.player_number)
            if a:
                (x,y) = a
                move = (int(x),int(y))
                return move 
        
        if dimension == 6:
            if(count_filled_positions(board) <= 40):
                (a,b) = strategy(board,self.player_number)
                if a:
                    (x,y) = b
                    move = (int(x),int(y))
                    return move
                
            a = make_move_6(board,self.player_number)
            (x,y) = a
            move = (int(x),int(y))
            return move
                    
            
           

