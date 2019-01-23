#!/usr/bin/env python

import numpy as np
import random
import copy
import timeit
from args import parse_args

class Game:

    def __init__(self, args):

        self.HISTORY = 1

        self._args   = args

        self._turn        = 'X'
        self._size        = self._args.game_size
        self._n_in_a_row  = self._args.game_n_in_a_row
        self._state_layer = self.HISTORY * 2 + 1

        self._board  = np.zeros((self._size, self._size), dtype=int)
        #self._last   = np.zeros((self._size, self._size), dtype=int)
        self._black = []
        self._white = []
        for i in range(self.HISTORY):
            self._black.append(np.zeros((self._size, self._size), dtype=int))
            self._white.append(np.zeros((self._size, self._size), dtype=int))

        self._moves = []


    def clone(self):
        game = Game(self._args)

        game._turn    = self._turn
        game._args    = self._args

        game._board   = np.copy(self._board)
        #game._last    = np.copy(self._last)
        game._black   = []
        game._white   = []
        for i in range(self.HISTORY):
            game._black.append(np.copy(self._black[i]))
            game._white.append(np.copy(self._white[i]))

        game._moves   = copy.copy(self._moves)
        return game


    def size(self):
        return (self._size, self._size)
    
    def state_shape(self):
        return (self._state_layer, self._size, self._size)

    def action_size(self):
        return self._size * self._size

    def _state(self):
        data = np.zeros((self.HISTORY*2+1, self._size, self._size), dtype=np.float32)
        data[0, :, :] = 1 if self._turn == 'X' else 0
        for i in range(self.HISTORY):
            data[i+1, :, :] = self._black[i].copy()
            data[self.HISTORY+i+1, :, :] = self._white[i].copy()
        #data[-1, :, :] = self._last.copy()
        return data
    
    def _state_inversed(self):
        data = np.zeros((self.HISTORY*2+1, self._size, self._size), dtype=np.float32)
        data[0, :, :] = 1 if self._turn == 'X' else 0
        for i in range(self.HISTORY):
            data[i+1, :, :] = self._white[i].copy()
            data[self.HISTORY+i+1, :, :] = self._black[i].copy()
        #data[-1, :, :] = self._last.copy()
        return data

    def state_normalized(self):
        if self._args.game_state_normalize:
            if self._turn == 'X':
                return self._state()
            else:
                return self._state_inversed()
        else:
            return self._state()
        

    def state_size(self):
        return self.HISTORY * 2 + 2

    def turn(self):
        return self._turn

    def moves(self):
        return self._moves
    
    def string(self):
        return self._string(self._board)

    def _string(self, board):
        result = ""
        for i in range(self._size):
            for j in range(self._size):
                result += str(board[i][j] if board[i][j]>=0 else 2)
        return result

    def get_state_symetries(self, probs):
        boards = {}
        boards[self._string(self._board)] = {'state': self.state_normalized(), 'probs': probs}
        
        probs_matrix = np.reshape(probs, (self._size, self._size))

        for i in (1, 2, 3):
            board_r = self._string(np.rot90(self._board, i))
            if board_r not in boards:
                game_r = self.clone()
                game_r._board   = np.rot90(self._board, i)
                for h in range(self.HISTORY):
                    game_r._black[h] = np.rot90(self._black[h], i)
                    game_r._white[h] = np.rot90(self._white[h], i)
                probs_r         = np.reshape(np.rot90(probs_matrix, i), self._size * self._size)
                boards[board_r] = {'state': game_r.state_normalized(), 'probs': probs_r}

        board_f = self._string(np.fliplr(self._board))
        if board_f not in boards:
            game_f = self.clone()
            game_f._board    = np.fliplr(self._board)
            for h in range(self.HISTORY):
                game_f._black[h]  = np.fliplr(self._black[h])
                game_f._white[h]  = np.fliplr(self._white[h])
            probs_matrix_f   = np.fliplr(probs_matrix)
            probs_f          = np.reshape(probs_matrix_f, self._size * self._size)
            
            boards[board_f] = {'state': game_f.state_normalized(), 'probs': probs_f}
            
            for i in (1,2,3):
                board_r = self._string(np.rot90(game_f._board, i))
                if board_r not in boards:
                    game_r = game_f.clone()
                    game_r._board   = np.rot90(game_f._board, i)
                    for h in range(self.HISTORY):
                        game_r._black[h] = np.rot90(game_f._black[h], i)
                        game_r._white[h] = np.rot90(game_f._white[h], i)
                    probs_r         = np.reshape(np.rot90(probs_matrix_f, i), self._size * self._size)
                    boards[board_r] = {'state': game_r.state_normalized(), 'probs': probs_r}
            
        return boards

    def display(self):
        print('+---'*self._size + '+')
        last_move = self._moves[-1] if len(self._moves) != 0 else None
        for i in range(self._size):
            print("|", end='')
            for j in range(self._size):
                if self._board[i][j] == 1:
                    dot = 'X'
                elif self._board[i][j] == -1:
                    dot = 'O'
                else:
                    dot = ' '
                if last_move is not None and last_move[0] == i and last_move[1] == j:
                    print("(" + dot + ")|", end='')
                else:
                    print(" " + dot + " |", end='')
            print(' %2d' % (self._size - i), end='')
            print()
            print('+---'*self._size + '+')
        for i in range(self._size):
            print("  %s " % chr(ord('A')+i), end='')
        print()


    def is_valid_move(self, row, col):
        if self._board[row][col] != 0:
            return False
        else:
            return True


    def all_valid_moves(self):
        moves = []
        for i in range(self._size):
            for j in range(self._size):
                if self._board[i][j] == 0:
                    moves.append((i,j))
        return moves
    
    def all_valid_actions(self):
        actions = []
        for i in range(self._size):
            for j in range(self._size):
                if self._board[i][j] == 0:
                    actions.append(i*self._size + j)
        return actions
        
    def valid_action_map(self):
        actions = [0] * self.action_size()
        for i in range(self._size):
            for j in range(self._size):
                if self._board[i][j] == 0:
                    actions[i*self._size + j] = 1
        return actions

    
    def make_action(self, action):
        row = int(action / self._size)
        col = action % self._size
        #print(row, col)
        return self.make_move((row, col))
        

    # return true or false: true means win, false means no
    def make_move(self, position):
        row = position[0]
        col = position[1]
        if row < 0 or col < 0 or row >= self._size or col >= self._size:
            raise Exception("row or col out of bound: %d, %d", row, col)

        if not self.is_valid_move(row, col):
            raise Exception("invalid mode (row, col), already occupied: $d", self._board[row][col])

        if self.is_game_over():
            raise Exception("game over, no more moves allowed!")

        color = 1 if self._turn == 'X' else -1

        self._board[row][col] = color
        #if len(self._moves) > 0:
        #    self._last[self._moves[-1][0], self._moves[-1][1]] = 0

        if self._turn == 'X':
            for i in reversed(range(self.HISTORY)):
                if i != 0:
                    self._black[i] = self._black[i-1].copy()
            self._black[0][row][col] = 1
        else:
            for i in reversed(range(self.HISTORY)):
                if i != 0:
                    self._white[i] = self._white[i-1].copy()
            self._white[0][row][col] = 1
        #self._last[row, col] = 1
        self._moves.append((row, col, self._turn))

        self._turn = 'O' if self._turn == 'X' else 'X'
        #print(self._turn)

        return self.has_won(row, col, color)


    """
    def undo_move(self):
        if len(self._moves) <= 0:
            raise Exception("beginning of game, no more move to undo")

        last_move=self._moves[-1]
        self._board[last_move[0]][last_move[1]] = 0
        self._black_0[last_move[0]][last_move[1]] = 0
        self._white_0[last_move[0]][last_move[1]] = 0

        self._moves = self._moves[:-1]
        self._turn = 'O' if self._turn == 'X' else 'X'
    """

    def is_n_in_a_row(self, row, col, color, dir_row, dir_col):
        count = 0
        for i in range(-self._n_in_a_row+1, self._n_in_a_row):
            curr_row = row + i*dir_row
            if curr_row<0 or curr_row>=self._size:
                continue

            curr_col = col + i*dir_col
            if curr_col<0 or curr_col>=self._size:
                continue

            if self._board[curr_row][curr_col] == color:
                count += 1
                if count >= self._n_in_a_row:
                    return True
            else:
                count = 0

        return False


    def has_won(self, row, col, color):
        if self.is_n_in_a_row(row, col, color, 1, 0):
            return True
        elif self.is_n_in_a_row(row, col, color, 0, 1):
            return True
        elif self.is_n_in_a_row(row, col, color, 1, 1):
            return True
        elif self.is_n_in_a_row(row, col, color, 1, -1):
            return True
        else: 
            return False


    def is_game_over(self):
        if len(self._moves) == 0:
            return False

        last_move = self._moves[-1]
        last_color = 1 if last_move[2] == 'X' else -1
        if self.has_won(last_move[0], last_move[1], last_color):
            return True

        if len(self.all_valid_moves()) > 0:
            return False
        else:
            return True
        
    def get_winner(self):

        if not self.is_game_over():
            return None
        
        last_move = self._moves[-1]
        last_color = 1 if last_move[2] == 'X' else -1
        if self.has_won(last_move[0], last_move[1], last_color):
            return 'X' if last_move[2] == 'X' else 'O'
        else:
            return None
        

    # return  1.0  if 1(X) won
    # return -1.0  if 2(O) won
    # return  0.0  if otherwise
    def game_score(self):
        if not self.is_game_over():
            return 0.0

        if len(self._moves) == 0:
            return 0.0

        last_move = self._moves[-1]
        last_color = 1 if last_move[2] == 'X' else -1
        if self.has_won(last_move[0], last_move[1], last_color):
            if last_color == 1:
                return 1.0
            elif last_color == -1:
                return -1.0
            else:
                return 0.0

        return 0.0


def play_game(game, display=False):
    count = 0
    for i in range(150):
        valid_moves = game.all_valid_moves()
        if len(valid_moves) <= 0:
            break
        move = random.choice(valid_moves)
        count += 1
        if game.make_move(move):
            #print("Winning Move :", move)
            break # break if winning move

    if display:
        game.display()

        if game.is_game_over():
            print("Game Over, Score:", game.game_score())
        else:
            print("Game Draw, Score:", game.game_score())

        for move in game.moves():
            print(move)

        print(game.string())
        #print(game._state())
        #print(game._state_inversed())
        print(game.state_normalized())
        print("Winner [%s]" % game.get_winner())

        

game = None

def test_game():

    global game
    
    args = parse_args()
    
    game = Game(args)
    play_game(game.clone(), True)

    # timeit
    t = timeit.timeit("play_game(game.clone(), False)", globals=globals(), number=args.timeit_count)
    print("timeit : %s ms  [%d] [total %s s]" % (round((t/args.timeit_count)*1000,2), args.timeit_count, round(t,2)))

    #move = random.choice(game.all_valid_moves())
    #game.make_move(move[0], move[1], 1)
    
        
if __name__ == "__main__":
    test_game()
