import numpy as np
import random
import torch

GENERATE4PROB = 0.25

class Env2048:
    def __init__(self, n_parallel, board_size):
        ## state variables
        self.state_dim = board_size[0]*board_size[1]
        self.boardstate = torch.ones(n_parallel, self.state_dim) * -1
        self.prev_boardstate = torch.ones(n_parallel, self.state_dim) * -1
        self.freeblocks = [list(range(self.state_dim)) for _ in range(n_parallel)]
        self.scores = torch.zeros(n_parallel)
        self.gameoverflag = torch.zeros(n_parallel)
        
        self.n_parallel = n_parallel
        self.move_logs = [[] for _ in range(n_parallel)]
        self.board_size = board_size # (n_row, n_col)
        
        self.update_new_block(update_flag=[True] * n_parallel)
        self.update_new_block(update_flag=[True] * n_parallel)
   
    def update_new_block(self, update_flag, new_id=None):
        assert new_id==None
        for i in range(self.n_parallel):
            if update_flag[i] == False:
                continue
            boardstate = self.boardstate[i]
            new_id = random.choice(self.freeblocks[i])
            freeblock = self.freeblocks[i]
            assert boardstate[new_id] == -1
            boardstate[new_id] = 4 if random.random() < GENERATE4PROB else 2
            freeblock.remove(new_id)
            
    def moveUpEvent(self, n):
        boardstate = self.boardstate[n]
        freeblocks = self.freeblocks[n]
        move_log = self.move_logs[n]
        
        prev_boardstate_buffer = boardstate.clone()
        integrated_blocks = []
        row, col = self.board_size
        scan_path = list(range(col, row*col))
        for id in scan_path:
            gone = False
            if boardstate[id]==-1:
                continue
            i = id
            while i-4 >= 0 and boardstate[i-4] == -1:
                boardstate[i-4] = boardstate[i]
                boardstate[i] = -1
                freeblocks.remove(i-4)
                freeblocks.append(i)
                self.action_success[n] = True
                i -= 4
            if i-4 >= 0 and (i-4 not in integrated_blocks) and boardstate[i] == boardstate[i-4]:
                boardstate[i-4] *= 2
                boardstate[i] = -1
                freeblocks.append(i)
                self.action_success[n] = True
                i -= 4
                gone = True
                integrated_blocks.append(i)
            if i != id:
                move_log.append((id, i, gone))
        if self.action_success[n]:
            self.prev_boardstate[n] = prev_boardstate_buffer
    def moveDownEvent(self, n):
        boardstate = self.boardstate[n]
        freeblocks = self.freeblocks[n]
        move_log = self.move_logs[n]
        
        prev_boardstate_buffer = boardstate.clone()
        integrated_blocks = []
        row, col = self.board_size
        scan_path = list(range(row*col-col-1, -1, -1))
        for id in scan_path:
            gone = False
            if boardstate[id]==-1:
                continue
            i = id
            while i+4 <16 and boardstate[i+4] == -1:
                boardstate[i+4] = boardstate[i]
                boardstate[i] = -1
                freeblocks.remove(i+4)
                freeblocks.append(i)
                self.action_success[n] = True
                i += 4
            if i+4 <16 and (i+4 not in integrated_blocks) and boardstate[i] == boardstate[i+4]:
                boardstate[i+4] *= 2
                boardstate[i] = -1
                freeblocks.append(i)
                self.action_success[n] = True
                i += 4
                gone = True
                integrated_blocks.append(i)
            if i != id:
                move_log.append((id, i, gone))
        if self.action_success[n]:
            self.prev_boardstate[n] = prev_boardstate_buffer

    def moveLeftEvent(self, n):
        boardstate = self.boardstate[n]
        freeblocks = self.freeblocks[n]
        move_log = self.move_logs[n]
        
        prev_boardstate_buffer = boardstate.clone()
        integrated_blocks = []
        row, col = self.board_size
        scan_path = [c+r*col for c in range(1, col) for r in range(row)]
        for id in scan_path:
            gone = False
            if boardstate[id]==-1:
                continue
            i = id
            while (i-1)%4 != 3 and boardstate[i-1] == -1:
                boardstate[i-1] = boardstate[i]
                boardstate[i] = -1
                freeblocks.remove(i-1)
                freeblocks.append(i)
                self.action_success[n] = True
                i -= 1
            if i%4 != 0 and (i-1 not in integrated_blocks) and boardstate[i] == boardstate[i-1]:
                boardstate[i-1] *= 2
                boardstate[i] = -1
                freeblocks.append(i)
                self.action_success[n] = True
                i -= 1
                gone = True
                integrated_blocks.append(i)
            if i != id:
                move_log.append((id, i, gone))
        if self.action_success[n]:
            self.prev_boardstate[n] = prev_boardstate_buffer
                
    def moveRightEvent(self, n):
        boardstate = self.boardstate[n]
        freeblocks = self.freeblocks[n]
        move_log = self.move_logs[n]
        
        prev_boardstate_buffer = boardstate.clone()
        integrated_blocks = []
        row, col = self.board_size
        scan_path = [c+r*col for c in range(col-2, -1, -1) for r in range(row)]
        for id in scan_path:
            gone = False
            if boardstate[id]==-1:
                continue
            i = id
            while (i+1)%4 != 0 and boardstate[i+1] == -1:
                boardstate[i+1] = boardstate[i]
                boardstate[i] = -1
                freeblocks.remove(i+1)
                freeblocks.append(i)
                self.action_success[n] = True
                i += 1
            if i%4 != 3 and (i+1 not in integrated_blocks) and boardstate[i] == boardstate[i+1]:
                boardstate[i+1] *= 2
                boardstate[i] = -1
                freeblocks.append(i)
                self.action_success[n] = True
                i += 1
                gone = True
                integrated_blocks.append(i)
            if i != id:
                move_log.append((id, i, gone))
        if self.action_success[n]:
            self.prev_boardstate[n] = prev_boardstate_buffer
    
    def step(self, actions):
        act_functions = [self.moveUpEvent, self.moveLeftEvent, self.moveDownEvent, self.moveRightEvent]
        self.action_success = [False for _ in range(self.n_parallel)]
        for n, action in enumerate(actions):
            act_functions[action](n)
        self.update_new_block(self.action_success)
        
        
        
        
    

if __name__ == "__main__":
    env = Env2048(4, (4, 4))
    print(env.boardstate.reshape(4, 4, 4))
    env.step([0, 1, 2, 3])
    print(env.boardstate.reshape(4, 4, 4))
    