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
        
        self.i=0
   
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
        score = 0
        n_merged = 0
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
                score += boardstate[i]
                n_merged += 1
            if i != id:
                move_log.append((id, i, gone))
        if self.action_success[n]:
            self.prev_boardstate[n] = prev_boardstate_buffer
        
        return score, n_merged
    def moveDownEvent(self, n):
        score = 0
        n_merged = 0
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
                score += boardstate[i]
                n_merged += 1
            if i != id:
                move_log.append((id, i, gone))
        if self.action_success[n]:
            self.prev_boardstate[n] = prev_boardstate_buffer
        return score, n_merged

    def moveLeftEvent(self, n):
        score = 0
        n_merged = 0
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
                score += boardstate[i]
                n_merged += 1
            if i != id:
                move_log.append((id, i, gone))
        if self.action_success[n]:
            self.prev_boardstate[n] = prev_boardstate_buffer
        return score, n_merged
    
    def moveRightEvent(self, n):
        score = 0
        n_merged = 0
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
                score += boardstate[i]
                n_merged += 1
            if i != id:
                move_log.append((id, i, gone))
        if self.action_success[n]:
            self.prev_boardstate[n] = prev_boardstate_buffer
        return score, n_merged
    def step(self, actions):
        score = torch.zeros(self.n_parallel)
        n_merged = torch.zeros(self.n_parallel)
        act_functions = [self.moveUpEvent, self.moveLeftEvent, self.moveDownEvent, self.moveRightEvent]
        self.action_success = [False for _ in range(self.n_parallel)]
        for n, action in enumerate(actions):
            score[n], n_merged[n] = act_functions[action](n)
        game_over_flag = self.__simulateGameOver()
        self.update_new_block(self.action_success)
        
        return {"state": self.boardstate.clone(), "score": score, "n_merged": n_merged, "done": game_over_flag}
        
    def __simulateGameOver(self):
        """simulate move every direction to check whether game is overd or not
        """
        action_success_buffer = self.action_success.copy()
        boardstate_buffer = self.boardstate.clone()
        prev_boardstate_buffer = self.prev_boardstate.clone()
        free_block_buffer = [l.copy() for l in self.freeblocks]
        movable = [False] * self.n_parallel
        for simulate_function in [self.moveUpEvent, self.moveLeftEvent, self.moveDownEvent, self.moveRightEvent]:
            for i in range(self.n_parallel):
                if movable[i]:
                    continue
                simulate_function(i)
                movable[i] = movable[i] or self.action_success[i]
                
            ## roll back for next simulation
            self.action_success = action_success_buffer.copy()
            self.boardstate = boardstate_buffer.clone()
            self.prev_boardstate = prev_boardstate_buffer.clone()
            self.freeblocks = [l.copy() for l in free_block_buffer]
        
        game_over_flag = torch.tensor([1-g for g in movable])
        return game_over_flag
                
        
    

if __name__ == "__main__":
    env = Env2048(1, (4, 4))
    print(env.boardstate.reshape(1, 4, 4))
    for _ in range(1000):
        move = [random.choice([0, 1, 2, 3])]
        ans = env.step(move)
        ans["state"]= ans["state"].reshape(4, 4)
        ans["action"] = move
        ans["success"] = env.action_success
        print(ans)
    