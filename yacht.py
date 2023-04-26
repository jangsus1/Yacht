import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from tqdm.auto import tqdm
import pickle
from functools import reduce
import shutil
import os
from copy import deepcopy
from itertools import chain
import time
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import sys
original_stdout = sys.stdout # Save a reference to the original standard output


MODEL_NAME = "state_change"
MODEL_DIR = f'./assets/models/{MODEL_NAME}.pth'

   

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def load_model():
  state_dict = torch.load(MODEL_DIR)
  return (state_dict['model'], state_dict['optimizer'], state_dict['num_episodes'])


class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.num_board = nn.Linear(6+1, 1024)
        self.special_board = nn.Linear(6, 1024)
        self.dices = nn.Linear(5, 1024)
        self.dices_left = nn.Linear(1, 128)
        self.l1 = nn.Linear(1024*3+128, 4096*4)
        self.l2 = nn.Linear(4096*4, 4096*2)
        self.l3 = nn.Linear(4096*2, 4096)
        self.l4 = nn.Linear(4096, 2048)
        self.head = nn.Linear(2048, outputs)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = x.float().to(device)

        num_boards = x[:, 0:6].view(-1, 6)
        special_boards = x[:, 6:12].view(-1, 6)
        partial_score = x[:, 12].view(-1, 1)
        dices_left = x[:, 13].view(-1, 1)
        dices = x[:, 14:19].view(-1, 5)

        num_boards = self.leaky_relu(self.num_board(torch.cat((num_boards, partial_score), 1)))
        dices_left = self.leaky_relu(self.dices_left(dices_left))
        special_boards = self.leaky_relu(self.special_board(special_boards))
        dices = self.leaky_relu(self.dices(dices))
        
        x = torch.cat((num_boards, special_boards, dices, dices_left), 1)
        x = self.leaky_relu(self.l1(x))
        x = self.leaky_relu(self.l2(x))
        x = self.leaky_relu(self.l3(x))
        x = self.leaky_relu(self.l4(x))
        x = self.leaky_relu(self.head(x))
        return x
    

class Yacht():
  def __init__(self):
    self.board = {
        "ones" : {"fixed" : False, "value" : 0},
        "twos" : {"fixed" : False, "value" : 0},
        "threes" : {"fixed" : False, "value" : 0},
        "fours" : {"fixed" : False, "value" : 0},
        "fives" : {"fixed" : False, "value" : 0},
        "sixes" : {"fixed" : False, "value" : 0},
        "choice" : {"fixed" : False, "value" : 0},
        "four_of_a_kind" : {"fixed" : False, "value" : 0},
        "full_house" : {"fixed" : False, "value" : 0},
        "small_straight" : {"fixed" : False, "value" : 0},
        "large_straight" : {"fixed" : False, "value" : 0},
        "yacht" : {"fixed" : False, "value" : 0}
    }
    
    self.ACTION = []
    for n in range(32):
      a = n//16
      b = n%16//8
      c = n%8//4
      d = n%4//2
      e = n%2
      self.ACTION.append((a,b,c,d,e))
    self.ACTION += [n for n in range(12)]
    self.dices = [0,0,0,0,0]
    self.reset()

    
  
  def reset(self):
    self.bonus = False
    self.partial_sum = 0
    self.turn_left = 12
    self.dice_left = 2
    self.roll_dice((0,0,0,0,0))
    for key, value in self.board.items():
      value['fixed'] = False
      value['value'] = 0


  def get_state(self):
    state = []
    for name, plate in self.board.items():
      if plate['fixed']:
        state.append(-1)
        continue
      four, full, small, large, yacht = self.check_pattern()
      if name=='ones':
        value = reduce(lambda acc, cur: (acc+cur) if cur==1 else acc, self.dices, 0)
      elif name == 'twos':
        value = reduce(lambda acc, cur: (acc+cur) if cur==2 else acc, self.dices, 0)
      elif name == 'threes':
        value = reduce(lambda acc, cur: (acc+cur) if cur==3 else acc, self.dices, 0)
      elif name == 'fours':
        value = reduce(lambda acc, cur: (acc+cur) if cur==4 else acc, self.dices, 0)
      elif name == 'fives':
        value = reduce(lambda acc, cur: (acc+cur) if cur==5 else acc, self.dices, 0)
      elif name == 'sixes':
        value = reduce(lambda acc, cur: (acc+cur) if cur==6 else acc, self.dices, 0)
      elif name == 'choice':
        value = sum(self.dices)
      elif name == 'four_of_a_kind':
        value = sum(self.dices) if four else 0
      elif name == 'full_house':
        value = sum(self.dices) if full else 0
      elif name == 'small_straight':
        value = 15 if small else 0
      elif name == 'large_straight':
        value = 30 if large else 0
      elif name == 'yacht':
        value = 50 if yacht else 0
      else:
        raise Exception(f"Cannot Find {name} and {plate}")
      if name in ['ones', 'twos', 'threes', 'fours', 'fives', 'sixes']:
        if (not self.bonus) and self.partial_sum+value>=63:
          value+=35
      state.append(value)

    state.append(self.partial_sum)
    state.append(self.dice_left)
    return np.array(state + self.dices, dtype=np.float32)


  def play_manually(self, action):
    if type(action) == type((0,0,0,0,0)):
      action = sum([int(action[4-i])*(2**i) for i in range(5)])
      return self.play(action)
    else:
      return self.play(action+31)


  def roll_dice(self, action):
    for i in range(5):
        if not action[i]: self.dices[i] = random.randint(1,6)
    self.dices.sort(reverse=True)

  def check_pattern(self):
    full_house = True
    four_of_a_kind = True
    yacht = True

    counts = [0,0,0,0,0,0]
    for d in self.dices: counts[d-1]+=1

    for c in counts: 
      if c not in [0,1,4]: four_of_a_kind = False
      if c not in [0,2,3]: full_house = False
      if c not in [0,5]: yacht = False
    
    exists = [c>0 for c in counts]
    sum_exists = sum(exists)
    if sum_exists > 1 : yacht = False
    if sum_exists > 2 : four_of_a_kind = full_house = False

    large_straight = sum(exists)==5 and (not exists[0] or not exists[-1])
    small_straight = sum(exists)==4 and ((not exists[0] and not exists[1]) or (not exists[0] and not exists[-1]) or (not exists[-1] and not exists[-2]))


    four_of_a_kind = four_of_a_kind or yacht
    full_house = full_house or yacht
    small_straight = small_straight or large_straight

    return (four_of_a_kind, full_house, small_straight, large_straight, yacht)
  

  def fill_score(self, name, plate):
    total = sum(self.dices)
    plate['fixed'] = True
    four, full, small, large, yacht = self.check_pattern()
    if name=='ones':
      value = reduce(lambda acc, cur: (acc+cur) if cur==1 else acc, self.dices, 0)
    elif name == 'twos':
      value = reduce(lambda acc, cur: (acc+cur) if cur==2 else acc, self.dices, 0)
    elif name == 'threes':
      value = reduce(lambda acc, cur: (acc+cur) if cur==3 else acc, self.dices, 0)
    elif name == 'fours':
      value = reduce(lambda acc, cur: (acc+cur) if cur==4 else acc, self.dices, 0)
    elif name == 'fives':
      value = reduce(lambda acc, cur: (acc+cur) if cur==5 else acc, self.dices, 0)
    elif name == 'sixes':
      value = reduce(lambda acc, cur: (acc+cur) if cur==6 else acc, self.dices, 0)
    elif name == 'choice':
      value = sum(self.dices)
    elif name == 'four_of_a_kind':
      value = sum(self.dices) if four else 0
    elif name == 'full_house':
      value = sum(self.dices) if full else 0
    elif name == 'small_straight':
      value = 15 if small else 0
    elif name == 'large_straight':
      value = 30 if large else 0
    elif name == 'yacht':
      value = 50 if yacht else 0
    else:
      raise Exception(f"Cannot Find {name} and {plate}")
    plate['value'] = value
    if name in ['ones', 'twos', 'threes', 'fours', 'fives', 'sixes']:
      self.partial_sum+=value
    bonus_score = self.check_bonus()
    return value + bonus_score
  
  def check_bonus(self):
    if self.bonus : return 0
    if self.partial_sum>=63 :
      self.partial_sum = 63
      self.bonus = True
      return 35
    return 0


  def play(self, action):
    if self.turn_left == 0 : raise Exception("GAMEOVER")
    if action < 32:
      if self.dice_left<=0: raise Exception("Not Enough Dices Left")
      action = self.ACTION[action]
      self.roll_dice(action)
      self.dice_left -= 1
      reward = 0
    else:
      action = self.ACTION[action]
      name, plate = list(self.board.items())[action]
      if plate['fixed']: raise Exception("Score Board Already Fixed")
      reward = self.fill_score(name, plate)
      self.roll_dice((0,0,0,0,0))
      self.dice_left=2
      self.turn_left-=1
    done = self.turn_left==0
    next_state = None if done else self.get_state()
    return done, reward, next_state

def select_best_action(state):
    policy_net.eval()
    valid_action_mask = get_valid_action_mask(state.unsqueeze(0)).squeeze(0)
    invalid_action_mask = valid_action_mask == False
    with torch.no_grad():
    # t.max(1) will return largest column value of each row.
    # second column on max result is index of where max element was
    # found, so we pick action with the larger expected reward.
        batched_state = state.unsqueeze(0)
        expectation, action = policy_net(batched_state).masked_fill(invalid_action_mask, -1e20).max(1)
    return expectation.item(), action.item()
    

def get_valid_action_mask(states):
    mask = torch.zeros((states.shape[0],44), dtype=torch.bool, device=device)
    dice_left = states[:,13]
    mask[:,0:32] = (dice_left>0).unsqueeze(1)
    mask[:,32:44] = states[:,0:12]>-1
    return mask

SCORE_NORMALIZE = 65
PARTIAL_SUM_NORMALIZE = 63
DICE_NORMALIZE = 6
DICE_LEFT_NORMALIZE = 2

def normalize_reward(reward):
  return reward/SCORE_NORMALIZE

def normalize_state(state):
  new_state = []
  for i in range(12):
    new_state.append(state[i]/SCORE_NORMALIZE if state[i]>0 else state[i])
  new_state.append( state[12] / PARTIAL_SUM_NORMALIZE)
  new_state.append(state[13] / DICE_LEFT_NORMALIZE)
  for i in range(5):
    new_state.append(state[14+i] / DICE_NORMALIZE)
  return new_state


n_states = 19
n_actions = 44

policy_net = DQN(n_states, n_actions).to(device)
model_state, optimizer_state, loaded_episode = load_model()
policy_net.load_state_dict(model_state)



boardnames = ["Ones", "Twos", "Threes", "Fours", "Fives", "Sixes", "Choice", "Four of a Kind", "Full House", "Small Straight", "Large Straight", "Yacht"]
def print_state_beautiful(state):
    scoreboard = state[:12]
    partial_sum = state[12]
    dice_left = state[13]
    dice = state[14:]
    print("---------------------")
    for i, name, score in zip(count(), boardnames, scoreboard):
        if score < 0: 
            print(f"{i}. {name}: Fixed")
        else:
            print(f"{i}. {name}: {int(score)}")
    print("")
    print(f"Partial Sum: {int(partial_sum)}")
    print(f"Dice Left: {int(dice_left)}")
    print(f"Dices: {[int(d) for d in dice]}")
    print("---------------------")

def encode_action(raw_action):
    if "," in raw_action:
        raw_action = raw_action.replace("(", "")
        raw_action = raw_action.replace(")", "")
        raw_action = raw_action.replace(" ", "")
        dices = raw_action.split(",")
        dices = [int(d.strip()) for d in dices]
        return 16*dices[0] + 8*dices[1] + 4*dices[2] + 2*dices[3] + dices[4]
    else:
        return 32+int(raw_action.strip())

def decode_action(action):
    if action < 32:
        d1 = action // 16
        d2 = (action%16) // 8
        d3 = (action%8) // 4
        d4 = (action%4) // 2
        d5 = (action%2)
        return f"({d1}, {d2}, {d3}, {d4}, {d5})"
    else:
        action-=32
        return f"Fill in {boardnames[action]}"


def main():
    game = Yacht()
    print("Game Start!")
    game.reset()
    score = 0
    record = []
    for t in count():
        raw_state = game.get_state()
        print_state_beautiful(raw_state)
        
        state = torch.tensor(normalize_state(raw_state), device=device)
        expectation, best_action = select_best_action(state)
        expectation *= SCORE_NORMALIZE
        print(f"Best Action: {decode_action(best_action)}, Expectation: {expectation}")
        
        user_input = "" # input()
        if user_input == "":
            user_action = best_action
        else:
            user_action = encode_action(user_input)
        
        
        done, reward, next_state = game.play(user_action)
        score+=reward
        print(f"Reward = {reward}, Score = {score}")
        record.append((raw_state, best_action, expectation, reward))
        raw_state = next_state
        if done: break
    print("Game Over")
    return score, record


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for i in range(10000):
        f = open('temp.txt', 'w')
        sys.stdout = f
        score, record = main()
        folder = f"results/{score}/"
        os.makedirs(folder, exist_ok=True)
        file_count = len(glob(folder+'*.txt'))
        os.rename("temp.txt", folder+f'{file_count}.txt')
        f.close()
        with open(folder+f'{file_count}.pkl', "wb") as f:
           pickle.dump(record, f)
           
