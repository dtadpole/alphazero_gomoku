#!/usr/bin/env python

import math
import random
import asyncio
import numpy as np
from datetime import datetime
from args import parse_args, AsyncNone
from game import Game
from model import Model


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

        
        
class TreeNode:
    
    def __init__(self, parent, args, prior_p):
        self._parent = parent
        self._args   = args
        self._children = {}  # a map from action to TreeNode
        self._N = 0
        self._P = prior_p
        self._Q = 0
        self._U = 0
        self._V = 0 # predicted Q value

    def is_leaf(self):
        return len(self._children) == 0
    
    def is_root(self):
        return parent == None
    
    def get_visits(self):
        return self._N
    
    def get_value(self, cpuct, turn):
        if self._parent is not None:
            self._U = cpuct * self._P * math.sqrt(self._parent.get_visits())/(1+self.get_visits())
        if self._args.mcts_reverse_q:
            return self._Q + self._U
        else:
            return (self._Q + self._U) if turn == 'X' else (-self._Q + self._U)
    
    def update(self, value):
        self._Q = (self._Q * self._N + value) / (self._N + 1)
        self._N += 1
        
    def update_recursive(self, value):
        self.update(value)
        if self._parent is not None:
            if self._args.mcts_reverse_q:
                self._parent.update_recursive(-value*self._args.mcts_gamma)
                #self._parent.update_recursive(-value)
            else:
                self._parent.update_recursive(value*self._args.mcts_gamma)
                #self._parent.update_recursive(value)

    def expand(self, action_probs_zip):
        for action, prob in action_probs_zip:
            if action not in self._children:
                self._children[action] = TreeNode(self, self._args, prob)

    def select(self, cpuct, turn):
        #dirichlet = np.random.dirichlet([self._args.mcts_alpha]*len(self._children))
        #a_p_q_u_n = [(n[0], n[1]._P, n[1]._Q,
        #              cpuct * ((n[1]._P * (1-self._args.mcts_noise)) + (d * self._args.mcts_noise)) \
        #              * math.sqrt(n[1]._parent.get_visits())/(1+n[1].get_visits()), \
        #              n[1]) \
        #             for n, d in zip(self._children.items(), dirichlet)]
        ##print(a_p_q_u_n)
        #selected = max(a_p_q_u_n, key=lambda item: item[2] + item[3])
        #return (selected[0], selected[4])
        return max(self._children.items(),
                   key=lambda action_node: action_node[1].get_value(cpuct, turn))

            

class MCTS:
    
    def __init__(self, model, args):
        
        self._model = model
        self._args  = args

        self._root  = TreeNode(None, self._args, 1.0)


    def _playout(self, game):
        
        node = self._root
        
        while True:
            if node.is_leaf():
                break
            # select next move.
            action, node = node.select(self._args.mcts_cpuct, game.turn())
            game.make_action(action)

        if game.is_game_over():
            if self._args.mcts_reverse_q:
                leaf_value = 1.0 if game.game_score() != 0 else 0
            else:
                leaf_value = game.game_score()
        else:
            if self._args.model_tensorrt:
                action_probs, leaf_value = self._model.game_predict_trt(game.state_normalized())
            else:
                action_probs, leaf_value = self._model.game_predict(game.state_normalized())

            leaf_value = leaf_value[0]
            valid_actions = game.all_valid_actions()
            dirichlet = np.random.dirichlet([self._args.mcts_alpha]*len(valid_actions))
            action_probs_zip = zip(valid_actions,
                                   action_probs[valid_actions]*(1.0-self._args.mcts_noise)+dirichlet*self._args.mcts_noise)
            #action_probs_zip = zip(valid_actions, action_probs)
            #print(action_probs_zip)
            node.expand(action_probs_zip)

        node._V = leaf_value

        node.update_recursive(leaf_value)
        
    async def get_action_probs(self, game, temperature=1.0, sleep=None):
        
        for i in range(self._args.mcts_playout):
            try:
                self._playout(game.clone())
            finally:
                if (sleep is not None) and (i % self._args.mcts_sleep_int == 0):
                    await asyncio.sleep(sleep)
                else:
                    await AsyncNone()

        if self._args.mcts_display:
            p_matrix     = [-9.0]*game.action_size()
            visit_matrix = [-9]*game.action_size()
            q_matrix     = [-9.0]*game.action_size()
            p_q_matrix   = [-9.0]*game.action_size()
            for action, node in self._root._children.items():
                p_matrix[action]     = round(node._P, 2)
                visit_matrix[action] = node.get_visits()
                q_matrix[action]     = round(node._Q, 2)
                p_q_matrix[action]   = round(node._V, 2)
            print("Prior :\n%s" % np.reshape(p_matrix, game.size()))
            print("visits :\n%s" % np.reshape(visit_matrix, game.size()))
            print("Q :\n%s" % np.reshape(q_matrix, game.size()))
            print("predicted Q :\n%s" % np.reshape(p_q_matrix, game.size()))

        action_visits = [(action, node.get_visits(), node._Q, node._P) for action, node in self._root._children.items()]
        actions, visits, Qs, Ps = zip(*action_visits)  # Qs is MCTS value, Ps is prior probability

        action_probs = softmax(1.0/temperature * np.log(np.array(visits) + 1e-10))

        # reverse V if mcts_reverse_q is set
        V = -self._root._V if self._args.mcts_reverse_q else self._root._V
        
        action_advs  = Qs - V                       # Adv(S, a) = Q(S, a) - V(S) ,  V is predicted State value
        
        action_logp = -action_probs * np.log(Ps)    # log probs of policy against MCTS simulation

        return actions, action_probs, action_advs, action_logp, V

    
    def update_with_action(self, last_action):
        if last_action in self._root._children:
            self._root = self._root._children[last_action]
            self._root._parent = None
        else:
            self._root = TreeNode(None, self._args, 1.0)


    async def step(self, game, step, temperature=1.0, sleep=None):

        if game.is_game_over():
            return []

        step_experiences = []

        actions, action_probs, action_advs, action_logp, pred_v = await self.get_action_probs(game, temperature, sleep=sleep)
        #print(len(actions), len(action_probs))
        #print(action_advs, cross_entropy)

        # add board symetries
        game_state = game.state_normalized()
        player = game.turn()
        
        prob_matrix = np.zeros(game.action_size())
        for action, prob in zip(actions, action_probs):
            prob_matrix[action] = prob
        prob_matrix = np.reshape(prob_matrix, game.size())
        
        advs_matrix = np.zeros(game.action_size())
        for action, advs in zip(actions, action_advs):
            advs_matrix[action] = advs
        advs_matrix = np.reshape(advs_matrix, game.size())
        
        logp_matrix = np.zeros(game.action_size())
        for action, logp in zip(actions, action_logp):
            logp_matrix[action] = logp
        logp_matrix = np.reshape(logp_matrix, game.size())


        step_experiences.append((game_state, pred_v,
                                 np.reshape(prob_matrix, game.action_size()),
                                 None, #np.reshape(advs_matrix, game.action_size()),
                                 None, #np.reshape(logp_matrix, game.action_size()),
                                 player, step))
        step_experiences.append((np.rot90(game_state, 1, axes=(1,2)), pred_v,
                                 np.reshape(np.rot90(prob_matrix, 1), game.action_size()),
                                 None, #np.reshape(np.rot90(advs_matrix, 1), game.action_size()),
                                 None, #np.reshape(np.rot90(logp_matrix, 1), game.action_size()),
                                 player, step))
        step_experiences.append((np.rot90(game_state, 2, axes=(1,2)), pred_v,
                                 np.reshape(np.rot90(prob_matrix, 2), game.action_size()),
                                 None, #np.reshape(np.rot90(advs_matrix, 2), game.action_size()),
                                 None, #np.reshape(np.rot90(logp_matrix, 2), game.action_size()),
                                 player, step))
        step_experiences.append((np.rot90(game_state, 3, axes=(1,2)), pred_v,
                                 np.reshape(np.rot90(prob_matrix, 3), game.action_size()),
                                 None, #np.reshape(np.rot90(advs_matrix, 3), game.action_size()),
                                 None, #np.reshape(np.rot90(logp_matrix, 3), game.action_size()),
                                 player, step))

        game_state_f = np.flip(game_state, 1)
        prob_matrix_f = np.flip(prob_matrix, 0)
        advs_matrix_f = np.flip(advs_matrix, 0)
        logp_matrix_f = np.flip(logp_matrix, 0)
        step_experiences.append((game_state_f, pred_v,
                                 np.reshape(prob_matrix_f, game.action_size()),
                                 None, #np.reshape(advs_matrix_f, game.action_size()),
                                 None, #np.reshape(logp_matrix_f, game.action_size()),
                                 player, step))
        step_experiences.append((np.rot90(game_state_f, 1, axes=(1,2)), pred_v,
                                 np.reshape(np.rot90(prob_matrix_f, 1), game.action_size()),
                                 None, #np.reshape(np.rot90(advs_matrix_f, 1), game.action_size()),
                                 None, #np.reshape(np.rot90(logp_matrix_f, 1), game.action_size()),
                                 player, step))
        step_experiences.append((np.rot90(game_state_f, 2, axes=(1,2)), pred_v,
                                 np.reshape(np.rot90(prob_matrix_f, 2), game.action_size()),
                                 None, #np.reshape(np.rot90(advs_matrix_f, 2), game.action_size()),
                                 None, #np.reshape(np.rot90(logp_matrix_f, 2), game.action_size()),
                                 player, step))
        step_experiences.append((np.rot90(game_state_f, 3, axes=(1,2)), pred_v,
                                 np.reshape(np.rot90(prob_matrix_f, 3), game.action_size()),
                                 None, #np.reshape(np.rot90(advs_matrix_f, 3), game.action_size()),
                                 None, #np.reshape(np.rot90(logp_matrix_f, 3), game.action_size()),
                                 player, step))


        #chosen_action = np.random.choice(
        #    actions,
        #    p=0.75*action_probs + 0.25*np.random.dirichlet(0.3*np.ones(len(action_probs)))
        #)
        chosen_action = np.random.choice(actions, p=action_probs)
        
        #print(chosen_action)
        
        game.make_action(chosen_action)
        self.update_with_action(chosen_action)

        if self._args.mcts_display:
            game.display()

        return step_experiences


    async def self_play(self, game, sleep=None):
        step = 0

        game_experiences = []

        while not game.is_game_over():

            try:
                
                step += 1

                start_time = datetime.now()

                step_experiences = await self.step(game, step, temperature=self._args.mcts_temperature, sleep=sleep)

                game_experiences.extend(step_experiences)

                end_time = datetime.now()

                if self._args.mcts_display:
                    print("[Step %d] Play Duration: %s" % (step, (end_time - start_time)))
                    print("-"*40)
                    
            finally:

                if sleep is not None:
                    await asyncio.sleep(sleep)
                else:
                    await AsyncNone()


        v = game.game_score()
        print("Game Score: %s [moves %d, experiences %d]" % (v, len(game.moves()), len(game_experiences)))
        
        if self._args.mcts_display:
            for move in game.moves():
                print(move)

        if v != 0:
            if self._args.mcts_reverse_q:
                result = [(row[0], row[1], row[2], row[3], row[4], v*(-1.0)**(row[5]=='X')*(self._args.mcts_gamma**(len(game.moves())-row[6]))) for row in game_experiences]
            else:
                result = [(row[0], row[1], row[2], row[3], row[4], v*(self._args.mcts_gamma**(len(game.moves())-row[6]))) for row in game_experiences]
            return result
        else:
            return []

        
async def test_mcts():

    args = parse_args()

    game  = Game(args)
    model = Model(game.state_shape(), game.action_size(), args)
    
    mcts  = MCTS(model, args)

    experiences = await mcts.self_play(game)
    for result in experiences:
        print(result[2])

    print("Game Score:", game.game_score())
    for move in game.moves():
        print(move)

        
if __name__ == "__main__":
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_mcts())

