#!/usr/bin/env python

import numpy as np
import aiohttp
import asyncio
import os
import sys
import random
import pickle
import traceback
from datetime import datetime
from pickle import Pickler, Unpickler
from args import parse_args, AsyncNone, get_data_folder, get_model_filepath, get_best_model_filepath
from game import Game
from model import Model
from mcts_tree import MCTS
from memory import Memory



async def get_model_online(args):
    
    default_game = Game(args)
    default_model = Model(default_game.state_shape(), default_game.action_size(), args)
    
    model_url = "http://" + args.memory_server + ":" + args.memory_port + "/model"
    async with aiohttp.ClientSession() as session:
        async with session.get(model_url) as response:
            base64_model = await response.text()
            #print(base64_model)
            model = default_model.from_base64(base64_model)
            return model


async def load_model_by_type(model_type, args):
    
    default_game = Game(args)
    default_model = Model(default_game.state_shape(), default_game.action_size(), args)

    if model_type == "best":

        try:
            model = default_model.load_model(get_best_model_filepath(args))
            if model is not None:
                return model
            else:
                raise Exception("File not found: %s" % get_best_model_filepath(args))
        except Exception as e:
            print(e)
            try:
                model = default_model.load_model(get_best_model_filepath(args) + '.bak')
                if model is not None:
                    return model
                else:
                    raise Exception("File not found: %s" % (get_best_model_filepath(args) + '.bak'))
            except Exception as e:
                print(e)
                return default_model
            
    elif model_type == "curr":
        
        try:
            model = default_model.load_model(get_model_filepath(args))
            if model is not None:
                return model
            else:
                raise Exception("File not found: %s" % get_model_filepath(args))
        except Exception as e:
            print(e)
            try:
                model = default_model.load_model(get_model_filepath(args) + '.bak')
                if model is not None:
                    return model
                else:
                    raise Exception("File not found: %s" % (get_model_filepath(args) + '.bak'))
            except Exception as e:
                print(e)
                return default_model
            
    elif model_type == "online":
        
        try:
            model = await get_model_online(args)
            return model
        except Exception as e:
            print(e)
            return default_model
    
    elif model_type == "human":
        
        return None
        
    else:
        
        raise Exception("Unknown model type : %s (must be one of 'best', 'curr', 'human', 'online')" % model_type)


def save_curr_model_as_best(args, curr_model):
    
    curr_model.save_model(get_best_model_filepath(args))
    
    print("Best Model Saved to %s" % get_best_model_filepath(args))


    
def get_move_from_user(game):
    
    while True:

        try:
            
            move = input("What is your move? ")
            move = move.upper().strip()

            col = ord(move[0])-ord('A')
            row = game.size()[0] - int(move[1:])
            
            if game.is_valid_move(row, col):
                return (row, col)
            else:
                print("Invalid Move : %s" % move)
            
        except Exception as e:
            
            print(e)
            #traceback.print_exc()
    
    
async def evaluate(args):

    #args.mcts_noise = 0.0
    #args.mcts_alpha = 1.0
    
    players = {
        "p1" : {
            "type": args.eval_model_1,
            "model": await load_model_by_type(args.eval_model_1, args),
            "win_count": 0
        },
        "p2" : {
            "type": args.eval_model_2,
            "model": await load_model_by_type(args.eval_model_2, args),
            "win_count": 0
        },
        "draw_count" : 0
    }


    eval_game_count = 1
    
    for i in range(args.eval_game_count):

        try:
            
            print("#"*20 + (" Game # %d [ %s vs %s ] " % (eval_game_count, args.eval_model_1, args.eval_model_2)) + "#"*20)

            game = Game(args)
            game.display()
            
            step = 1
            
            while not game.is_game_over():
                
                print("-"*30 + " Game %d | Step %d  [ %s : %d ]   [ %s : %d ]   [ draw : %d ] " % (eval_game_count, step,
                                                                                                   players["p1"]["type"], 
                                                                                                   players["p1"]["win_count"],
                                                                                                   players["p2"]["type"],
                                                                                                   players["p2"]["win_count"],
                                                                                                   players["draw_count"]
                                                                                                   ) + "-"*30)
                step_start_time = datetime.now()

                curr_player = players["p1"] if (eval_game_count + step) % 2 == 0 else players["p2"]
                #print(curr_player)
                
                if curr_player['model'] is None:
                    move = get_move_from_user(game)
                    game.make_move(move)
                else:
                    mcts = MCTS(curr_player['model'], args)
                    await mcts.step(game, step, temperature=1e-3)
                    
                game.display()

                print("-"*50)
                step_end_time = datetime.now()
                print("Step Duration: %s" % (step_end_time - step_start_time))
                
                step += 1

            # game over - print game score and collect stats
            print("Game Score: %s" % game.game_score())
            
            if game.game_score() > 0:
                
                if (eval_game_count - 1) % 2 == 0:
                
                    players["p1"]["win_count"] += 1
                    
                else:
                    
                    players["p2"]["win_count"] += 1
                    
            elif game.game_score() < 0:
                
                if (eval_game_count - 1) % 2 == 0:
                    
                    players["p2"]["win_count"] += 1
                    
                else:
                    
                    players["p1"]["win_count"] += 1
                    
            else:
                
                players["draw_count"] += 1
                    
            print("-"*70)
            print("Total %d Games.   [ %s : %d ]   [ %s : %d ]   [ draw : %d ]" % (eval_game_count,
                                                                                   players["p1"]["type"], 
                                                                                   players["p1"]["win_count"],
                                                                                   players["p2"]["type"],
                                                                                   players["p2"]["win_count"],
                                                                                   players["draw_count"]))
            print("-"*70)
            
            if (players["p1"]["type"] == "human" or players["p2"]["type"] == "human") and i < args.eval_game_count - 1:
                
                response = input("Continue? [Y/n] ")
                response = response.upper().strip()
                if response == 'N' or response == "NO":
                    asyncio.get_event_loop().stop()
                    #sys.exit()

        except Exception as e:
            
            print(e)
            
            traceback.print_exc()
            
            await asyncio.sleep(1)
            
        finally:
            
            await asyncio.sleep(1)
            
            eval_game_count += 1

            
    try:

        # record best model if threshold exceeded
        if players["p1"]["type"] == "best" and players["p2"]["type"] == "curr":
            best_win = players["p1"]["win_count"]
            curr_win = players["p2"]["win_count"]
            if curr_win > args.eval_best_threshold * eval_game_count:
                print("New BEST Model Found !!! \tCurr vs Best : [ %d vs %d ]" % (curr_win, best_win))
                save_curr_model_as_best(args, players["p2"]["model"])

        if players["p1"]["type"] == "curr" and players["p2"]["type"] == "best":
            curr_win = players["p1"]["win_count"]
            best_win = players["p2"]["win_count"]
            if curr_win > args.eval_best_threshold * eval_game_count:
                print("New BEST Model Found !!!  [ Curr vs Best : %d vs %d ]" % (curr_win, best_win))
                save_curr_model_as_best(args, players["p1"]["model"])
                
    except Exception as e:
        
        print(e)
        traceback.print_exc()

    print("End of Games")
    asyncio.get_event_loop().stop()
    #sys.exit()


if __name__ == "__main__":

    args = parse_args()
    
    play_task = asyncio.ensure_future(evaluate(args))
    
    asyncio.get_event_loop().run_forever()

