#!/usr/bin/env python

import numpy as np
import aiohttp
import asyncio
import os
import random
import pickle
import traceback
from datetime import datetime
from pickle import Pickler, Unpickler
from args import parse_args, AsyncNone
from game import Game
from model import Model
from mcts_tree import MCTS
from memory import Memory


async def get_model(args):
    
    game = Game(args)
    model = Model(game.state_shape(), game.action_size(), args)
    
    model_url = "http://" + args.memory_server + ":" + args.memory_port + "/model"
    async with aiohttp.ClientSession() as session:
        async with session.get(model_url) as response:
            base64_model = await response.text()
            #print(base64_model)
            model = model.from_base64(base64_model)
            return model


async def post_experiences(args, game_experiences, game_score):
    
    print("Posting Experiences ...")
    experiences_url = "http://" + args.memory_server + ":" + args.memory_port + "/experiences"
    async with aiohttp.ClientSession() as session:
        post_data = pickle.dumps({
            'game_experiences': game_experiences,
            'game_score': game_score
        }, protocol=2)
        async with session.post(experiences_url, data=post_data) as response:
            response_text = await(response.text())
            print("Post Experiences [%s]" % response_text)
            return

        
# record local game count
local_game_count = 1
        
async def play(args, memory=None, sleep=None):

    global local_game_count
    
    model = None
    
    while True:

        try:

            print("#"*20 + (" Local Game # %d " % local_game_count) + "#"*20)

            if memory is None:
                if (model is None) or (local_game_count % args.play_model_refresh == 0):
                    print("Loading Model ...")
                    model = await get_model(args)
                    print("Loaded %s" % model)
                    #print("here2")
                    #print("Loaded Model")

            else:
                print("Loading Model ...")
                model = memory.get_model()
                model.build_trt_engine(model.state_dict(), np.dtype(args.play_dtype))
                print("Loaded %s" % model)
                #print("here1")
                
            game = Game(args)
            mcts = MCTS(model, args)

            # play game
            play_start_time = datetime.now()
            print("Playing Game ...")
            game_experiences = await mcts.self_play(game, sleep)
            play_end_time = datetime.now()
            print("Game Score [%s], Play Duration: %s" % (game.game_score(), (play_end_time - play_start_time)))

            local_game_count += 1

            if args.play_post_exp:
                if memory is None:
                    await post_experiences(args, game_experiences, game.game_score())
                else:
                    memory.add_play_experiences(game_experiences, game.game_score())
            else:
                print("Skip Post Experience.")

        except Exception as e:
            
            model = None

            print(e)
            
            traceback.print_exc()
            
            await asyncio.sleep(1)
            
        finally:
            
            await AsyncNone()



if __name__ == "__main__":

    args = parse_args()
    
    play_task = asyncio.ensure_future(play(args))
    
    asyncio.get_event_loop().run_forever()

