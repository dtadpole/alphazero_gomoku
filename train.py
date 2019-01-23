#!/usr/bin/env python

import numpy as np
import os
import random
import asyncio
import traceback
from datetime import datetime
from pickle import Pickler, Unpickler
from args import parse_args, AsyncNone
from game import Game
from model import Model
from mcts_tree import MCTS
from memory import Memory
from play import play



async def train(args, memory):

    last_trained_game = 0
    
    last_saved_game = 0

    while True:

        try:
            curr_game_count = memory.get_game_count()
            curr_experiences = memory.get_memory_experiences()

            if curr_game_count >= (last_trained_game + args.train_game_count) and len(curr_experiences) >= args.model_batch_size:

                train_start_time = datetime.now()

                print("#"*30 + (" Training Game # %d " % curr_game_count) + "#"*30)
                print("Start Training : [Last Trained %d, Curr %d]" % (last_trained_game, curr_game_count))
                
                batch_experiences = random.sample(curr_experiences, args.model_batch_size)
                memory.get_model().game_train(batch_experiences)
                
                memory.add_experience_trained(len(batch_experiences) * args.model_epochs,
                                              int(len(batch_experiences) / args.model_batch_size) * args.model_epochs)

                print("Memory Stats : %s" % memory._stats)

                train_end_time = datetime.now()
                print("Train Duration: %s" % (train_end_time - train_start_time))

                last_trained_game = curr_game_count

                # save model and experiences
                if curr_game_count >= last_saved_game + args.train_save_count:
                    await memory.save()
                    last_saved_game = curr_game_count

                print("#"*80)
                
            else:
                
                print("="*20 + (" Bypass Training : [Last %d, Curr %d] " % (last_trained_game, curr_game_count)) + "="*20)
                
                await asyncio.sleep(2)
                
        except Exception as e:
            
            print(e)
            
            traceback.print_exc()

            asyncio.sleep(1)

        finally:

            await AsyncNone()


if __name__ == "__main__":

    # parse args
    args = parse_args()
    
    loop = asyncio.get_event_loop()
    
    # load memory
    memory = Memory(args)
    memory.load()
    
    # start play task
    if args.train_server:
        play_task = asyncio.ensure_future(play(args, memory=None, sleep=0.001), loop=loop)
    else:
        play_task = asyncio.ensure_future(play(args, memory=memory, sleep=None), loop=loop)
        

    # start train task
    train_task = asyncio.ensure_future(train(args, memory), loop=loop)

    if args.train_server:
        # start memory server
        memory_task = asyncio.ensure_future(memory.start_memory_server(), loop=loop)

    # start asyncio loop
    loop.run_forever()

