#!/usr/bin/env python

import numpy as np
import os
import io
import random
import pickle
import traceback
import asyncio
import aiofiles
from datetime import datetime
from pickle import Pickler, Unpickler
from aiohttp import web
from args import parse_args, get_data_folder
from game import Game
from model import Model
from mcts_tree import MCTS


class Memory:
    
    def __init__(self, args):
        
        self._args = args

        game = Game(args)
        self._default_model = Model(game.state_shape(), game.action_size(), args)

        self._model = None
        
        self._memory_experiences = []
        
        self._game_count         = 0
        self._total_memory_size  = 0
        self._stats              = {}


    def start_memory_server(self):

        #print("start memory websockets server...")
        #
        #self._start_server = websockets.server.serve(self.ws_receive_data,
        #                                             self._args.memory_server,
        #                                             self._args.memory_port,
        #                                             max_size=2**22)
        #
        #asyncio.ensure_future(self._start_server)
        #
        #print("started memory websockets server...")

        print("start memory http server...")

        app = web.Application()
        app.add_routes([web.get('/model', self.http_handle_model),
                        web.post('/experiences', self.http_handle_experiences)])

        web.run_app(app, host=self._args.memory_server, port=self._args.memory_port)

        print("started memory http server...")


    async def http_handle_model(self, request):
        
        print("-"*30)
        print("handle_model %s" % request)
        base64_model = self._model.to_base64()
        return web.Response(text=base64_model)

    
    async def http_handle_experiences(self, request):

        try:
            
            print("-"*30)
            print("handle_experiences %s" % request)
            payload_io = io.BytesIO()
            while True:
                chunk = await request.content.read(2**16)
                if chunk:
                    #print("experiences chunk size %s" % len(chunk))
                    payload_io.write(chunk)
                else:
                    break

            payload = payload_io.getvalue()
            post_data = pickle.loads(payload)
            game_experiences = post_data['game_experiences']
            game_score = post_data['game_score']
            print("Received Game Experiences: %d [score: %s] [payload: %.2fM]" % (len(game_experiences), game_score, len(payload) / 1024.0 / 1024.0))

            self.add_play_experiences(game_experiences, game_score)

            return web.Response(text='ok')
        
        except Exception as e:
            
            print(e)
            
            traceback.print_exc()
            
            return web.Response(text=str(e))

    
    def get_game_count(self):
        return self._game_count
    
    
    def get_memory_experiences(self):
        return self._memory_experiences

    
    def add_play_experiences(self, game_experiences, game_score):

        self._game_count += 1
        self._stats['game_count'] = self._game_count
        
        if game_score> 0:
            self._stats['win']  = (self._stats['win']  + 1) if 'win'  in self._stats else 1
        elif game_score < 0:
            self._stats['lose'] = (self._stats['lose'] + 1) if 'lose' in self._stats else 1
        else:
            self._stats['draw'] = (self._stats['draw'] + 1) if 'draw' in self._stats else 1

        self._memory_experiences.extend(game_experiences)
        self._total_memory_size += len(game_experiences)
        self._stats['total_memory_size'] = self._total_memory_size

        # trim memory experience size to target size
        target_memory_size = max(self._args.memory_min, min(self._args.memory_max,
                                 int((self._total_memory_size - self._args.memory_min) / 2) \
                                                                  + self._args.memory_min))
        if len(self._memory_experiences) > target_memory_size:
            self._memory_experiences = self._memory_experiences[len(self._memory_experiences) - target_memory_size:]

        print("Memory Experience Size: %d [this %d, total %d]" % \
              (len(self._memory_experiences), len(game_experiences), self._total_memory_size))

        #self.save()
        

    def add_experience_trained(self, experiences_trained, batches_trained):
        # add experiences trained stats
        self._stats['experiences_trained'] = (self._stats['experiences_trained'] \
                                              + experiences_trained) if 'experiences_trained' in self._stats else experiences_trained
        self._stats['batches_trained'] = (self._stats['batches_trained'] \
                                          + batches_trained) if 'batches_trained' in self._stats else batches_trained
        

    def get_model(self):
        return self._model
    
    
    async def save(self):
        
        folder = get_data_folder(self._args)
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        try:
            model_filepath = folder + '/' + "model.data"
            self._model.save_model(model_filepath)
            print("Saved model [%d]" % self._model.model_size())
        except Exception as e:
            print(e)
            print("model NOT saved.")

        try:
            save_memory_experiences = self._memory_experiences.copy()
            save_stats = self._stats.copy()
            data_to_save = pickle.dumps([save_memory_experiences, self._stats])
            print("Data to Save [%.1fM]" % int(len(data_to_save) / 1024.0 / 1024.0))

            experience_filepath = folder + '/' + "experience.data"
            # backup the file
            if os.path.isfile(experience_filepath+'.bak'):
                os.rename(experience_filepath+'.bak', experience_filepath+'.bak2')
            if os.path.isfile(experience_filepath):
                os.rename(experience_filepath, experience_filepath+'.bak')
            async with aiofiles.open(experience_filepath, "wb+") as f:
                CHUNK_SIZE = 2**24
                buffer = io.BufferedReader(io.BytesIO(data_to_save), buffer_size=CHUNK_SIZE)
                while True:
                    chunk = buffer.read(CHUNK_SIZE)
                    #print("save chunk %d ..." % len(chunk))
                    if len(chunk) > 0:
                        await f.write(chunk)
                        #await asyncio.sleep(1)
                    else:
                        break
                f.closed

            print("Saved experiences: %d" % len(save_memory_experiences))
            print("Saved stats: %s" % save_stats)

        except Exception as e:
            print(e)
            traceback.print_exc()
            print("experiences NOT saved.")

            
    def load(self):

        folder = get_data_folder(self._args)
        if not os.path.exists(folder):
            os.makedirs(folder)
            print("model NOT loaded. [use default model]")
            self._model = self._default_model
            print("experiences NOT laoded. [use empty experiences]")
            self._memory_experiences = []
            self._stats              = {}
            return

        # load model
        model_filepath = folder + '/' + "model.data"
        if not os.path.exists(model_filepath):
            print("model NOT loaded. [use default model]")
            self._model = self._default_model

        try:
            self._model = self._default_model.load_model(model_filepath)
            if self._model is None:
                raise Exception("model NOT loaded")
            print("Loaded Model: %s" % self._model)
        except Exception as e:
            print(e)
            print("model NOT loaded. [use default model]")
            self._model = self._default_model

        # load experiences
        experience_filepath = folder + '/' + "experience.data"
        if (not os.path.exists(experience_filepath)) and (not os.path.exists(experience_filepath + '.bak')):
            print("experiences NOT laoded. [use empty experiences]")
            self._memory_experiences = []
            self._stats              = {}

        try:
            with open(experience_filepath, "rb") as f:
                loaded = Unpickler(f).load()
                self._memory_experiences = loaded[0]
                self._stats              = loaded[1] if len(loaded) > 1 else {}
                print("Loaded Experiences: %d" % len(self._memory_experiences))
                print("Loaded Game Stats: %s" % self._stats)
                self._game_count = self._stats['game_count'] if 'game_count' in self._stats else 0
                self._total_memory_size = self._stats['total_memory_size'] if 'total_memory_size' in self._stats else len(self._memory_experiences)

            f.closed

        except Exception as e:
            
            try:
                
                with open(experience_filepath + '.bak', "rb") as f:
                    loaded = Unpickler(f).load()
                    self._memory_experiences = loaded[0]
                    self._stats              = loaded[1] if len(loaded) > 1 else {}
                    print("Loaded Experiences: %d" % len(self._memory_experiences))
                    print("Loaded Game Stats: %s" % self._stats)
                    self._game_count = self._stats['game_count'] if 'game_count' in self._stats else 0
                    self._total_memory_size = self._stats['total_memory_size'] if 'total_memory_size' in self._stats else len(memory_experiences)

                f.closed
                
            except Exception as e:
                
                print(e)
                #traceback.print_exc()
                print("experiences NOT laoded. [use empty experiences]")
                self._memory_experiences = []
                self._stats              = {}
                
                
        print(self._model)
        



if __name__ == "__main__":
    
    args = parse_args()

    memory = Memory(args)
    
    memory.load()
    
    asyncio.ensure_future(memory.save())

    asyncio.ensure_future(memory.start_memory_server())

    asyncio.get_event_loop().run_forever()

