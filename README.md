# AlphaZero Gomoku

This is an implementation of Gomoku game with AlphaZero, a.k.a. Five in a Row

The code is built on python3, with pytorch and asyncio.

This implementation supports multiple asynchronous inference servers. Tensorrt inference is an option.

---

To train model with a single process using CPU:

<code>python train.py</code>

---

To train model with a single process using GPU (0 is the GPU id):

<code>python train.py --model_cuda 0</code>

---

To train model with distributed processes (1 training server, multiple inference servers):

<code>python train.py --model_cuda 0 [--memory_server 127.0.0.1 --memory_port 8080]</code>

<code>python play.py --model_cuda 0 [--memory_server 127.0.0.1 --memory_port 8080]</code>

<code>python play.py --model_cuda 1 [--memory_server 127.0.0.1 --memory_port 8080]</code>

---

Trained models are stored in the following directory by default:

<code>./save_<model_arch>_<q_type>_<state_type>_<channels>_<game_size>/</code>
  
---

To evaluate training model or play with human:

<code>python evaluate.py --mcts_model_1 (best|curr|human) --mcts_model_2 (best|curr|human)</code>

---

To show a list of options:

<code>python args.py --help</code>

---  
