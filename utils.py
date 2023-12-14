import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def anim(frames, titles=None, filename=None, show=True):
    plt.figure(figsize=(frames[0].shape[1]/72.0/4, frames[0].shape[0]/72.0/4), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
        if titles is not None:
            plt.title(titles[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    if filename is not None:
        anim.save(filename, writer="ffmpeg")
    if show:
        plt.show()

def normalize_state(state, observation_space):
    # 連続値の状態を[-1,1]の範囲に正規化
    if isinstance(state, dict):
        normalized_state = {}
        for key in state.keys():
            state_mean = 0.5 * (observation_space[key].high + observation_space[key].low)
            state_halfwidth = 0.5 * (observation_space[key].high - observation_space[key].low)
            normalized_state[key] = ((state[key].astype(np.float32) - state_mean) / state_halfwidth).astype(np.float32)
    else:
        state_mean = 0.5 * (observation_space.high + observation_space.low)
        state_halfwidth = 0.5 * (observation_space.high - observation_space.low)
        normalized_state = ((state.astype(np.float32) - state_mean) / state_halfwidth).astype(np.float32)
    return normalized_state

def obs2state(obs, observation_space, trans, image_list=['rgb_image', 'depth_image']):
    normalized_obs = normalize_state(obs, observation_space)
    image = normalized_obs[image_list[0]]
    for name in image_list[1:]:
        image = np.concatenate([image, normalized_obs[name]], axis=2)
    state = {'observation': normalized_obs['observation'], 'image': trans(image)}
    return state

def return_transition(state, next_state, reward, action, terminated, truncated):
    return {
        'state': state,
        'next_state': next_state,
        'reward': reward,
        'action': action,
        'success': int(terminated), # タスクの成功
        'done': int(terminated or truncated) # エピソードの終了（成功、失敗、エピソードの上限に達する）
    }

def yes_no_input(check):
    while True:
        print("\n### Please check \033[31m" + check + "\033[0m ###")
        choice = input("Are you sure you want to continue running this code? [y/N]: ").lower()
        if choice in ['y', 'ye', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
