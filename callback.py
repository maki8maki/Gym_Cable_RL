import os

import numpy as np
from gymnasium import Env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

from utils import anim


class MyEvalVallback(EvalCallback):
    def __init__(
        self,
        eval_env: Env | VecEnv,
        callback_on_new_best: BaseCallback | None = None,
        callback_after_eval: BaseCallback | None = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str | None = None,
        best_model_save_filenames: list[str] | None = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=None,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )

        self.best_model_save_filenames = best_model_save_filenames

    def _init_callback(self) -> None:
        super()._init_callback()

        if self.best_model_save_filenames is not None:
            for path in self.best_model_save_filenames:
                os.makedirs(os.path.dirname(path), exist_ok=True)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_filenames is not None:
                    for path in self.best_model_save_filenames:
                        self.model.save(path)
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class VideoRecordCallback(BaseCallback):
    def __init__(
        self,
        env: Env,
        save_freq: int,
        save_filename: str,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.env = env
        self.save_freq = save_freq
        self.save_filename = save_filename
        self.deterministic = deterministic
        self.images = []
        self.titles = []

    def _init_callback(self) -> None:
        super()._init_callback()
        os.makedirs(os.path.dirname(self.save_filename), exist_ok=True)

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            obs, _ = self.env.reset()
            self.images.append(self.env.render())
            self.titles.append(f"Step {self.num_timesteps}")
            while True:
                ac, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, _, terminated, truncated, _ = self.env.step(ac)
                self.images.append(self.env.render())
                self.titles.append(f"Step {self.num_timesteps}")
                if terminated or truncated:
                    break
        return True

    def _on_training_end(self) -> None:
        anim(self.images, self.titles, self.save_filename, show=False)
