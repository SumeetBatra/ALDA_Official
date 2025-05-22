import logging
import os
import torch
import torch.nn.functional as F
import numpy as np
import wandb
import einops

from disentangle.utils.metrics import log_reconstruction_metrics, peak_signal_to_noise_ratio
from trainers.trainer_base import TrainerBase
from dmcontrol_generalization_benchmark.src.env.wrappers import FrameStack, DMCObsWrapper, ColorWrapper
from dmcontrol_generalization_benchmark.src.env.dmc2gym import dmc2gym
from dmcontrol_generalization_benchmark.src import utils
from autoencoders.quantized_autoencoder import QuantizedEncoder, QuantizedDecoder, OuterEncoder
from disentangle.latents.associative import AssociativeLatent
from models.sac import Actor, Critic, RLProjection, HistoryEncoder, weight_init
from common.utils import VideoRecorder, eval_mode, GradMagnitudeTracker
from typing import Dict, Any
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm

logger = logging.getLogger('alda')
logger.setLevel(logging.INFO)


class ALDA(TrainerBase):
    # SAC params
    encoder_tau: float = 0.05
    hidden_dim: int = 1024

    log_grad_norm: bool = False

    # actor params
    actor_lr: float = 1e-3
    actor_beta: float = 0.9
    actor_log_std_min: float = -10
    actor_log_std_max: float = 2
    actor_update_freq: int = 2

    # critic params
    critic_lr: float = 1e-3
    critic_beta: float = 0.9
    critic_tau: float = 0.01
    critic_target_update_freq: int = 2

    # agent
    n_train_steps: int = 1_000_000
    discount: float = 0.99
    init_steps: int = 1000
    batch_size: int = 128
    buffer_capacity: int = 1_000_000
    save_buffer: bool = False

    # entropy maximization
    init_temperature: float = 0.1
    alpha_lr: float = 1e-4
    alpha_beta: float = 0.5

    # encoder
    embedding_size: int = 256
    encoder_lr: float = 1e-3

    # ALDA
    num_latents: int = 12
    values_per_latent: int = 12
    beta: float = 100.
    use_quant_loss: bool = False

    # eval and saving
    eval_n_steps: int = 10_000
    checkpoint_n_steps: int = 50_000
    n_eval_episodes: int = 10
    log_every: int = 500

    def __init__(self, **kwargs):
        super(ALDA, self).__init__(**kwargs)
        ALDA.set_attributes(self, kwargs)

        self.env = None
        self.env_name = None
        self.spec = None
        self.distract_env = None
        self.color_env = None

        # env params
        self.obs_shape = None
        self.action_size = None
        self.video_recorder = None

        self.buffer = None

        # models
        self.actor_encoder = None
        self.critic_encoder = None
        self.decoder = None
        self.latent_model = None
        self.actor = None
        self.critic = None
        self.critic_target = None
        self.encoder_target = None
        self.latent_target = None

        self.actor_optimizer = None
        self.critic_optimizer = None
        self.log_alpha_optimizer = None
        self.ae_optimizer = None
        self.latent_optimizer = None

        # for gradient magnitude tracking
        self.actor_tracker = None
        self.critic_tracker = None
        self.encoder_tracker = None
        self.decoder_tracker = None

        # entropy
        self.log_alpha = None
        self.target_entropy = None

        self.logging_info = {}
        self.eval_aux = {}
        self.env_steps = 0

    def initialize_env_dmc(self, spec: Dict[str, Any]):
        env_config = spec['trainer']['config']['env']
        self.env = dmc2gym.make(
            domain_name=env_config['domain_name'],
            task_name=env_config['task_name'],
            frame_skip=env_config['action_repeat'],
            seed=self.seed,
            visualize_reward=False,
            from_pixels=True,
            height=64,
            width=64,
            episode_length=env_config['episode_length'],
            is_distracting_cs=False,
        )
        self.env = FrameStack(self.env, env_config['frame_stack'])
        self.env = DMCObsWrapper(self.env)

        # eval env with distracting background
        paths = []
        loaded_paths = [os.path.join(dir_path, 'DAVIS/JPEGImages/480p') for dir_path in utils.load_config('datasets')]
        for path in loaded_paths:
            if os.path.exists(path):
                paths.append(path)

        # distracting env
        self.distract_env = dmc2gym.make(
            domain_name=env_config['domain_name'],
            task_name=env_config['task_name'],
            frame_skip=env_config['action_repeat'],
            seed=self.seed,
            visualize_reward=False,
            from_pixels=True,
            height=64,
            width=64,
            episode_length=env_config['episode_length'],
            is_distracting_cs=True,
            background_dataset_paths=paths,
            distracting_cs_intensity=0.025,
        )
        self.distract_env = FrameStack(self.distract_env, env_config['frame_stack'])
        self.distract_env = DMCObsWrapper(self.distract_env)

        # eval env with color changes
        self.color_env = dmc2gym.make(
            domain_name=env_config['domain_name'],
            task_name=env_config['task_name'],
            frame_skip=env_config['action_repeat'],
            seed=self.seed,
            visualize_reward=False,
            from_pixels=True,
            height=64,
            width=64,
            episode_length=env_config['episode_length'],
            is_distracting_cs=False,
        )
        self.color_env = FrameStack(self.color_env, env_config['frame_stack'])
        self.color_env = ColorWrapper(self.color_env, 'color_hard')
        self.color_env = DMCObsWrapper(self.color_env)

        video_dir = Path(self.exp_dir) / 'videos'
        video_dir.mkdir(exist_ok=True)
        self.video_recorder = VideoRecorder(str(video_dir), height=448, width=448)

    def build(self, spec: Dict[str, Any]):
        self.spec = spec

        self.initialize_env_dmc(spec)

        self.obs_shape = self.env.observation_space['rgb'].shape
        self.action_size = self.env.action_space.shape[0]

        self.buffer = utils.ReplayBuffer(
            obs_shape=self.obs_shape,
            action_shape=self.env.action_space.shape,
            capacity=self.buffer_capacity,
            batch_size=self.batch_size
        )

        shared_trunk = QuantizedEncoder(
            obs_shape=(3, 64, 64),
            num_latents=self.num_latents,
        ).to(self.device)

        shared_history_encoder = HistoryEncoder(self.num_latents, self.embedding_size).to(self.device)

        self.actor_encoder = OuterEncoder(shared_trunk,
                                          shared_history_encoder,
                                          RLProjection(self.embedding_size, self.embedding_size)).to(self.device)
        self.critic_encoder = OuterEncoder(shared_trunk,
                                           shared_history_encoder,
                                           RLProjection(self.embedding_size, self.embedding_size)).to(self.device)

        self.decoder = QuantizedDecoder(
            obs_shape=(3, 64, 64),
            transition_shape=(256, 4, 4),
            num_latents=self.num_latents
        ).to(self.device)

        self.actor_encoder.apply(weight_init)
        self.critic_encoder.apply(weight_init)
        self.decoder.apply(weight_init)

        self.latent_model = AssociativeLatent(
            num_latents=self.num_latents,
            num_values_per_latent=self.values_per_latent,
            beta=self.beta
        ).to(self.device)

        self.actor = Actor(
            embedding_size=self.embedding_size,
            action_shape=self.env.action_space.shape,
            hidden_dim=self.hidden_dim,
            log_std_min=self.actor_log_std_min,
            log_std_max=self.actor_log_std_max,
        ).to(self.device)

        self.critic = Critic(
            embedding_size=self.embedding_size,
            action_shape=self.env.action_space.shape,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self.critic_target = deepcopy(self.critic)
        self.encoder_target = deepcopy(self.critic_encoder)
        self.latent_target = deepcopy(self.latent_model)

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(self.env.action_space.shape)

        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.actor_encoder.parameters()), lr=self.actor_lr,
            betas=(self.actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters())
            + list(self.critic_encoder.parameters()), lr=self.critic_lr,
            betas=(self.critic_beta, 0.999)
        )

        self.ae_optimizer = torch.optim.AdamW(
            list(shared_trunk.parameters()) + list(self.decoder.parameters()), lr=1e-3, weight_decay=0.1
        )

        self.latent_optimizer = torch.optim.Adam(self.latent_model.parameters(), lr=1e-3)

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr, betas=(self.alpha_beta, 0.999)
        )

        self.train_mode(True)
        self.critic_target.train(True)
        self.encoder_target.train(True)

        if self.log_grad_norm:
            self.actor_tracker = GradMagnitudeTracker(self.actor)
            self.critic_tracker = GradMagnitudeTracker(self.critic)
            self.encoder_tracker = GradMagnitudeTracker(self.critic_encoder)
            self.decoder_tracker = GradMagnitudeTracker(self.decoder)

    def train_mode(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.latent_model.train(training)
        self.actor_encoder.train(training)
        self.critic_encoder.train(training)

    def eval(self):
        self.train_mode(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            embeds = self.compute_embeddings(obs, self.actor_encoder)
            mu, _, _, _ = self.actor(embeds, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            embeds = self.compute_embeddings(obs, self.actor_encoder)
            mu, pi, _, _ = self.actor(embeds, compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done):
        with torch.no_grad():
            next_embeds_actor = self.compute_embeddings(next_obs, self.actor_encoder)
            next_embeds_critic = self.compute_embeddings(next_obs, self.encoder_target, latent_target=True)
            _, policy_action, log_pi, _ = self.actor(next_embeds_actor)
            target_Q1, target_Q2 = self.critic_target(next_embeds_critic, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        embeds = self.compute_embeddings(obs, self.critic_encoder)
        current_Q1, current_Q2 = self.critic(embeds, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)

        self.logging_info.setdefault('train/critic_loss', []).append(critic_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.log_grad_norm:
            critic_grads = self.critic_tracker.get_grad_magnitudes()
            critic_max = max(list(critic_grads.values()))
            self.logging_info.setdefault('grad_norms/critic_max', []).append(critic_max)

    def update_actor_and_alpha(self, obs, update_alpha=True):
        actor_embeds = self.compute_embeddings(obs, self.actor_encoder, detach=True)
        critic_embeds = self.compute_embeddings(obs, self.critic_encoder, detach=True)
        _, pi, log_pi, log_std = self.actor(actor_embeds)
        actor_Q1, actor_Q2 = self.critic(critic_embeds, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)

        self.logging_info.setdefault('train/actor_loss', []).append(actor_loss)
        self.logging_info.setdefault('train/actor_entropy', []).append(entropy.mean())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            self.logging_info.setdefault('train/alpha_loss', []).append(alpha_loss)
            self.logging_info.setdefault('train/alpha_value', []).append(self.alpha)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        if self.log_grad_norm:
            actor_grads = self.actor_tracker.get_grad_magnitudes()
            actor_max = max(list(actor_grads.values()))
            self.logging_info.setdefault('grad_norms/actor_max', []).append(actor_max)

    def soft_update_critic_target(self):
        utils.soft_update_params(
            self.critic.Q1, self.critic_target.Q1, self.critic_tau
        )
        utils.soft_update_params(
            self.critic.Q2, self.critic_target.Q2, self.critic_tau
        )
        utils.soft_update_params(
            self.critic_encoder, self.encoder_target,
            self.encoder_tau
        )
        utils.soft_update_params(
            self.latent_model, self.latent_target,
            self.encoder_tau
        )

    def compute_embeddings(self, obs, encoder, latent_target=False, detach=False):
        batch_size = obs.shape[0]
        obs = einops.rearrange(obs, 'b (f c) h w -> (b f) c h w', c=3)
        pre_z = encoder.shared_trunk(obs)

        latent_model = self.latent_model if latent_target is False else self.latent_target
        z_hat = latent_model(pre_z)['z_quantized']
        z_hat = einops.rearrange(z_hat, '(b f) e -> b f e', b=batch_size)
        z_hat = encoder.shared_history_encoder(z_hat)
        if detach:
            z_hat = z_hat.detach()
        embeds = encoder.projection(z_hat)
        return embeds

    def update_alda(self, obs):
        # use the batch of latest frames only for training
        obs = obs[:, -3:]
        pre_z = self.critic_encoder.shared_trunk(obs)
        outs_latent = self.latent_model(pre_z)
        x_hat_logits = self.decoder(outs_latent['z_hat'])
        outs = {'pre_z': pre_z, **outs_latent}

        commitment_loss = (outs['z_continuous'] - outs['z_quantized'].detach()).pow(2).mean(1)
        quantization_loss = torch.Tensor([0.0]).to(self.device)
        if self.use_quant_loss:
            quantization_loss = (outs['z_continuous'].detach() - outs['z_quantized']).pow(2).mean(1)
        bce_loss = F.binary_cross_entropy_with_logits(x_hat_logits, target=obs)

        psnr = peak_signal_to_noise_ratio(x_hat_logits, obs)

        aux = {
            'x_hat_logits': x_hat_logits,
            'x_true': obs,
            'z_hat': outs_latent['z_hat']
        }

        lambdas = {
            'binary_cross_entropy': 1.0,
            'quantization': 0.01,
            'commitment': 0.01,
            'l2': 0.1,
            'l1': 0.0
        }

        total_loss = lambdas['commitment'] * commitment_loss + \
                     lambdas['quantization'] * quantization_loss + \
                     lambdas['binary_cross_entropy'] * bce_loss

        total_loss = total_loss.mean()
        self.ae_optimizer.zero_grad()
        self.latent_optimizer.zero_grad()
        total_loss.backward()
        self.ae_optimizer.step()
        self.latent_optimizer.step()

        if self.log_grad_norm:
            enc_grads = self.encoder_tracker.get_grad_magnitudes()
            dec_grads = self.decoder_tracker.get_grad_magnitudes()
            enc_max = max(list(enc_grads.values()))
            dec_max = max(list(dec_grads.values()))
            self.logging_info.setdefault('grad_norms/encoder_max', []).append(enc_max)
            self.logging_info.setdefault('grad_norms/decoder_max', []).append(dec_max)

        # logging
        # self.eval_aux = aux
        self.logging_info.setdefault('alda/total_loss', []).append(total_loss)
        self.logging_info.setdefault('alda/commitment_loss', []).append(commitment_loss.mean())
        self.logging_info.setdefault('alda/quantization_loss', []).append(quantization_loss.mean())

        self.logging_info.setdefault('alda/bce_loss', []).append(bce_loss.mean())
        self.logging_info.setdefault('alda/psnr', []).append(psnr.mean().detach().cpu())

    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs, next_obs = self.preprocess_obs(obs), self.preprocess_obs(next_obs)

        self.update_critic(obs, action, reward, next_obs, not_done)

        self.update_alda(obs)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

    def preprocess_obs(self, obs: torch.Tensor):
        obs = torch.from_numpy(obs).to(self.device) if isinstance(obs, np.ndarray) else obs.to(self.device)
        obs = obs.float() / 255.
        return obs

    def postprocess_image(self, img: torch.Tensor):
        """
        Convert 1 x 9 x W x H normalized tensor to 3 x W x H unnormalized numpy image
        """
        img = F.interpolate(img, size=(448, 224), mode='bilinear').squeeze(0)
        # img = (img + 1.0) * 0.5 * 255.0
        img = img * 255.0
        img = img.permute(1, 2, 0).to(torch.uint8)
        return img.detach().cpu().numpy()

    def evaluate(self, step, distracting_env=False, color_env=False):
        episode_rewards = []
        if distracting_env:
            env = self.distract_env
        elif color_env:
            env = self.color_env
        else:
            env = self.env
        for i in range(self.n_eval_episodes):
            obs, _ = env.reset()
            self.video_recorder.init(enabled=(i == 0))

            done = False
            episode_reward = 0
            env_step = 0
            while not done:
                with eval_mode(self):
                    action = self.select_action(self.preprocess_obs(obs['rgb'][None]))
                obs, reward, done, _, _ = env.step(action)

                # pre_z = self.critic_encoder.shared_trunk(self.preprocess_obs(obs['rgb'][-3:][None]))
                # embed = self.latent_model(pre_z)['z_hat']
                # frame = F.sigmoid(self.decoder(embed))
                # frame = self.postprocess_image(frame)

                self.video_recorder.record(env)
                episode_reward += reward
                env_step += 1

            _test_env = ''
            if distracting_env:
                _test_env = '_distracting'
            elif color_env:
                _test_env = '_color'

            self.video_recorder.save(f'{step}{_test_env}.mp4')
            logger.info(f'eval/episode_reward{_test_env}: {episode_reward}')
            if distracting_env:
                self.logging_info.setdefault('eval/episode_reward_distracting', []).append(episode_reward)
            elif color_env:
                self.logging_info.setdefault('eval/episode_reward_color_hard', []).append(episode_reward)
            else:
                self.logging_info.setdefault('eval/episode_reward', []).append(episode_reward)
            episode_rewards.append(episode_reward)

        # perturb latents and visualize reconstructions
        if self.eval_aux and self.use_wandb:
            # log_reconstruction_metrics(self.eval_aux, step, use_wandb=self.use_wandb)
            num_samples = 16
            num_perturbations = 20

            latent_mins = self.eval_aux['z_hat'].min(dim=0).values
            latent_maxs = self.eval_aux['z_hat'].max(dim=0).values

            for i_latent in range(self.num_latents):
                latent_perturbed = torch.tile(self.eval_aux['z_hat'][:num_samples], (
                    num_perturbations, 1, 1))  # (num_perturbations, num_samples, num_latents)
                latent_perturbed[:, :, i_latent] = torch.linspace(latent_mins[i_latent].item(),
                                                                  latent_maxs[i_latent].item(), num_perturbations)[:,
                                                   None]

                x = []
                for i_perturbation in range(num_perturbations):
                    z = latent_perturbed[i_perturbation]
                    x.append(
                        F.sigmoid(self.decoder(z))
                    )
                x = torch.stack(x)
                image = einops.rearrange(x, 'v n c h w -> (n h) (v w) c')
                wandb.log({f'latent {i_latent} perturbations': wandb.Image(image.detach().cpu().numpy()),
                           'env_step': self.env_steps})

        return np.mean(episode_rewards)

    def train(self):
        start_step, episode, episode_reward, done = 0, 0, 0, True
        for step in tqdm(range(self.n_train_steps)):
            if done:

                # eval
                if step % self.eval_n_steps == 0:
                    logger.info(f'Evaluating episode {episode}, step {step}')
                    self.evaluate(step)
                    self.evaluate(step, distracting_env=True)
                    self.evaluate(step, color_env=True)

                # save
                if step > start_step and step % self.checkpoint_n_steps == 0:
                    logger.info('Saving checkpoint...')
                    self.save_checkpoint()

                obs, _ = self.env.reset()

                self.logging_info.setdefault('train/episode_reward', []).append(episode_reward)

                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

            # Log metrics
            if step % self.log_every == 0:
                for k, v in self.logging_info.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    self.logging_info[k] = sum(v) / len(v)
                self.logging_info['env_step'] = self.env_steps

                if self.use_wandb:
                    wandb.log(self.logging_info)

                self.logging_info.clear()

            # sample action for data collection
            if step < self.init_steps:
                action = self.env.action_space.sample()
            else:
                with eval_mode(self):
                    action = self.sample_action(self.preprocess_obs(obs['rgb'][None]))

            # run training update
            if step >= self.init_steps:
                num_updates = self.init_steps if step == self.init_steps else 1
                for _ in range(num_updates):
                    self.update(self.buffer, step)

            # take step
            next_obs, reward, done, _, _ = self.env.step(action)
            done_bool = 0 if episode_step + 1 == self.env.env._max_episode_steps else float(done)
            self.buffer.add(obs['rgb'], action, reward, next_obs['rgb'], done_bool)
            episode_reward += reward
            obs = next_obs

            episode_step += 1
            self.env_steps += 1

    def save_checkpoint(self):
        filepath = self.cp_dir.joinpath(f'sac_{self.env_name}_step_{self.env_steps}.pt')
        logger.info(f'Saving checkpoint to {filepath}')

        torch.save(
            {
                'env_steps': self.env_steps,
                'actor_encoder': self.actor_encoder.state_dict(),
                'critic_encoder': self.critic_encoder.state_dict(),
                'latent_model': self.latent_model.state_dict(),
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'encoder_target': self.encoder_target.state_dict(),
                'latent_target': self.latent_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'ae_optimizer': self.ae_optimizer.state_dict(),
                'latent_optimizer': self.latent_optimizer.state_dict(),
                'log_alpha': self.log_alpha,
                'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
            },
            filepath
        )
        # save replay buffer to disk
        if self.save_buffer:
            buffer_fp = self.cp_dir.joinpath(f'sac_{self.env_name}_replay_buffer')
            self.buffer.save(buffer_fp)

    def load_checkpoint(self, path: str):
        path = Path(path)
        if not path.exists():
            raise RuntimeError(f"Checkpoint {path} does not exist.")
        logger.info(f'Loading checkpoint from {path}...')
        checkpoint = torch.load(path, map_location=self.device)

        self.env_steps = checkpoint['env_steps']
        self.actor_encoder.load_state_dict(checkpoint['actor_encoder'])
        self.critic_encoder.load_state_dict(checkpoint['critic_encoder'])
        self.latent_model.load_state_dict(checkpoint['latent_model'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.encoder_target.load_state_dict(checkpoint['encoder_target'])
        self.latent_target.load_state_dict(checkpoint['latent_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.ae_optimizer.load_state_dict(checkpoint['ae_optimizer'])
        self.latent_optimizer.load_state_dict(checkpoint['latent_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])

        # buffer_fp = self.cp_dir.joinpath(f'sac_{self.env_name}_replay_buffer.npz')
        # self.buffer.load(buffer_fp)
