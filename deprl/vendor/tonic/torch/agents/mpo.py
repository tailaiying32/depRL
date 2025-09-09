import torch

from deprl.vendor.tonic import replays  # noqa
from deprl.vendor.tonic.torch import models, normalizers, updaters
from deprl.vendor.tonic.utils import logger

from .agent import Agent


def default_model():
    return models.ActorCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((256, 256), torch.nn.ReLU),
            head=models.GaussianPolicyHead(),
        ),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP((256, 256), torch.nn.ReLU),
            head=models.ValueHead(),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )


class MPO(Agent):
    """Maximum a Posteriori Policy Optimisation.
    MPO: https://arxiv.org/pdf/1806.06920.pdf
    MO-MPO: https://arxiv.org/pdf/2005.07513.pdf
    """

    def __init__(
        self, model=None, replay=None, actor_updater=None, critic_updater=None
    ):
        self.model = model or default_model()
        self.replay = replay or replays.Buffer(return_steps=5)
        self.actor_updater = (
            actor_updater or updaters.MaximumAPosterioriPolicyOptimization()
        )
        self.critic_updater = critic_updater or updaters.ExpectedSARSA()

    def initialize(self, observation_space, action_space, seed=None):
        super().initialize(seed=seed)
        self.model.initialize(observation_space, action_space)
        self.replay.initialize(seed)
        self.actor_updater.initialize(self.model, action_space)
        self.critic_updater.initialize(self.model)

    def step(self, observations, steps):
        actions = self._step(observations)
        actions = actions.detach().cpu().numpy()

        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_actions = actions.copy()

        return actions

    def test_step(self, observations, steps):
        # Sample actions for testing.
        return self._test_step(observations).detach().cpu().numpy()

    def update(self, observations, rewards, resets, terminations, steps):
        # Store the last transitions in the replay.
        self.replay.store(
            observations=self.last_observations,
            actions=self.last_actions,
            next_observations=observations,
            rewards=rewards,
            resets=resets,
            terminations=terminations,
        )

        # Prepare to update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        # Update the model if the replay is ready.
        if self.replay.ready(steps):
            self._update(steps)

    def _step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).sample()

    def _test_step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).loc

    def _update(self, steps):
        keys = (
            "observations",
            "actions",
            "next_observations",
            "rewards",
            "discounts",
        )

        # Update both the actor and the critic multiple times.
        for batch in self.replay.get(*keys, steps=steps):
            batch = {k: torch.as_tensor(v) for k, v in batch.items()}
            infos = self._update_actor_critic(**batch)

            for key in infos:
                for k, v in infos[key].items():
                    logger.store(key + "/" + k, v.detach().cpu().numpy())

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

    def _update_actor_critic(
        self, observations, actions, next_observations, rewards, discounts
    ):
        
        # Add gradient checks and clipping

        # After computing the actor gradients but before optimizer.step()
        # Find this section in the _update_actor_critic method

        # Add this before any optimizer.step() calls:
        # Clip actor gradients
        for param in self.model.actor.parameters():
            if param.grad is not None:
                param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0)
        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), max_norm=1.0)

        # Similarly for critic parameters
        for param in self.model.critic.parameters():
            if param.grad is not None:
                param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0)
        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=1.0)
        
        critic_infos = self.critic_updater(
            observations, actions, next_observations, rewards, discounts
        )
        actor_infos = self.actor_updater(observations)
        self.model.update_targets()
        return dict(critic=critic_infos, actor=actor_infos)

