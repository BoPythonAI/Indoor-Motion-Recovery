"""Fresh terrain prior every control step; only learned corrections are cached."""
from types import SimpleNamespace
import torch
from contracts import validate_checkpoint, validate_runtime
from model import ChunkPolicy


def load_model(path, device='cpu', weights='default'):
    state = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint(state)
    key = {'default': 'model', 'online': 'online_model', 'ema': 'ema_model'}[weights]
    if key not in state:
        raise ValueError(f'Checkpoint does not contain requested {weights} weights')
    model = ChunkPolicy(state['obs_dim'], state['kind'], state['horizon']).to(device)
    model.load_state_dict(state[key], strict=True)
    return model.eval().requires_grad_(False), state


class WarmStudentController:
    def __init__(self, task, model, prior, seed=17, execute_steps=1,
                 noise_mode='independent', inference_steps=8):
        if noise_mode not in ('independent', 'shift') or not 1 <= execute_steps <= model.horizon:
            raise ValueError('Invalid diffusion noise mode or chunk execution length')
        self.task, self.model, self.prior = task, model, prior
        self.feature_version = task.feature_version
        self.execute_steps, self.noise_mode, self.inference_steps = execute_steps, noise_mode, inference_steps
        self.generator = torch.Generator(device=task.device).manual_seed(seed)
        self.cache = self.cursor = self.initial_noise = self.noise_ready = None
        self.last_correction = self.last_prior = None

    def reset(self, ids=None):
        if ids is None or self.cache is None:
            self.cache = self.cursor = self.initial_noise = self.noise_ready = None
        else:
            self.cursor[ids] = self.execute_steps
            self.noise_ready[ids] = False
        self.last_correction = self.last_prior = None
        self.task._pending_prior = None

    @torch.no_grad()
    def __call__(self, history):
        if history.shape != (self.task.n, self.model.obs_dim):
            raise ValueError('Student requires the current four-frame warm history')
        if self.cache is None:
            shape = (len(history), self.model.horizon, 69)
            self.cache = torch.empty(shape, device=history.device)
            self.initial_noise = torch.empty(shape, device=history.device)
            self.cursor = torch.full((len(history),), self.execute_steps, device=history.device, dtype=torch.long)
            self.noise_ready = torch.zeros(len(history), device=history.device, dtype=torch.bool)
        query = (self.cursor >= self.execute_steps).nonzero(as_tuple=True)[0]
        if len(query):
            noise = None
            if self.model.kind == 'diffusion' and self.noise_mode == 'shift':
                noise = torch.randn((len(query), self.model.horizon, 69), device=history.device, generator=self.generator)
                reuse = self.noise_ready[query]
                if reuse.any():
                    noise[reuse, :self.model.horizon - self.execute_steps] = self.initial_noise[query[reuse], self.execute_steps:]
                self.initial_noise[query] = noise
                self.noise_ready[query] = True
            self.cache[query] = self.model.sample(history[query], self.inference_steps, self.generator, noise)
            self.cursor[query] = 0
        correction = self.cache[torch.arange(len(history), device=history.device), self.cursor].clone()
        self.cursor += 1
        # Never cache prior deltas across steps: the tracker observes new physics every step.
        prior = self.prior.current()
        self.task.set_prior(prior)
        self.last_correction, self.last_prior = correction, prior.delta
        if not torch.isfinite(correction).all():
            raise ValueError('Student emitted a nonfinite correction')
        return SimpleNamespace(mean=prior.delta + correction), None


def make_student(task, config, prior=None, checkpoint_key='student_checkpoint'):
    from warm_prior import FrozenTerrainPrior
    model, state = load_model(config[checkpoint_key], task.device, config.get('student_weights', 'default'))
    prior = prior or FrozenTerrainPrior(task.agent, config)
    validate_runtime(state, task, prior, config)
    expected_kind = config.get('method') if checkpoint_key == 'student_checkpoint' else config.get('rollin_method')
    if model.kind != expected_kind:
        raise ValueError('Student checkpoint kind does not match requested bc/diffusion method')
    seed_key = 'rollin_seed' if checkpoint_key == 'rollin_student_checkpoint' else 'inference_seed'
    sampling_seed = config.get(seed_key)
    if sampling_seed is None:
        sampling_seed = config.get('seed', 17)
    controller = WarmStudentController(task, model, prior, seed=sampling_seed,
        execute_steps=config.get('rollin_execute_steps' if checkpoint_key == 'rollin_student_checkpoint' else 'execute_steps', 1),
        noise_mode=config.get('rollin_noise_mode' if checkpoint_key == 'rollin_student_checkpoint' else 'noise_mode', 'independent'),
        inference_steps=config.get('rollin_inference_steps' if checkpoint_key == 'rollin_student_checkpoint' else 'inference_steps', 8))
    controller.state = state
    controller.inference_seed = sampling_seed
    return controller
