"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th
from matplotlib import pyplot as plt
import os

from nn import mean_flat
from losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
                #print(x.shape)
                #x_tmp=x[0,:,:,:]
                #x_per=th.permute(x_tmp,(1,2,0))
                #plt.imshow(x_per)
                #plt.show()
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)
        #print(pred_xstart.shape)
        #pred_tmp=pred_xstart[0,:,:,:]
        #pred_per=th.permute(pred_tmp,(1,2,0))
        #plt.imshow(pred_per)
        #plt.show()


        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        #print(sample.shape)
        #smp_tmp=sample[0,:,:,:]
        #smp_per=th.permute(smp_tmp,(1,2,0))
        #plt.imshow(smp_per)
        #plt.show()

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]
                
                #turn this following piece of code into actual code if necessary
                def map_tensor_to_range(tensor):
                    min_val = th.min(tensor)
                    max_val = th.max(tensor)
                    tensor_mapped = (tensor - min_val) / (max_val - min_val)
                    return tensor_mapped


                if i==3999:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachtensample/sample1/sample3999.pt"
                    th.save(img_per, file_p)

                threshold = 0.5
                directory = "/Users/nikahosseini/Desktop/trained_data/ring/eachtensample/sample1"

                for i in range(3990, -10, -10):
                    img_tmp = img[0, :, :, :]
                    img_per = th.permute(img_tmp, (1, 2, 0))
                    img_per = img_per[:, :, 0]
                    img_per = map_tensor_to_range(img_per)
                    img_per = np.where(img_per >= threshold, 1, 0)
    
                    file_name = f"sample{str(i).zfill(4)}.pt"
                    file_path = os.path.join(directory, file_name)
                    th.save(img_per, file_path)

                '''elif i==3900:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample48.pt"
                    th.save(img_per, file_p)

                elif i==3800:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample47.pt"
                    th.save(img_per, file_p)

                elif i==3700:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample46.pt"
                    th.save(img_per, file_p)

                elif i==3600:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample45.pt"
                    th.save(img_per, file_p)

                elif i==3500:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample44.pt"
                    th.save(img_per, file_p)

                elif i==3400:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample43.pt"
                    th.save(img_per, file_p)

                elif i==3300:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample42.pt"
                    th.save(img_per, file_p)

                elif i==3200:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample41.pt"
                    th.save(img_per, file_p)

                elif i==3100:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample4.pt"
                    th.save(img_per, file_p)

                elif i==3000:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample39.pt"
                    th.save(img_per, file_p)

                elif i==2900:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample38.pt"
                    th.save(img_per, file_p)

                elif i==2800:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample37.pt"
                    th.save(img_per, file_p)

                elif i==2700:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample36.pt"
                    th.save(img_per, file_p)

                elif i==2600:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample35.pt"
                    th.save(img_per, file_p)

                elif i==2500:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample34.pt"
                    th.save(img_per, file_p)
                
                elif i==2400:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample33.pt"
                    th.save(img_per, file_p)
                    


                elif i==2300:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample32.pt"
                    th.save(img_per, file_p)

                elif i==2200:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample31.pt"
                    th.save(img_per, file_p)

                elif i==2100:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample3.pt"
                    th.save(img_per, file_p)


                elif i==2000:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample29.pt"
                    th.save(img_per, file_p)

                elif i==1900:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample28.pt"
                    th.save(img_per, file_p)

                elif i==1800:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample27.pt"
                    th.save(img_per, file_p)

                elif i==1700:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample26.pt"
                    th.save(img_per, file_p)

                elif i==1600:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample25.pt"
                    th.save(img_per, file_p)

                
                elif i==1500:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample24.pt"
                    th.save(img_per, file_p)

                elif i==1400:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample23.pt"
                    th.save(img_per, file_p)

                elif i==1300:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample22.pt"
                    th.save(img_per, file_p)

                elif i==1200:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample21.pt"
                    th.save(img_per, file_p)

                elif i==1100:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample2.pt"
                    th.save(img_per, file_p)
                    
                elif i==1000:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample19.pt"
                    th.save(img_per, file_p)

                elif i==900:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample18.pt"
                    th.save(img_per, file_p)

                elif i==800:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample17.pt"
                    th.save(img_per, file_p)

                elif i==700:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample16.pt"
                    th.save(img_per, file_p)

                elif i==600:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample15.pt"
                    th.save(img_per, file_p)

                elif i==500:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample14.pt"
                    th.save(img_per, file_p)

                elif i==400:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample13.pt"
                    th.save(img_per, file_p)

                elif i==300:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample12.pt"
                    th.save(img_per, file_p)

                elif i==200:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample11.pt"
                    th.save(img_per, file_p)

                elif i==100:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample1.pt"
                    th.save(img_per, file_p)

                elif i==0:
                    img_tmp=img[0,:,:,:]
                    img_per=th.permute(img_tmp,(1,2,0))
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    img_per = img_per[:,:,0]
                    img_per = map_tensor_to_range(img_per)
                    print(img_per)
                    threshold = 0.5
                    img_per = np.where(img_per >= threshold, 1, 0)
                    print(img_per)
                    print(type(img_per))
                    print(img_per.shape)
                    #plt.imshow(img_per)
                    #plt.show()
                    #tens = th.from_numpy(img_per)
                    file_p = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/sampling/sample5/sample0.pt"
                    th.save(img_per, file_p)'''

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        #print(sample.shape)
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
       
        x_t = self.q_sample(x_start, t, noise=noise)

       
        

        def map_tensor_to_range(tensor):
                    min_val = th.min(tensor)
                    max_val = th.max(tensor)
                    tensor_mapped = (tensor - min_val) / (max_val - min_val)
                    return tensor_mapped

        tmp_xstart=x_start[0,:,:,:]
        per_xstart=th.permute(tmp_xstart,(1,2,0))
        per_xstart = per_xstart[:,:,0]
        print(per_xstart)
        per_xstart[per_xstart == -1] = 0
        print([per_xstart])
        per_xstart = per_xstart.numpy()
        print(type(per_xstart))
        pathtofile = "/Users/nikahosseini/Desktop/SUMMER24/Report20/d1/xstart.pt"
        th.save(per_xstart, pathtofile)
        plt.imshow(per_xstart)
        plt.show()
        print('the original image was shown')

        #put all steps in for loop
        for t in range(10, 4000, 10):
            t_tensor = th.tensor([t])
            x_s = self.q_sample(x_start, t_tensor, noise=noise)
            tmp_xs = x_s[0, :, :, :]
            per_xs = th.permute(tmp_xs, (1, 2, 0))
            per_xs = per_xs[:, :, 0]
            per_xs = map_tensor_to_range(per_xs)
            threshold = 0.5
            per_xs = np.where(per_xs >= threshold, 1, 0)

            # Generate the filename with the leading zeros
            t_str = str(t).zfill(4)
            pathtofile = f"/Users/nikahosseini/Desktop/SUMMER24/Report20/d2/x{t_str}.pt"
            th.save(per_xs, pathtofile)

            plt.imshow(per_xs)
            plt.show()
            print(f'The second step of noisy image for t={t} was shown')

        '''t=th.tensor([100])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        print(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x100.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([200])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        print(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x200.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([300])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        print(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x300.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([400])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        print(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x400.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([500])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        print(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x500.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([600])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        print(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x600.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([700])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        print(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x700.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([800])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        print(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x800.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([900])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        print(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x900.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        

        t=th.tensor([1000])
        x_f = self.q_sample(x_start, t, noise=noise)
        #print(x_f)
        tmp_xf=x_f[0,:,:,:]
        per_xf=th.permute(tmp_xf,(1,2,0))
        per_xf = per_xf[:,:,0]
        print(per_xf)
        #print(per_xf)
        per_xf = map_tensor_to_range(per_xf)
        #per_xf[per_xf == -1] = 0
        #print(per_xf)
        threshold = 0.5
        per_xf = np.where(per_xf >= threshold, 1, 0)
        print(per_xf)
        #print(per_xf)
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x1000.pt"
        th.save(per_xf, pathtofile)
        plt.imshow(per_xf)
        plt.show()
        print('the first step of noisy image was shown')

        t=th.tensor([1100])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        print(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x1100.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([1200])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        print(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x1200.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([1300])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        print(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x1300.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([1400])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        print(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x1400.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([1500])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x1500.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([1600])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x1600.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([1700])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x1700.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([1800])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x1800.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([1900])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x1900.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([2000])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x2000.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([2100])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x2100.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([2200])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x2200.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([2300])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x2300.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([2400])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x2400.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')


        t=th.tensor([2500])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x2500.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([2600])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x2600.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([2700])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x2700.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([2800])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x2800.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([2900])
        x_s = self.q_sample(x_start, t, noise=noise)
        tmp_xs=x_s[0,:,:,:]
        per_xs=th.permute(tmp_xs,(1,2,0))
        per_xs = per_xs[:,:,0]
        print(per_xs)
        per_xs = map_tensor_to_range(per_xs)
        threshold = 0.5
        per_xs = np.where(per_xs >= threshold, 1, 0)
        print(per_xs)
        #per_xs[per_xs == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x2900.pt"
        th.save(per_xs, pathtofile)
        plt.imshow(per_xs)
        plt.show()
        print('the second step of noisy image was shown')

        t=th.tensor([3000])
        x_th = self.q_sample(x_start, t, noise=noise)
        tmp_xth=x_th[0,:,:,:]
        per_xth=th.permute(tmp_xth,(1,2,0))
        per_xth = per_xth[:,:,0]
        per_xth = map_tensor_to_range(per_xth)
        threshold = 0.5
        per_xth = np.where(per_xth >= threshold, 1, 0)
        #per_xth[per_xth == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x3000.pt"
        th.save(per_xth, pathtofile)
        plt.imshow(per_xth)
        plt.show()
        print('the third step of noisy image was shown')

        t=th.tensor([3100])
        x_th = self.q_sample(x_start, t, noise=noise)
        tmp_xth=x_th[0,:,:,:]
        per_xth=th.permute(tmp_xth,(1,2,0))
        per_xth = per_xth[:,:,0]
        per_xth = map_tensor_to_range(per_xth)
        threshold = 0.5
        per_xth = np.where(per_xth >= threshold, 1, 0)
        #per_xth[per_xth == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x3100.pt"
        th.save(per_xth, pathtofile)
        plt.imshow(per_xth)
        plt.show()
        print('the third step of noisy image was shown')

        t=th.tensor([3200])
        x_th = self.q_sample(x_start, t, noise=noise)
        tmp_xth=x_th[0,:,:,:]
        per_xth=th.permute(tmp_xth,(1,2,0))
        per_xth = per_xth[:,:,0]
        per_xth = map_tensor_to_range(per_xth)
        threshold = 0.5
        per_xth = np.where(per_xth >= threshold, 1, 0)
        #per_xth[per_xth == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x3200.pt"
        th.save(per_xth, pathtofile)
        plt.imshow(per_xth)
        plt.show()
        print('the third step of noisy image was shown')

        t=th.tensor([3300])
        x_th = self.q_sample(x_start, t, noise=noise)
        tmp_xth=x_th[0,:,:,:]
        per_xth=th.permute(tmp_xth,(1,2,0))
        per_xth = per_xth[:,:,0]
        per_xth = map_tensor_to_range(per_xth)
        threshold = 0.5
        per_xth = np.where(per_xth >= threshold, 1, 0)
        #per_xth[per_xth == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x3300.pt"
        th.save(per_xth, pathtofile)
        plt.imshow(per_xth)
        plt.show()
        print('the third step of noisy image was shown')

        t=th.tensor([3400])
        x_th = self.q_sample(x_start, t, noise=noise)
        tmp_xth=x_th[0,:,:,:]
        per_xth=th.permute(tmp_xth,(1,2,0))
        per_xth = per_xth[:,:,0]
        per_xth = map_tensor_to_range(per_xth)
        threshold = 0.5
        per_xth = np.where(per_xth >= threshold, 1, 0)
        #per_xth[per_xth == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x3400.pt"
        th.save(per_xth, pathtofile)
        plt.imshow(per_xth)
        plt.show()
        print('the third step of noisy image was shown')

        t=th.tensor([3500])
        x_fo = self.q_sample(x_start, t, noise=noise)
        tmp_xfo=x_fo[0,:,:,:]
        per_xfo=th.permute(tmp_xfo,(1,2,0))
        per_xfo = per_xfo[:,:,0]
        per_xfo = map_tensor_to_range(per_xfo)
        threshold = 0.5
        per_xfo = np.where(per_xfo >= threshold, 1, 0)
        #per_xfo[per_xfo == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x3500.pt"
        th.save(per_xfo, pathtofile)
        plt.imshow(per_xfo)
        plt.show()
        print('the fourth step of noisy image was shown')

        t=th.tensor([3600])
        x_th = self.q_sample(x_start, t, noise=noise)
        tmp_xth=x_th[0,:,:,:]
        per_xth=th.permute(tmp_xth,(1,2,0))
        per_xth = per_xth[:,:,0]
        per_xth = map_tensor_to_range(per_xth)
        threshold = 0.5
        per_xth = np.where(per_xth >= threshold, 1, 0)
        #per_xth[per_xth == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x3600.pt"
        th.save(per_xth, pathtofile)
        plt.imshow(per_xth)
        plt.show()
        print('the third step of noisy image was shown')

        t=th.tensor([3700])
        x_th = self.q_sample(x_start, t, noise=noise)
        tmp_xth=x_th[0,:,:,:]
        per_xth=th.permute(tmp_xth,(1,2,0))
        per_xth = per_xth[:,:,0]
        per_xth = map_tensor_to_range(per_xth)
        threshold = 0.5
        per_xth = np.where(per_xth >= threshold, 1, 0)
        #per_xth[per_xth == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x3700.pt"
        th.save(per_xth, pathtofile)
        plt.imshow(per_xth)
        plt.show()
        print('the third step of noisy image was shown')
        
        t=th.tensor([3800])
        x_th = self.q_sample(x_start, t, noise=noise)
        tmp_xth=x_th[0,:,:,:]
        per_xth=th.permute(tmp_xth,(1,2,0))
        per_xth = per_xth[:,:,0]
        per_xth = map_tensor_to_range(per_xth)
        threshold = 0.5
        per_xth = np.where(per_xth >= threshold, 1, 0)
        #per_xth[per_xth == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x3800.pt"
        th.save(per_xth, pathtofile)
        plt.imshow(per_xth)
        plt.show()
        print('the third step of noisy image was shown')

        t=th.tensor([3900])
        x_th = self.q_sample(x_start, t, noise=noise)
        tmp_xth=x_th[0,:,:,:]
        per_xth=th.permute(tmp_xth,(1,2,0))
        per_xth = per_xth[:,:,0]
        per_xth = map_tensor_to_range(per_xth)
        threshold = 0.5
        per_xth = np.where(per_xth >= threshold, 1, 0)
        #per_xth[per_xth == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/ring/eachhundred/x3900.pt"
        th.save(per_xth, pathtofile)
        plt.imshow(per_xth)
        plt.show()
        print('the third step of noisy image was shown')'''

        t=th.tensor([3999])
        x_fo = self.q_sample(x_start, t, noise=noise)
        tmp_xfo=x_fo[0,:,:,:]
        per_xfo=th.permute(tmp_xfo,(1,2,0))
        per_xfo = per_xfo[:,:,0]
        per_xfo = map_tensor_to_range(per_xfo)
        threshold = 0.5
        per_xfo = np.where(per_xfo >= threshold, 1, 0)
        #per_xfo[per_xfo == -1] = 0
        pathtofile = "/Users/nikahosseini/Desktop/SUMMER24/Report20/d1/xlast.pt"
        th.save(per_xfo, pathtofile)
        plt.imshow(per_xfo)
        plt.show()
        print('the fourth step of noisy image was shown')

        

        '''t=th.tensor([2999])
        x_th = self.q_sample(x_start, t, noise=noise)
        t=th.tensor([3000])
        x_fo = self.q_sample(x_start, t, noise=noise)
        x_previous=self._predict_xstart_from_xprev(x_fo, t, x_th)
        tmp_xprevious=x_previous[0,:,:,:]
        per_xprevious=th.permute(tmp_xprevious,(1,2,0))
        #per_xprevious = per_xprevious[:,:,0]
        #per_xprevious[per_xprevious >= 0] = 1  # Set elements greater than or equal to zero to 1
        #per_xprevious[per_xprevious < 0] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/denoising/backward/firstgraph/backxfourth.pt"
        th.save(per_xprevious, pathtofile)
        plt.imshow(per_xprevious)
        plt.show()



        
      
        t=th.tensor([1999])
        x_s=self.q_sample(x_start, t, noise=noise)
        t=th.tensor([2000])
        x_th=self.q_sample(x_start, t, noise=noise)
        x_previous=self._predict_xstart_from_xprev(x_th, t, x_s)
        tmp_xprevious=x_previous[0,:,:,:]
        per_xprevious=th.permute(tmp_xprevious,(1,2,0))
        #per_xprevious = per_xprevious[:,:,0]
        #per_xprevious[per_xprevious >= 0] = 1  # Set elements greater than or equal to zero to 1
        #per_xprevious[per_xprevious < 0] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/denoising/backward/firstgraph/backxthird.pt"
        th.save(per_xprevious, pathtofile)
        plt.imshow(per_xprevious)
        plt.show()

        

        
        t=th.tensor([999])
        x_f=self.q_sample(x_start, t, noise=noise)
        t=th.tensor([1000])
        x_s=self.q_sample(x_start, t, noise=noise)
        x_previous=self._predict_xstart_from_xprev(x_s, t, x_f)
        tmp_xprevious=x_previous[0,:,:,:]
        per_xprevious=th.permute(tmp_xprevious,(1,2,0))
        #per_xprevious = per_xprevious[:,:,0]
        #per_xprevious[per_xprevious >= 0] = 1  # Set elements greater than or equal to zero to 1
        #per_xprevious[per_xprevious < 0] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/denoising/backward/firstgraph/backxsecond.pt"
        th.save(per_xprevious, pathtofile)
        plt.imshow(per_xprevious)
        plt.show()'''


        
        
        t=th.tensor([0])
        x_z=self.q_sample(x_start, t, noise=noise)
        t=th.tensor([1])
        x_o=self.q_sample(x_start, t, noise=noise)
        x_previous=self._predict_xstart_from_xprev(x_o, t, x_z)
        tmp_xprevious=x_previous[0,:,:,:]
        per_xprevious=th.permute(tmp_xprevious,(1,2,0))
        per_xprevious = per_xprevious[:,:,0]
        per_xprevious = map_tensor_to_range(per_xprevious)
        #per_xf[per_xf == -1] = 0
        #print(per_xprevious)
        threshold = 0.5
        per_xprevious = np.where(per_xprevious >= threshold, 1, 0)
        #print(per_xprevious)
        #per_xprevious[per_xprevious >= 0] = 1  # Set elements greater than or equal to zero to 1
        #per_xprevious[per_xprevious < 0] = 0
        pathtofile = "/Users/nikahosseini/Desktop/trained_data/denoising/backward/thirdgraph/backxorigin.pt"
        th.save(per_xprevious, pathtofile)
        plt.imshow(per_xprevious)
        plt.show()



        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        #print(t)
        #print(self.num_timesteps)
        #print(batch_size)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
