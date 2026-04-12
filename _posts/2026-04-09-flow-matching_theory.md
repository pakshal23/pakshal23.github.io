---
layout: post
title: "Flow Matching: Theory and Applications to Inverse Problems"
date: 2026-04-09
---

Over the past few years, flow matching has gained a lot of traction in the field of generative modelling. It is the engine behind some of the best open-weight image generation models out there such as Stable Diffusion 3.5 and FLUX.2. The images below, for instance, were both generated with Flux.2 [dev].

<figure style="display: flex; flex-direction: row; gap: 12px; justify-content: center; align-items: flex-start; margin: 24px 0; flex-wrap: wrap;">
  <div style="display: flex; flex-direction: column; align-items: center; flex: 1; min-width: 200px; max-width: 360px;">
    <img src="{{ site.url }}/assets/images/zurich.webp" alt="Zurich at golden hour, generated with Flux 2.0" style="width: 100%; border-radius: 6px;">
    <figcaption style="font-size: 0.85em; color: #888; margin-top: 8px; text-align: center;">
      Generated with Flux.2 [dev] &nbsp;·&nbsp;
      <details style="display: inline;">
        <summary style="display: inline; cursor: pointer; color: #aaa;">(prompt)</summary>
        <span style="font-style: italic;">"A stunning aerial view of Zurich at golden hour, the Limmat river winding through the city, old town rooftops and church spires visible, warm light reflecting off the water, cinematic, photorealistic"</span>
      </details>
    </figcaption>
  </div>
  <div style="display: flex; flex-direction: column; align-items: center; flex: 1; min-width: 200px; max-width: 360px;">
    <img src="{{ site.url }}/assets/images/chess.webp" alt="Chess board mid-game, generated with Flux 2.0" style="width: 100%; border-radius: 6px;">
    <figcaption style="font-size: 0.85em; color: #888; margin-top: 8px; text-align: center;">
      Generated with Flux.2 [dev] &nbsp;·&nbsp;
      <details style="display: inline;">
        <summary style="display: inline; cursor: pointer; color: #aaa;">(prompt)</summary>
        <span style="font-style: italic;">"A chess board mid-game, dramatic side lighting, shallow depth of field, cinematic"</span>
      </details>
    </figcaption>
  </div>
</figure>

Having worked with VAEs, GANs, and diffusion models in the past, I figured it was time (belatedly, I must admit) to understand flow matching. My reading was based on two excellent sources: <a href="https://arxiv.org/pdf/2506.02070" target="_blank" rel="noopener noreferrer"><em>An Introduction to Flow Matching and Diffusion Models</em></a> and <a href="https://arxiv.org/pdf/2510.21890" target="_blank" rel="noopener noreferrer"><em>The Principles of Diffusion Models</em></a>. I found this powerful framework to be very elegant. What also intrigued me were the close links and commonalities between flow matching and diffusion models. This post is my attempt to share what I understood. Also, since I worked on inverse problems during my PhD, I was naturally curious about the application of flow models in that setting, and so the second half of the post is a brief overview of two such different approaches: <a href="https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_FlowDPS__Flow-Driven_Posterior_Sampling_for_Inverse_Problems_ICCV_2025_paper.pdf" target="_blank" rel="noopener noreferrer">FlowDPS</a> and <a href="https://openreview.net/pdf?id=QGd34p02mI" target="_blank" rel="noopener noreferrer">FLOWER</a>.

---

## Theory

### Goal: Learning a Flow

The goal in flow matching is to learn a time-dependent vector field $$\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t)$$ that transforms samples from a source distribution $$p_0=p_{\text{source}}$$ (for example, Gaussian noise) to samples from a target distribution $$p_1=p_{\text{target}}$$ (our data distribution) by solving the ordinary differential equation (ODE):

$$\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t} = \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t).$$

Intuitively, the vector field tells a particle at position $$\mathbf{x}$$ at time $$t$$ which direction to move. By following this field from $$t=0$$ to $$t=1$$, a noise sample is gradually transformed into a realistic image (or whatever your data looks like).

### Probability Paths

Any vector field $$\mathbf{v}_t(\mathbf{x}) = \mathbf{v}(\mathbf{x}, t)$$ that achieves the above goal induces a probability path $$p_t(\mathbf{x})$$ — a family of distributions interpolating between $$p_0 = p_{\text{source}}$$ and $$p_1 = p_{\text{target}}$$.

At a high level, our roadmap should be to specify a valid probability path $$p_t$$ interpolating between $$p_0 = p_{\text{source}}$$ and $$p_1 = p_{\text{target}}$$, and train a network to produce a vector field that generates this path. The natural question is: can we do this in a tractable way? The answer is yes. In flow matching, we define $$p_t$$ via a conditioning variable $$\mathbf{z}$$ and a conditional path $$p_t(\mathbf{x} \mid \mathbf{z})$$, so that

$$p_t(\mathbf{x}) = \int p_t(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})\, \mathrm{d}\mathbf{z}.$$

Why this conditional structure? The reason will become clear when we discuss the training loss but essentially this conditional structure is what makes the training tractable.

While there is freedom of choice for $$p(\mathbf{z})$$ and $$p_t(\mathbf{x} \mid \mathbf{z})$$, we just need to ensure that the boundary conditions $$p_0=p_{\text{source}}$$ and $$p_1=p_{\text{target}}$$ are satisfied. Two commonly used constructions are:

**1. Affine conditional flows:** We set $$\mathbf{z} = (\mathbf{x}_0, \mathbf{x}_1) \sim p_{\text{source}}(\mathbf{x}_0)\, p_{\text{target}}(\mathbf{x}_1)$$ and define the conditional path as:

$$p_t(\mathbf{x} \mid \mathbf{z}) = \delta\big(\mathbf{x} - (\alpha_t\, \mathbf{x}_1 + \beta_t\, \mathbf{x}_0)\big),$$

where $$\alpha_t, \beta_t$$ are differentiable schedules satisfying $$\alpha_0 = 0,\, \beta_0 = 1$$ and $$\alpha_1 = 1,\, \beta_1 = 0$$. A popular special case is **rectified flows**, which use the linear schedule $$\alpha_t = t,\, \beta_t = 1 - t$$, resulting in the simple interpolation path: $$\mathbf{x}_t = t\,\mathbf{x}_1 + (1-t)\,\mathbf{x}_0$$.

**2. Gaussian probability paths:** We set $$\mathbf{z} = \mathbf{x}_1 \sim p_{\text{target}}(\mathbf{x}_1)$$ and use a Gaussian conditional:

$$p_t(\mathbf{x} \mid \mathbf{z}) = \mathcal{N}\!\big(\mathbf{x};\, \alpha_t\, \mathbf{x}_1,\, \beta_t^2\, \mathbf{I}\big),$$

with the same boundary conditions on $$\alpha_t, \beta_t$$ as in the case of affine flows.



### Conditional Vector Field
For any reasonably well-behaved chosen conditional probability path $$p_t(\mathbf{x} \mid \mathbf{z})$$, there exists a conditional vector field $$\mathbf{v}_t(\mathbf{x} \mid \mathbf{z})$$ that generates it via the ODE

$$\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t} = \mathbf{v}_{t}(\mathbf{x} \mid \mathbf{z}).$$

For the common choices of the conditional path discussed earlier, such a field can be derived in closed form.

**1. Affine conditional flows:** We have 

$$\mathbf{v}_t(\mathbf{x} \mid \mathbf{z}) = \dot{\alpha}_t\, \mathbf{x}_1 + \dot{\beta}_t\, \mathbf{x}_0,$$

where $$\dot{\alpha}_t$$ and $$\dot{\beta}_t$$ denote the time derivatives of the schedules. For the special case of rectified flows ($$\alpha_t = t,\, \beta_t = 1-t$$), this simplifies to $$\mathbf{v}_t(\mathbf{x} \mid \mathbf{z}) = \mathbf{x}_1 - \mathbf{x}_0$$.

**2. Gaussian probability paths:** We have 

$$\mathbf{v}_t(\mathbf{x} \mid \mathbf{z}) = \dot{\alpha}_t\, \mathbf{x}_1 + \frac{\dot{\beta}_t}{\beta_t}(\mathbf{x} - \alpha_t\, \mathbf{x}_1).$$

<details>
<summary style="cursor: pointer; color: #888;">Under the hood: the continuity equation</summary>
<div style="margin-top: 10px; margin-bottom: 6px;">
The conditional path and its generating field are linked by the <strong>continuity equation</strong>:

$$\frac{\partial p_t(\mathbf{x} \mid \mathbf{z})}{\partial t} + \nabla \cdot \big(p_t(\mathbf{x} \mid \mathbf{z})\, \mathbf{v}_t(\mathbf{x} \mid \mathbf{z})\big) = 0.$$

This is a conservation law: probability mass is neither created nor destroyed as it flows. For any sufficiently smooth conditional path with full support, this equation admits a solution. However, the solution is not unique.
</div>
</details>

<div style="margin-bottom: 16px;"></div>

### Marginal Vector Field

It can be shown (by relying on the continuity equation) that the marginal vector field defined as

$$\mathbf{v}_t(\mathbf{x}) = \int \mathbf{v}_t(\mathbf{x} \mid \mathbf{z})\, \frac{p_t(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})}{p_t(\mathbf{x})}\, \mathrm{d}\mathbf{z} = \mathbb{E}_{\mathbf{z} \sim p_t(\mathbf{z} \mid \mathbf{x})}\big[\mathbf{v}_t(\mathbf{x} \mid \mathbf{z})\big]$$

generates the marginal probability path $p_t(\mathbf{x})$. At each point $$\mathbf{x}$$ and time $$t$$, $$\mathbf{v}_t$$ averages over all the directions that different conditioning variables $$\mathbf{z}$$ suggest, weighted by how likely each $$\mathbf{z}$$ is given the current position.

While this characterization of the marginal vector field is indeed elegant, it is merely an abstraction as we cannot really evaluate it. This integral is intractable as we don't know $$p_{\text{target}}$$ in generative modelling (we only have access to samples from it), which is why we are trying to approximate $$\mathbf{v}_t(\mathbf{x})$$ with a neural network. In the next section, we will see how the conditional vector field allows us to set up a tractable training procedure to learn the marginal vector field from our data.

### Training: The CFM Loss

The ideal training objective would be to regress $$\mathbf{v}_{\boldsymbol{\theta}}$$ directly against the marginal field $$\mathbf{v}_t$$ by minimizing the loss

$$\mathcal{L}_\text{FM}(\boldsymbol{\theta}) = \mathbb{E}_{t \sim p(t),\, \mathbf{x}_t \sim p_t} \left\| \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) - \mathbf{v}_t(\mathbf{x}_t) \right\|^2,$$

where $$p(t)$$ is typically chosen to be a uniform distribution over $$[0, 1]$$. However, due to reasons discussed earlier, using this loss is not feasible. The magic trick here is that regressing against the conditional field instead yields the same gradients. The conditional flow matching loss is

$$\mathcal{L}_\text{CFM}(\boldsymbol{\theta}) = \mathbb{E}_{t \sim p(t),\, \mathbf{z} \sim p(\mathbf{z}),\, \mathbf{x}_t \sim p_t(\cdot \mid \mathbf{z})} \left\| \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) - \mathbf{v}_t(\mathbf{x}_t \mid \mathbf{z}) \right\|^2,$$

and it can be shown that $$\mathcal{L}_\text{FM}(\boldsymbol{\theta}) = \mathcal{L}_\text{CFM}(\boldsymbol{\theta}) + C$$, where $$C$$ does not depend on $$\boldsymbol{\theta}$$. Since we have the closed form of $$\mathbf{v}_t(\mathbf{x}_t \mid \mathbf{z})$$, this loss is easy to evaluate. 

### Inference
Once we have a learnt vector field $$\mathbf{v}_{\boldsymbol{\theta}}$$, we can generate samples from $$p_{\text{target}}$$ by sampling $$\mathbf{x}_0 \sim p_{\text{source}}$$ and integrating the ODE forward to $$t=1$$ using a numerical solver (e.g., Euler).


### A Closer Look at Affine Conditional Flows and Gaussian Probability Paths

Despite having different conditioning variables, conditional paths and conditional vector fields, affine conditional flows and Gaussian probability paths share an interesting property:

*When $$p_{\text{source}} = \mathcal{N}(\mathbf{0}, \mathbf{I})$$ and the same schedules $$\alpha_t, \beta_t$$ are used, the two constructions yield identical marginal paths $$p_t(\mathbf{x})$$ and marginal vector fields $$\mathbf{v}_t(\mathbf{x})$$. Further, this marginal vector field is linked to the score function $$\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$ via the formula*

$$\mathbf{v}_t(\mathbf{x}) = \frac{\dot{\alpha}_t}{\alpha_t}\mathbf{x} + \left(\frac{\dot{\alpha}_t \beta_t^2}{\alpha_t} - \dot{\beta}_t \beta_t\right)\nabla_{\mathbf{x}} \log p_t(\mathbf{x}).$$

<details>
<summary style="cursor: pointer; color: #888;">Derivation: marginal path equivalence</summary>
<div style="margin-top: 10px; margin-bottom: 6px;">

<strong>Affine conditional flows:</strong> The marginal path is given by

$$p_t(\mathbf{x}) = \iint \delta\big(\mathbf{x} - \alpha_t \mathbf{x}_1 - \beta_t \mathbf{x}_0\big)\, p_{\text{source}}(\mathbf{x}_0)\, p_{\text{target}}(\mathbf{x}_1)\, \mathrm{d}\mathbf{x}_0\, \mathrm{d}\mathbf{x}_1.$$

Integrating out $\mathbf{x}_0$ gives us

$$p_t(\mathbf{x}) = \int \frac{1}{\beta_t^d}\, p_{\text{source}}\!\left(\frac{\mathbf{x} - \alpha_t \mathbf{x}_1}{\beta_t}\right) p_{\text{target}}(\mathbf{x}_1)\, \mathrm{d}\mathbf{x}_1.$$

When $p_{\text{source}} = \mathcal{N}(\mathbf{0}, \mathbf{I})$, the above expression can be written as

$$p_t(\mathbf{x}) = \int \mathcal{N}(\mathbf{x};\, \alpha_t \mathbf{x}_1,\, \beta_t^2 \mathbf{I})\, p_{\text{target}}(\mathbf{x}_1)\, \mathrm{d}\mathbf{x}_1.$$

<strong>Gaussian probability paths:</strong> By definition, we have 

$$p_t(\mathbf{x} \mid \mathbf{z}) = p_t(\mathbf{x} \mid \mathbf{x}_1) = \mathcal{N}(\mathbf{x};\, \alpha_t \mathbf{x}_1,\, \beta_t^2 \mathbf{I}).$$

Consequently, we have 

$$p_t(\mathbf{x}) = \int p_t(\mathbf{x} \mid \mathbf{x}_1)\, p_{\text{target}}(\mathbf{x}_1)\, \mathrm{d}\mathbf{x}_1 = \int \mathcal{N}(\mathbf{x};\, \alpha_t \mathbf{x}_1,\, \beta_t^2 \mathbf{I})\, p_{\text{target}}(\mathbf{x}_1)\, \mathrm{d}\mathbf{x}_1.$$

</div>
</details>

<div style="margin-bottom: 10px;"></div>

<details>
<summary style="cursor: pointer; color: #888;">Derivation: marginal vector field equivalence and its formula in terms of the score function</summary>
<div style="margin-top: 10px; margin-bottom: 6px;">

<strong>Gaussian probability paths:</strong> We have $\mathbf{z} = \mathbf{x}_1$ and 

$$\mathbf{v}_t(\mathbf{x} \mid \mathbf{x}_1) = \dot{\alpha}_t \mathbf{x}_1 + \frac{\dot{\beta}_t}{\beta_t}(\mathbf{x} - \alpha_t \mathbf{x}_1).$$

Using the integral formula for the marginal vector field and linearity of expectation, we get

$$\mathbf{v}_t(\mathbf{x}) = \mathbb{E}_{\mathbf{x}_1 \sim p(\mathbf{x}_1 \mid \mathbf{x}_t=\mathbf{x})}\!\left[\dot{\alpha}_t \mathbf{x}_1 + \frac{\dot{\beta}_t}{\beta_t}(\mathbf{x} - \alpha_t \mathbf{x}_1)\right] = \frac{\dot{\beta}_t}{\beta_t}\mathbf{x} + \left(\dot{\alpha}_t - \frac{\dot{\beta}_t \alpha_t}{\beta_t}\right)\mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}].$$

<strong>Affine conditional flows:</strong> We have $\mathbf{z} = (\mathbf{x}_0, \mathbf{x}_1)$ and 

$$\mathbf{v}_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1) = \dot{\alpha}_t \mathbf{x}_1 + \dot{\beta}_t \mathbf{x}_0.$$ 

Using the integral formula for the marginal vector field, we get

$$\mathbf{v}_t(\mathbf{x}) = \dot{\alpha}_t\, \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}] + \dot{\beta}_t\, \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}].$$

Since $\mathbf{x}_t = \alpha_t \mathbf{x}_1 + \beta_t \mathbf{x}_0$, we can take conditional expectations on both sides to get 

$$\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}] = \frac{\mathbf{x} - \alpha_t\, \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}]}{\beta_t}.$$ 

Substituting this into the expression for the marginal vector field, we get

$$\mathbf{v}_t(\mathbf{x}) = \frac{\dot{\beta}_t}{\beta_t}\mathbf{x} + \left(\dot{\alpha}_t - \frac{\dot{\beta}_t \alpha_t}{\beta_t}\right)\mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}],$$

which is the same as what we derived for Gaussian conditional flows. <br>
<br>
<strong>Tweedie's formula.</strong> Since $p_t(\mathbf{x})$ is a Gaussian mixture (shown previously), its score function is well-defined. Using Tweedie's formula, we can write

$$\mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}] = \frac{\mathbf{x} + \beta_t^2\, \nabla_{\mathbf{x}} \log p_t(\mathbf{x})}{\alpha_t}. $$

Finally, substituting the above into the previously derived expression for the marginal vector field, we get

$$\mathbf{v}_t(\mathbf{x}) = \frac{\dot{\beta}_t}{\beta_t}\mathbf{x} + \left(\dot{\alpha}_t - \frac{\dot{\beta}_t \alpha_t}{\beta_t}\right)\frac{\mathbf{x} + \beta_t^2\, \nabla_{\mathbf{x}} \log p_t(\mathbf{x})}{\alpha_t} = \frac{\dot{\alpha}_t}{\alpha_t}\mathbf{x} + \left(\frac{\dot{\alpha}_t \beta_t^2}{\alpha_t} - \dot{\beta}_t \beta_t\right)\nabla_{\mathbf{x}} \log p_t(\mathbf{x}).$$

</div>
</details>

<div style="margin-bottom: 16px;"></div>

### Some Links with Diffusion Models

I also found the links between flow matching and diffusion models to be quite interesting.

**1. The conditioning trick:** The key trick that makes training in flow matching tractable — conditioning on $$\mathbf{z}$$ — also shows up in denoising score matching. In both cases, conditioning on a known quantity makes an otherwise intractable objective easy to evaluate.

**2. Gaussian flow matching:** In Gaussian flow matching, the marginal vector field is an invertible affine reparametrization of the score function $$\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$. So flow matching and score-based diffusion models are learning the same underlying object, just in different parametrizations. Moreover, specific choices of the schedules $$\alpha_t, \beta_t$$ recover well-known diffusion model families such as the variance-preserving (VP) and variance-exploding (VE) variants as special cases.

---

## Applications to Inverse Problems

*Note: In this section, I write $$p(\mathbf{x}_t \mid \mathbf{y})$$ instead of $$p_t(\mathbf{x} \mid \mathbf{y})$$, placing the time index on the variable rather than on $$p$$. This allows me to refer to distributions at multiple time points simultaneously.*

### Setting

Inverse problems involve recovering an unknown signal $$\mathbf{x}$$ from its noisy measurements $$\mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\eta}$$, where $$\mathbf{A}$$ is a forward operator (assumed to be linear here) and $$\eta$$ is noise. Examples include image deblurring and MRI/CT reconstruction. These problems are usually ill-posed, that is, several signals could have yielded the same set of measurements. To counteract this ill-posedness, we use some prior information about the signal in the reconstruction process. In the Bayesian reconstruction framework, we specify a prior distribution $$p(\mathbf{x})$$ representing the distribution of clean signals, and we run an algorithm that draws samples from the posterior distribution $$p(\mathbf{x} \mid \mathbf{y})$$, which is the distribution of clean signals consistent with the measurements.

There are several works on using flow models as priors for inverse problems. I talk about two here — <a href="https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_FlowDPS__Flow-Driven_Posterior_Sampling_for_Inverse_Problems_ICCV_2025_paper.pdf" target="_blank" rel="noopener noreferrer">FlowDPS</a> and <a href="https://openreview.net/pdf?id=QGd34p02mI" target="_blank" rel="noopener noreferrer">FLOWER</a> — because they take different approaches to design the sampling algorithm, and I found this to be interesting.

### FlowDPS

The main idea of FlowDPS is to run the flow ODE with the **conditional vector field** $$\mathbf{v}_t(\mathbf{x} \mid \mathbf{y})$$ instead of the unconditional one. For affine conditional flows, we have previously seen the link between the vector field and the score function. Using this link and Bayes' rule, the conditional field can be decomposed as

$$\mathbf{v}_t(\mathbf{x} \mid \mathbf{y}) = \mathbf{v}_t(\mathbf{x}) + \left(\frac{\dot{\alpha}_t \beta_t^2}{\alpha_t} - \dot{\beta}_t \beta_t\right) \nabla_{\mathbf{x}} \log p(\mathbf{y} \mid \mathbf{x}_t = \mathbf{x}),$$

where the first term is the unconditional (pretrained) vector field, and the second term is a guidance term that steers the trajectory towards the measurements. The problem is that $$p(\mathbf{y} \mid \mathbf{x}_t = \mathbf{x})$$ is intractable.

We can write $$p(\mathbf{y} \mid \mathbf{x}_t = \mathbf{x})$$ as

$$p(\mathbf{y} \mid \mathbf{x}_t = \mathbf{x}) = \int p(\mathbf{y} \mid \mathbf{x}_1)\, p(\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x})\, \mathrm{d}\mathbf{x}_1.$$

FlowDPS approximates $$p(\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x})$$ as $$\delta\big(\mathbf{x}_1 - \hat{\mathbf{x}}_1(\mathbf{x}, t)\big)$$, where

$$\hat{\mathbf{x}}_1(\mathbf{x}, t) = \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}] = \frac{\beta_t\, \mathbf{v}_t(\mathbf{x}) - \dot{\beta}_t\, \mathbf{x}}{\dot{\alpha}_t \beta_t - \dot{\beta}_t \alpha_t}$$

is the posterior mean of $$p(\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x})$$. With this approximation, we get

$$\nabla_{\mathbf{x}} \log p(\mathbf{y} \mid \mathbf{x}_t = \mathbf{x}) = \nabla_{\mathbf{x}} \log p\big(\mathbf{y} \mid \hat{\mathbf{x}}_1(\mathbf{x}, t)\big).$$

The right-hand side is tractable as $$p\big(\mathbf{y} \mid \hat{\mathbf{x}}_1(\mathbf{x}, t)\big)$$ is simply the measurement likelihood evaluated at the clean image estimate. The gradient of this term can then be obtained by backpropagating through $$\hat{\mathbf{x}}_1(\mathbf{x}, t)$$.

### FLOWER

FLOWER takes a completely different approach to the posterior sampling problem. The starting point here is the observation

$$p(\mathbf{x}_{t'} \mid \mathbf{y}) = \int p(\mathbf{x}_{t'} \mid \mathbf{x}_t, \mathbf{y})\, p(\mathbf{x}_t \mid \mathbf{y})\, \mathrm{d}\mathbf{x}_t.$$

This equation implies that if we have a sample $$\bar{\mathbf{x}}_t \sim p(\mathbf{x}_t \mid \mathbf{y})$$, then a sample from $$p(\cdot \mid \mathbf{x}_t = \bar{\mathbf{x}}_t, \mathbf{y})$$ is a sample from $$p(\mathbf{x}_{t'} \mid \mathbf{y})$$. Thus, the goal is to construct $$p(\cdot \mid \mathbf{x}_t, \mathbf{y})$$ as this would allow us to generate a chain of samples going from $$p(\mathbf{x}_0 \mid \mathbf{y})$$ to the desired $$p(\mathbf{x}_1 \mid \mathbf{y})$$.

To construct $$p(\cdot \mid \mathbf{x}_t, \mathbf{y})$$, FLOWER relies on the following conditioning trick:

$$p(\mathbf{x}_{t'} \mid \mathbf{x}_t, \mathbf{y}) = \int p(\mathbf{x}_{t'} \mid \mathbf{x}_t, \mathbf{y}, \mathbf{x}_1)\, p(\mathbf{x}_1 \mid \mathbf{x}_t, \mathbf{y})\, \mathrm{d}\mathbf{x}_1 = \int p(\mathbf{x}_{t'} \mid \mathbf{x}_1)\, p(\mathbf{x}_1 \mid \mathbf{x}_t, \mathbf{y})\, \mathrm{d}\mathbf{x}_1.$$

FLOWER assumes that the conditional paths are such that the simplification $$p(\mathbf{x}_{t'} \mid \mathbf{x}_t, \mathbf{y}, \mathbf{x}_1) = p(\mathbf{x}_{t'} \mid \mathbf{x}_1)$$ holds. For example, this is true for Gaussian probability paths, where $$\mathbf{x}_{t'} = \alpha_{t'}\mathbf{x}_1 + \beta_{t'}\boldsymbol{\varepsilon}$$ with $$\boldsymbol{\varepsilon}$$ being fresh noise independent of $$\mathbf{x}_t$$, but not for affine conditional flows.

FLOWER proposes a three-step iteration, where the first two steps yield a sample from $$p(\mathbf{x}_1 \mid \mathbf{x}_t, \mathbf{y})$$, and the third step uses this sample to draw a sample from $$p(\mathbf{x}_{t'} \mid \mathbf{x}_1)$$.

---

*Last updated: April 2026*
