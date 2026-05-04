date : 03-05

Autor : Michael Zameshaev

Status : #writing

Tags : [[momentum]], [[grid confinement]], [[inexact projections]], [[]]

# Motivation and introduction

*Philosophy of muon momentum and why it is necessary* 
	Comparing weight changes with GD, we get 
	$$
	\Delta_{\text{GD}} = U \Sigma V^T \quad \mid \quad \Delta_{\text{Muon}} = U  V^T
	$$
	This is powerful, but also dangerous. During stochastic training we do not observe the exact population gradient; we observe a finite-batch, noisy matrix. Noise can create spurious singular modes, including both small tail modes and large outlier modes. Since Muon flattens the singular spectrum of the update, weak low-SNR modes can receive disproportionately large influence compared to GD. Momentum is therefore not just a standard acceleration trick: it acts as a temporal filter before orthogonalization. Persistent singular directions accumulate in the momentum buffer, while unstable noise-induced directions are averaged out. Thus, Muon should be understood not as orthogonalizing a single noisy gradient, but as orthogonalizing a smoothed estimate of the underlying update direction.
	The danger is not only noisy singular values, but noisy singular subspaces. When singular values are close or low-SNR, the associated singular vectors can rotate substantially across batches; applying a polar map to each raw gradient would turn these rotations into large update changes.
	 Author's take : This is incredibly shitty explanation and should be rediscussed later

*Lipschitz continuity for muon analysis and why it is shitty*
	Consider a loss function, where $\|\nabla L(W) - \nabla L(W')\|_F \le c\|W - W'\|_F$
	We can derive bounds on weight changes ($P := \text{any projectoin func}$)
	$$
	L(W_{t + 1}) \leq L(W_t) + (G_t, -\beta \cdot P(G_t)) + \frac{c\beta^2}{2} \|P(G_t)\|_F^2.
	$$
	Which can be simplified as
	$$
	L(W_{t + 1}) \leq L(W_t) - \beta \|G_t\|_* + \frac{c\beta^2}{2} (d_1 \wedge d_2).
	$$
	Where bounds of paper are derived
	$$
	\min_{0 \le t < T} \|\nabla L(W_t)\|_* \le \frac{L(W_0)-L^\star}{T\beta}+\frac{c}{2}d\beta.
	$$
	 Although it helps to derive worst bounds, assumption neglects other important peculiarities of muon, as inexact projections (aka NS), grid confinement and spectre dependence
	Author's take (consider later) :  lipshitz constant can change as we approach minima

*Why even bothering considering quadratic functions since linear layers are linear?*
	 Let's look back at original blogpost ^[https://jeremybernste.in/writing/deriving-muon]
	 We use approximation of loss $L(W)$ using first order taylor and $\|\Delta y\|$ linear weight changes, so practically we are stuck with complex loss, so why don't we approximate it with simple quadratics, and work only with gradients and hessians

# Main work
We start off with some quadratic functions, (I'll add up our linear ones)
- (1) $L(W) = 0.5 \|W\|_F^2 \quad \nabla L(W) = U V^T$
- (2) $L(W) = 0.5 W^T A W + B W + C\quad \nabla L(W) = P(A W + B)$ ($A \in \mathbb{S}$)
- (3) $L(W) = 0.5 \|X W - Y\|^2_F \quad \nabla L(W) = P(X^T X W + X^T Y)$ basically the same as (2), actually (1) sampled from this loss
- (4) $L(W) = 0.5 \|X W - Y\|^2_F + \|W\|_* = P(X^T X W + X^T Y + U_W V^T_W)$

*Some cool facts*
- (1) momentum does not help in grid confinement (duh)
- (1) singular values change by $\alpha$ and in loss they occur as $0.5 \text{Tr} (W^T W) = \sum \sigma_i^2$ which leads to $O(\sqrt \varepsilon)$ proximity
- (1) GD outperforms muon on strong convexity $O(log(\frac1\varepsilon))$ VS $O(\sqrt \frac 1 \varepsilon)$

*Finite-budget*
Okay, muon sucks! But... if we iterate it infinitely, what about finite budget, for example $T = 500$. Maybe the constant of gradient descent affects convergence more than big-O?
MOMO lectures offer these constants
- mu-convexity & Lipshitz - $C = \frac L \mu = \kappa$
- convexity & Lipshitz - $C = L$ (with $O(\sqrt \frac 1 \varepsilon)$)
- Lipshitz - $C = L$ (with $O(\sqrt \frac 1 \varepsilon)$)
Still, using this constants offers us a worst-cast scenario, what about average-case (what if the change not only the $\kappa$ but entire $\Sigma$ overall)?
_TODO_ : read papers ^[https://arxiv.org/pdf/2002.04756] and ^[ https://arxiv.org/abs/2506.15054] - on singular analysis of convergence speeds

*Parts on random sampling*
Matricies are derived from $A = Q^T \Sigma Q \quad Q \mid  Q R = \mathcal{N}(0, I)$ which gives $Q$ Haar distribution 
Also there is other sampling variants : Wishart-distribution, Low-rank pertrubations

*Experiments*
Using sampling above we can control $\Sigma$ values and observe if muon actually wins in finite budget steps $:= T$  on task (2)
(1 - muon wins all the time, 0 - GD wins all the time), ALSO! $\kappa := 1e4$ 

| kind of $\Sigma$ distribution | t = T/10 | t = T/2 | t = T |
| :---------------------------- | :------: | :-----: | :---: |
| max_spiked                    |    0     |    0    |   0   |
| min_spiked                    |    1     |    1    |   1   |
| uniform                       |    0     |    0    |   0   |
| gaussian                      |    0     |    0    |   0   |
| linear_decay_to_max           |    0     |    0    |   0   |
| u_shaped                      |    0     |    0    |   1   |
| geometric_decay_to_max        |    1     |    1    |   1   |
_Observation_ : min_spiked and geometric_decay are prefferrable to muon, which can be explained as having trouble finding best learning rate (big for small eigenvalues and small for big eigenvalues)
This also can explain why beta-distributed eigenvalues help muon in the end (GD chooses lr step to be best at steepest inclines and forgets about small eigenvalues at the end of convergence where muon finally catches up)

_Fun (useful actually) fact!_ distribution of eigenvalues actually observed in applied neural networks! Which explains muon's effectiveness. For example ^[https://arxiv.org/pdf/1706.04454]

Plot for educational purposes : observing weight changes after last step $T = 500$
![https://i.ibb.co/ccwXNSHs/image.png](https://i.ibb.co/ccwXNSHs/image.png)

_Does NS help converge better?_
_TODO_

_Why Newton-Schultz is better than exact projections_ (Paper does not answer it properly imho)
Let's add Gaussian noise to the spectre! (This approach can be quite controversial since noise from sampling doesn't work like that)
One of the problems: adding values to the spectre can't be independent since we divide the gradient by the norm before projecting. 
!This logic can also be applied to usual normalisation (big eigenvalues make small values smaller)

Using toy example (1) we get some obvious results (carefully chosen noise dispersion allows muon to break grid confinement) 
![](https://i.ibb.co/4RCffgDs/image.png)
There are some plots which show that min_spiked sometimes works better with added noise (probably due to having trouble in the grid confinement quite early), this also works with max_ spiked, since it has quite a small eigenvalue in the end.
# Results

References and links :
