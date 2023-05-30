# lora_models
Some ideas and code for LoRA models (Low Rank Adaptation)

The code in [merging_loras](https://github.com/Amelie-Schreiber/lora_models/blob/main/merging_loras.ipynb) notebook shows how to compute the Fréchet mean (a.k.a. the Karcher mean) of the weight matrices of two LoRA models (Low Rank Adaptation models). This treats the weight matrices as points on the Grassmannian and computes their (Karcher) mean *in the Grassmannian manifold*. This effectively gives a representation of the average of the two that preserves geometric information about the two models. This could be useful for ensemble learning or knowledge distillation. Note, this notebook will work for any two LoRA models that have update matrices $\Delta W_1^{(n)} = B_1^{(n)}A_1^{(n)}$, and $\Delta W_2^{(n)} = B_2^{(n)}A_2^{(n)}$ that have the same middle dimension (here `n` denotes the layer). In other words, if $A_1^{(n)} \in \mathbb{R}^{r \times n}$ then we must have $A_2^{(n)} \in \mathbb{R}^{r \times n}$, and similarly for $B_1^{(n)}$ and $B_2^{(n)}$. If the middle dimension are not the same (and if the rank of $A_i^{(n)}$ and $B_i^{(n)}$ are not full) the Karcher mean computation will not work. Moreover, for the scalars $\alpha$ (corresponding to the suffix `.alpha` in the two `.safetensors` files) in the network, we simply compute the arithemtic mean. Note, this can be generalize to multiple LoRA models, and if the models are close together in the Grassmannian in the subspace similarity metric 

$$
\phi(\Delta W_1^{(n)}, \Delta W_2^{(n)}, i, j) = \frac{||U_{\Delta W_1^{(n)}}^{(i)} {U_{\Delta W_2^{(n)}}^{(j)}}^T||_F^2}{\min(i, j)}
$$

then this should provide a geometrically meaningful representation of the models that effectively distills them into a single model. However, since the Karcher means yield square orthonormal matrices, some additional deep learning procedure to obtain a model with the architecture of the original base model will be required. 

## Why?

Well, since the Grassmannian and Grassmannian manifold learning is used in image analysis, and LoRA models are seeing a surge in applications to Stable Diffusion and similar text-to-image and multimodal vision models, the Karcher mean of two models seems to make sense. Also, the computation of the Karcher mean of two (or more) models doesn't appear to be very computationally expensive, as all computations in the above, for two relatively large LoRA models was carried out on a small personal laptop. The real extent of the applications to ensembe learning and knowledge distillation are still TBD. 

In the Low-Rank Adaptation (LoRA) setting, the update matrices $\Delta W_1^{(n)}$ and $\Delta W_2^{(n)}$ are decomposed into $A$ and $B$ matrices, $\Delta W_1^{(n)} = B_1^{(n)}A_1^{(n)}$ and $\Delta W_2^{(n)} = B_2^{(n)}A_2^{(n)}$.

The $A_i^{(n)}$ and $B_i^{(n)}$ matrices can be viewed as residing in different subspaces of the original layer's weight space. They capture different aspects of the task-specific adaptation: $A_i^{(n)}$ captures the "direction" of the adaptation in the weight space, while $B_i^{(n)}$ captures how much the weights are adjusted along this direction.

If we compute the Karcher mean of the $A$ matrices and the $B$ matrices separately, we're essentially finding the "average" direction and "average" adjustment magnitude across the two tasks. This could be beneficial in a multitask learning setting, where we want to find a "compromise" adaptation that performs well on multiple tasks.

From a mathematical perspective, let's denote the Karcher mean of the $A$ matrices as $A_{\text{mean}}^{(n)}$, and the Karcher mean of the $B$ matrices as $B_{\text{mean}}^{(n)}$. They can be computed by solving the following optimization problems:

$$
A_{\text{mean}}^{(n)} = \arg\min_{A \in Gr(d, N)} \left[ d^2(A, A_1^{(n)}) + d^2(A, A_2^{(n)}) \right]
$$

$$
B_{\text{mean}}^{(n)} = \arg\min_{B \in Gr(d, N)} \left[ d^2(B, B_1^{(n)}) + d^2(B, B_2^{(n)}) \right]
$$

where $d$ denotes the geodesic distance on the Grassmannian, and $Gr(d, N)$ is the Grassmannian of $d$-dimensional subspaces in $\mathbb{R}^N$. 

Once we've computed $A_{\text{mean}}^{(n)}$ and $B_{\text{mean}}^{(n)}$, we can form an "average" update matrix $\Delta W_{\text{mean}}^{(n)} = B_{\text{mean}}^{(n)}A_{\text{mean}}^{(n)}$. This matrix cannot then just be added to the original weights of the $n$-th layer, allowing us to perform a kind of "averaged" low-rank adaptation that might perform well on both tasks, because it is not of the correct shape. However, it can be used to *learn* an average update matrix $\Delta ' W_{\text{mean}}^{(n)}$ that is of the correct shape. 

This approach is not only interesting from a practical point of view, but also from a theoretical point of view. It draws on concepts from Riemannian geometry and manifold optimization, and it could provide new insights into the geometric structure of the space of all possible adaptations to a given model. It's a fascinating direction for further research, connecting deep learning, Riemannian geometry, and optimization on Grassmann manifolds.
