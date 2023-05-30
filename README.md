# lora_models
Some ideas and code for LoRA models (Low Rank Adaptation)

The code in [merging_loras](https://github.com/Amelie-Schreiber/lora_models/blob/main/merging_loras.ipynb) notebook shows how to compute the Fréchet mean (a.k.a. the Karcher mean) of the weight matrices of two LoRA models (Low Rank Adaptation models). This treats the weight matrices as points on the Grassmannian and computes their (Karcher) mean *in the Grassmannian manifold*. This effectively gives a representation of the average of the two that preserves geometric information about the two models. This could be useful for ensemble learning or knowledge distillation. Note, this notebook will work for any two LoRA models that have update matrices $\Delta W_1^{(n)} = B_1^{(n)}A_1^{(n)}$, and $\Delta W_2^{(n)} = B_2^{(n)}A_2^{(n)}$ that have the same middle dimension (here `n` denotes the layer). In other words, if $A_1^{(n)} \in \mathbb{R}^{r \times n}$ then we must have $A_2^{(n)} \in \mathbb{R}^{r \times n}$, and similarly for $B_1^{(n)}$ and $B_2^{(n)}$. If the middle dimension are not the same (and if the rank of $A_i^{(n)}$ and $B_i^{(n)}$ are not full) the Karcher mean computation will not work. Moreover, for the scalars $\alpha$ (corresponding to the suffix `.alpha` in the two `.safetensors` files) in the network, we simply compute the arithemtic mean. Note, this can be generalize to multiple LoRA models, and if the models are close together in the Grassmannian in the subspace similarity metric 

$$
\phi(\Delta W_1^{(n)}, \Delta W_2^{(n)}, i, j) = \frac{||U_{\Delta W_1^{(n)}}^{(i)} {U_{\Delta W_2^{(n)}}^{(j)}}^T||_F^2}{\min(i, j)}
$$

then this should provide a geometrically meaningful representation of the models that effectively distills them into a single model. However, since the Karcher means yield square orthonormal matrices, some additional deep learning procedure to obtain a model with the architecture of the original base model will be require. 

## Why?

Well, since the Grassmannian and Grassmannian manifold learning is used in image analysis, and LoRA models are seeing a surge in applications to Stable Diffusion and similar text-to-image and multimodal vision models, the Karcher mean of two models to make sense. Also, the computation of the Karcher mean of two (or more) models doesn't appear to be very computationally expensive, as all computations in the above, for two relatively large LoRA models was carried out on a small personal laptop. The real extent of the applications to ensembe learning and knowledge distillation are still TBD. 
