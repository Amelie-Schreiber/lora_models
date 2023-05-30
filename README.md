# lora_models
Some ideas and code for LoRA models (Low Rank Adaptation)

Don't take this repo too seriously, it's mostly just me playing with ChatGPT and trying to figure out how to use Grassmannians in an ML context. 

The code in [merging_loras](https://github.com/Amelie-Schreiber/lora_models/blob/main/merging_loras.ipynb) notebook shows how to compute the Fréchet mean (a.k.a. the Karcher mean) of the weight matrices of two LoRA models (Low Rank Adaptation models). This treats the weight matrices as points on the Grassmannian and computes their (Karcher) mean *in the Grassmannian manifold*. This effectively gives a representation of the average of the two that preserves geometric information about the two models. This could be useful for ensemble learning or knowledge distillation. Note, this notebook will work for any two LoRA models that have update matrices $\Delta W_1^{(n)} = B_1^{(n)}A_1^{(n)}$, and $\Delta W_2^{(n)} = B_2^{(n)}A_2^{(n)}$ that have the same middle dimension (here `n` denotes the layer). In other words, if $A_1^{(n)} \in \mathbb{R}^{r \times n}$ then we must have $A_2^{(n)} \in \mathbb{R}^{r \times n}$, and similarly for $B_1^{(n)}$ and $B_2^{(n)}$. If the middle dimension are not the same (and if the rank of $A_i^{(n)}$ and $B_i^{(n)}$ are not full) the Karcher mean computation will not work. Moreover, for the scalars $\alpha$ (corresponding to the suffix `.alpha` in the two `.safetensors` files) in the network, we simply compute the arithemtic mean. 

## Why?

Well, since the Grassmannian and Grassmannian manifold learning is used in image analysis, and LoRA models are seeing a surge in applications to Stable Diffusion and similar text-to-image and multimodal vision models, the Karcher mean of two models seems to make sense. The real extent of the applications to ensembe learning and knowledge distillation are still TBD. 
