# lora_models
Some ideas and code for LoRA models (Low Rank Adaptation)

The code in [merging_loras](https://github.com/Amelie-Schreiber/lora_models/blob/main/merging_loras.ipynb) notebook shows how to compute the Fréchet mean (a.k.a. the Karcher mean) of the weight matrices of two LoRA models (Low Rank Adaptation models). This treats the weight matrices as points on the Grassmannian and computes their (Karcher) mean *in the Grassmannian manifold*. This effectively gives a representation of the average of the two that preserves geometric information about the two models. This could be useful for ensemble learning or knowledge distillation. Note, this notebook will work for any two LoRA models that have update matrices $\Delta W_1^{(n)} = B_1^{(n)}A_1^{(n)}$, and $\Delta W_2^{(n)} = B_2^{(n)}A_2^{(n)}$ that have the same middle dimension (here `n` denotes the layer). In other words, if $A_1^{(n)} \in \mathbb{R}^{r \times n}$ then we must have $A_2^{(n)} \in \mathbb{R}^{r \times n}$, and similarly for $B_1^{(n)}$ and $B_2^{(n)}$. If the middle dimension are not the same (and if the rank of $A_i^{(n)}$ and $B_i^{(n)}$ are not full) the Karcher mean computation will not work. Moreover, for the scalars $\alpha$ (corresponding to the suffix `.alpha` in the two `.safetensors` files) in the network, we simply compute the arithemtic mean. Note, this can be generalize to multiple LoRA models, and if the models are close together in the Grassmannian in the subspace similarity metric 

$$
\phi(\Delta W_1^{(n)}, \Delta W_2^{(n)}, i, j) = \frac{||U_{\Delta W_1^{(n)}}^{(i)} {U_{\Delta W_2^{(n)}}^{(j)}}^T||_F^2}{\min(i, j)}
$$

then this should provide a geometrically meaningful representation of the models that effectively distills them into a single model. However, since the Karcher means yield square orthonormal matrices, some additional deep learning procedure to obtain a model with the architecture of the original base model will be required. 

## What is this subspace similarity metric $\phi$?

In our effort to comprehend the intricate geometry of the parameter space in deep learning architectures, we us an innovative subspace similarity metric, denoted as $\phi$, which is designed to quantify the resemblance between subspaces defined by weight updates in deep neural networks. This metric was introduced in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). This subspace similarity metric draws from the concept of subspace angles and exploits the mathematical properties of the Grassmann manifold -- a space embodying the structure of linear subspaces within a high-dimensional space.

The inputs to the metric $\phi$ include two weight update matrices, $\Delta W_1^{(n)}$ and $\Delta W_2^{(n)}$, and two integers, $i$ and $j$. These integers represent the number of columns to consider from the corresponding singular vectors of the Singular Value Decomposition (SVD) of these weight update matrices. 

To fully appreciate the metric, let's delineate its components:

1. **Singular Value Decomposition (SVD):** The Singular Value Decomposition of a matrix $A$ is a factorization of the form $A = U\Sigma V^T$ where $U$ and $V$ are orthogonal matrices and $\Sigma$ is a diagonal matrix. In context of the weight matrices $\Delta W_1^{(n)}$ and $\Delta W_2^{(n)}$, we apply SVD to derive the orthogonal matrices $U_{\Delta W_1^{(n)}}$ and $U_{\Delta W_2^{(n)}}$. The columns of these matrices, the left singular vectors of the original matrices, form an orthonormal basis for the column space of the corresponding weight update matrices.

2. **Subspaces** $U_{\Delta W_1^{(n)}}^{(i)}$ and $U_{\Delta W_2^{(n)}}^{(j)}$: The subspaces $U_{\Delta W_1^{(n)}}^{(i)}$ and $U_{\Delta W_2^{(n)}}^{(j)}$ refer to the subspaces spanned by the first $i$ and $j$ columns of $U_{\Delta W_1^{(n)}}$ and $U_{\Delta W_2^{(n)}}$ respectively. They signify the $i$- and $j$-dimensional spaces that encapsulate the most variance in the respective weight update matrices.

3. **Frobenius Norm and Matrix Multiplication:** 

$$
\phi(\Delta W_1^{(n)}, \Delta W_2^{(n)}, i, j) = \frac{||U_{\Delta W_1^{(n)}}^{(i)} {U_{\Delta W_2^{(n)}}^{(j)}}^T||_F^2}{\min(i, j)}
$$

This segment of the metric quantifies the similarity between the two subspaces. The matrix $U_{\Delta W_1^{(n)}}^{(i)} {U_{\Delta W_2^{(n)}}^{(j)}}^T$ is the matrix formed by the dot product of every pair of singular vectors from the two subspaces. The Frobenius norm is then computed for this matrix, which computes the square root of the sum of the absolute squares of its elements, and then squared. The Frobenius norm of the product of the singular vectors captures the sum of the squared cosine of the angles between the subspaces. Squaring the Frobenius norm emphasizes the contributions from the larger angles.

4. **Normalization by** $\min(i, j)$: The Frobenius norm of the matrix product is then normalized by the minimum of $i$ and $j$. This step ensures that the metric is scale-invariant and doesn't grow with the dimensionality of the subspaces.

The subspace similarity metric $\phi(\Delta W_1^{(n)},\Delta W_2^{(n)}, i, j)$ thereby provides a rigorous, mathematically sound measure of the similarity between the subspaces spanned by the weight updates in a deep neural network. Higher values of $\phi$ imply that the weight updates are moving in similar directions in the high-dimensional space, suggesting that the learning process is exploring similar regions of the parameter space. Conversely, lower values of $\phi$ suggest that the learning process is exploring different regions of the parameter space with the two different weight update matrices.

The relevance of the Grassmann manifold in this context arises from the fact that the space of all subspaces of a given dimension in a high-dimensional space forms a Grassmann manifold. The metric $\phi$ essentially measures distances on this manifold, providing a natural measure of similarity between subspaces. 

By analyzing the trajectory of weight updates in the space of all possible subspaces (i.e., the Grassmann manifold), we gain a geometric understanding of the learning dynamics of deep neural networks. This perspective can provide new insights into the complex optimization landscapes associated with these models and can potentially inform new training strategies, regularizations, or architectures to improve learning performance.

In the context of Low Rank Adaptation (LoRA), the matrices $U_{\Delta W_1^{(n)}}$ and $U_{\Delta W_2^{(n)}}$ can be interpreted as update weight matrices. The subspace similarity metric described above can be used to analyze the similarity between different update matrices, providing insight into the training dynamics of the model. However, it's important to note that this approach is not limited to LoRA and can be applied to any neural network model.

In summary, our research leverages the mathematical properties of the Grassmann manifold to develop a novel subspace similarity metric for understanding the learning dynamics of deep neural networks. By studying the subspaces spanned by weight updates during training, we gain a geometric perspective on the high-dimensional optimization landscapes navigated by these powerful models.

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

It is also important to note, again, that this can be performed with multiple LoRA models at once, even if they are of different sizes. In particular, if we follow the methods in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) and compute the matrices $U_{A_1^{(n)}}^{i_1}$, $U_{A_2^{(n)}}^{i_2}$, ..., $U_{A_k^{(n)}}^{i_k}$  with $i_1 = i_2 = ... = i_k$, we can compute the Karcher mean of the subspaces spanned by the first $i_1 = ... = i_k$ singular columns (or rows depending on your convention) of the decomposition (SVD) of $A_1^{(n)}$, $A_2^{(n)}$, ..., $A_k^{(n)}$. So assuming the $i_r$ are all the same (r=1, ..., k)$

$$
A_{\text{mean}}^{(n)} = \arg\min_{A \in Gr(i_r, N)} \sum_{r=1}^k d^2(A, U_{A_r^{(n)}}^{i_r})
$$

Similarly for $U_{B_1^{(n)}}^{j_1}$, $U_{B_2^{(n)}}^{j_2}$, ..., $U_{B_2^{(n)}}^{j_k}$ with $j_1 = j_2 = ... = j_k$, we can compute the Karcher mean for all of them. So, choosing all of the $j_r$ to be the same (r=1, ..., k)$

$$
B_{\text{mean}}^{(n)} = \arg\min_{B \in Gr(j_r, N)} \sum_{r=1}^k d^2(B, U_{B_r^{(n)}}^{j_r})
$$

This approach is not only interesting from a practical point of view, but also from a theoretical point of view. It draws on concepts from Riemannian geometry and manifold optimization, and it could provide new insights into the geometric structure of the space of all possible adaptations to a given model. It's a fascinating direction for further research, connecting deep learning, Riemannian geometry, and optimization on Grassmann manifolds.
