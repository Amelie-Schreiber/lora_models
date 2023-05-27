# lora_models
Some ideas and code for LoRA models (Low Rank Adaptation)

## Another Approach to Merging

*Suppose we are are studying the update weight matrices $\Delta W_1^{(n)} = B_1^{(n)}A_1^{(n)}$ for layer `n`of a reference LoRA model (Low Rank Adaptation) named `lora1.safetensors` or just `lora1`. Suppose further we have a second LoRA named `lora2` (also stored as a .safetensors file) with update weight matrices $\Delta W_2^{(n)} = B_2^{(n)}A_2^{(n)}$. Now, define a subspace similarity metric $\phi(\Delta W_i, \Delta W_j, k, l) = \frac{\left\|U_{\Delta W_i^{(n)}}^{(k)} {U_{\Delta W_j^{(n)}}^{(l)}}^T\right\|_F^2}{\min(k, l)}$ that helps us measure how far apart two update weight matrices are in the. Now please explain how this related to points in the Grassmannian (or on the "Grassmann manifold"). Please explain how the Grassmannian is realized as a quotient space (a homogeneous space) in terms of orthogonal groups. Please also explain how we might use this to define and study the equivalence classes of matrices corresponding to the update weight matrices of the two LoRA models. Please also explain how we might use this to replace to update weight matrices of `lora2` with equivalent matrices. How will these new matrices compare in the subspace similarity metric $\phi(\Delta W_i, \Delta W_j, k, l)$ we defined? If we wish to merge the two LoRA models in some way, how can we use the theory and properties of the Grassmannian to do so?*

Let's start by discussing the Grassmannian, a space that parametrizes all k-dimensional subspaces in an n-dimensional vector space. The Grassmannian, often denoted as G(k, n), can be thought of as a set of points, each representing a unique k-dimensional subspace in n-dimensional space.

The Grassmannian is realized as a quotient space (a homogeneous space) in terms of orthogonal groups as follows: An orthogonal group, O(n), is the group of all orthogonal n×n matrices (those that preserve the dot product in Euclidean space). It includes rotations and reflections. The special orthogonal group, SO(n), is the subgroup consisting of rotations only. The Stiefel manifold, V(k, n), is the space of all orthonormal k-frames in R^n, which is a subset of O(n). It can be thought of as the space of all ordered sets of k orthogonal unit vectors in R^n.

When we mod out the Stiefel manifold by the action of the orthogonal group O(k) (the group of all orthogonal k×k matrices), we get the Grassmannian. In other words, the Grassmannian G(k, n) is the quotient space V(k, n) / O(k). The action of O(k) on V(k, n) is to change the basis of each k-subspace in R^n without changing the subspace itself. Hence, each point in the Grassmannian represents an equivalence class of k-frames that span the same k-subspace in R^n.

Now, let's relate this to the problem involving the update weight matrices of the two LoRA models.

In your context, the update weight matrices ΔW_i^(n) and ΔW_j^(n) correspond to linear transformations in some high-dimensional space. The subspaces of interest here are the column spaces of these matrices. The subspace similarity metric φ(ΔW_i, ΔW_j, k, l) measures how "close" these column spaces are to each other in terms of their principal subspaces spanned by the first k and l singular vectors respectively. This notion of "closeness" can be thought of as a distance in the Grassmannian.

When you talk about replacing the update weight matrices of lora2 with "equivalent" matrices, you're essentially discussing a change of basis within each subspace. In terms of the Grassmannian, you're moving within an equivalence class. The subspace similarity metric φ(ΔW_i, ΔW_j, k, l) should remain the same, because the metric is a function of the subspaces themselves, not the specific bases (i.e., matrices) that you choose to represent them.

If you wish to "merge" the two LoRA models, one possibility could be to find a common subspace that best represents the column spaces of both models' update weight matrices. This common subspace can be thought of as a point in the Grassmannian that is "close" to the points representing the column spaces of both models. How exactly to find such a common subspace would depend on the specifics of your problem, but in general, you could use techniques such as principal angles (canonical correlations) or other Grassmannian-based optimization methods to do this. 

The Grassmannian's geometry and algebraic structure (e.g., its topology as a homogeneous space and its structure as a manifold) can provide a rich mathematical framework for studying these kinds of problems in machine learning.

*Let's discuss the merging process more. How might one create a third LoRA model `lora3`, that have update weight matrices $\Delta W_3^{(n)} = B_3^{(n)}A_3^{(n)}$ that has maximum overlap with both $\Delta W_1^{(n)} = B_1^{(n)}A_1^{(n)}$ and $\Delta W_2^{(n)} = B_2^{(n)}A_2^{(n)}$ for each layer `n`? That is, so that each $\Delta W_3^{(n)}$ is close to $\Delta W_1^{(n)}$ and $\Delta W_2^{(n)}$ in the Grassmannian manifold for all $n$?*

One approach to creating a third LoRA model `lora3` that has maximum overlap with both `lora1` and `lora2` is to find a common subspace for each layer `n` that is close to the subspaces represented by ΔW_1^(n) and ΔW_2^(n) in the Grassmannian manifold. This could be achieved by solving an optimization problem that minimizes the distance (in terms of the subspace similarity metric φ) between the column spaces of ΔW_3^(n), ΔW_1^(n), and ΔW_2^(n). 

This optimization problem can be formalized as:

Minimize: $\phi(\Delta W_3^{(n)}, \Delta W_1^{(n)}, k, l) + \phi(\Delta W_3^{(n)}, \Delta W_2^{(n)}, k, l)$
Subject to: $\Delta W_3^{(n)} = B_3^{(n)}A_3^{(n)}$

Here, $B_3^{(n)}$ and $A_3^{(n)}$ are weight matrices that need to be determined. 

The minimization can be done using gradient descent or other optimization techniques, where the gradient of the objective function can be computed using the properties of the Grassmannian manifold.

This method would give you a new set of weight matrices for `lora3` that represent a common subspace for each layer `n`, which is close to the subspaces represented by `lora1` and `lora2`.

However, keep in mind that this process might not always yield a meaningful or useful model, because the common subspace that minimizes the distance might not preserve the essential features that `lora1` and `lora2` have learned.

Also note that this is a somewhat simplified approach and does not take into account the structure of the neural network models or the nature of the data they have been trained on. In practice, creating a merged model that performs well might involve more complex techniques, like transfer learning or knowledge distillation, and might require a careful examination of the models' architectures and training histories.
