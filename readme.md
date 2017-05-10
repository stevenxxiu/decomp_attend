# A Decomposable Attention Model for Natural Language Inference
This is an attempt to reproduce the paper using tensorflow.

## Some unexplained parts
Projecting the embeddings down does not matter if we only had $F$ but does matter for $G$. Let $G$ have top half $G_1$ and bottom half $G_2$, then we require $u\cdot G_1 + v\cdot M_2 = u\cdot A\cdot B_1 + v\cdot A\cdot B_2$, so then $A\cdot [B_1 B_2] = [M_1 M_2]$, but LHS has rank $\le 200$ and RHS can have rank $300$. For the entire model the total # of parameters are in fact identical with or without the projection matrix for $n = 200$.

## Unclear parts
How is the null embedding initialized? Is it trained? We suppose it is random and not trained like the other embeddings.

F, G, H are probably all ReLU networks but this isn't specified in the paper. It is also somewhat unclear that they are 2 layer ReLU networks, as this matches the total # of parameters: $301 * 200 + 201 * 200 + 601 * 200 + 201 * 200 + 401 * 200 + 201 * 200 + 201 * 3$. This gives us better results.

Not sure why paper states that they use the non-binary parse tokens as data when the binary parse and non-binary parse, but the tokens are the same when compared no matter the parse.

It is unclear what consists of OOV, we suppose it is any vector that is not a GloVe vector.

The OOV vectors are also probably normalized to l2 norm of 1 as this gives much better results.
