# A Decomposable Attention Model for Natural Language Inference
This is an attempt to reproduce the paper using tensorflow.

## Some issues in the paper
Why hash OOV terms to 1 in 100 vectors? Wouldn't 1 vector alone be better?

If embeddings are project down to dimensions >= F's output dimension, then this is equivalent to altered weights in $F$, in the case of non intra-sentence attention, and to altered weights in $F_intra$, in the case of intra-sentence attention, so why project down at all? We do not project down for these reasons.

## Unclear parts
How is the null embedding initialized? Is it trained? We suppose it is random and not trained like the other embeddings.

F, G, H are probably all ReLU networks but this isn't specified in the paper. It is also somewhat unclear that they are 2 layer ReLU networks, as this matches the total # of parameters: $301 * 200 + 201 * 200 + 601 * 200 + 201 * 200 + 401 * 200 + 201 * 200 + 201 * 3$. This gives us better results.

Not sure why paper states that they use the non-binary parse tokens as data when the binary parse and non-binary parse, but the tokens are the same when compared no matter the parse.
