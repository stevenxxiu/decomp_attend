# A Decomposable Attention Model for Natural Language Inference
This is an attempt to reproduce the paper using tensorflow.

## Some issues
Why hash OOV terms to 1 in 100 vectors? Wouldn't 1 vector alone be better?

If embeddings are project down, then this is equivalent to altered weights in $F$, in the case of non intra-sentence attention, and to altered weights in $F_intra$, in the case of intra-sentence attention, so why project down at all? We do not project down for these reasons.

## Unclear parts
F, G, H are probably all ReLU networks but this isn't specified in the paper.
