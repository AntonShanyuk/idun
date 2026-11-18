# JARF: a network's gradient directions in front of the trees

`idun.jarf` composes idun_tabm and idun_gbm (which stay independent of each other) along the lines of Rauniyar (2025):
a trained network responds to combinations of its inputs, a tree splits one input at a time; give the trees the
directions the network is most sensitive to as extra features, and an axis-aligned split on such a feature acts as an
oblique split in the original inputs. The public interface is in [python-api.md](../api/python-api.md) (`idun.tabm.directions`,
`idun.jarf`); this page is the mechanism.

## 1. The gradient of the prediction

For a single-target TabM classifier without a temporal encoder, the prediction is the expected class of the
member-averaged softmax, `ŷ(x) = Σ_c c · mean_j softmax(o_j(x))_c`. Its gradient with respect to the inputs is taken on
the float reference (`idun.tabm.reference`): the logits' gradient of the expected class is `p ⊙ (c − ŷ) / k` per
member, and from there the blocks' backward pass runs as the loss's does (`da · 1[z > 0] · s`, then `(dt @ W) ⊙ r`),
without dropout; every member reads the same `x`, so the input gradient is the members' sum. A test checks it against
central finite differences in float64.

## 2. The EGOP and its directions

The expected gradient outer product `H = E[g gᵀ]` is estimated over a seeded sample of training rows and over the
surrogate networks given (a bag of seeds): rows in chunks of 512, models in order, every product summed in float64.
`H` is symmetric positive semi-definite; its eigenvectors, largest eigenvalue first, with each column's largest entry
made positive, are the directions, and its diagonal is each input's mean squared gradient, the per-input sensitivity.

## 3. The features

Each direction `v` gives one new feature per row, `x · v`, where `x` is the row as the networks see it
(`idun.nn.inputs.dense_inputs`: the named values with NaN as 0, the codes as `(code − 2) / 2`, the country one-hot; on
a generic table the `TableEncoder`'s standardized inputs). The trees read bins, so each projection is binned once: on
the packed cache at the distinct quantiles `1..254 / 255` of a sample of training rows, on a generic table by `idun.Dataset`'s own quantile rule. The directions' features are appended to the named
features; the obfuscated features may be reduced to the most sensitive ones (`keep`). The `binning` id of the result
carries the directions' digest and the kept set, so a booster refuses data extended another way.

## 4. Determinism

The EGOP repeats its bits on one machine; `eigh` (LAPACK) and the float32 projection matmul do not promise their last
bits across machines or matrix shapes ([determinism.md](determinism.md) §9). Hence the projection runs in fixed chunks
of 50,000 rows, and a `Directions` is computed once and stored with the model that uses it; fits and predictions from
stored directions are bit-exact.
