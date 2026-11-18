# Changelog

## 2.0.0 (2026-11-18)

- **idun_gbm**: histogram gradient boosting with LightGBM's model and parameter names, on the CPU, Metal and CUDA,
  every backend building the same trees bit for bit. Depth-wise and leaf-wise trees, quantized gradients, LightGBM's
  categorical splits and regularisers (`lambda_l1`, `lambda_l2`, `min_gain_to_split`, `min_sum_hessian_in_leaf`,
  `max_delta_step`), `extra_trees` and `feature_fraction_bynode`, second-order splits for Newton objectives
  (`second_order`), custom objectives (`Objective`, `NewtonObjective`), binary classification (`GBMClassifier`,
  `LogLoss`) and multi-class boosting (`train_multiclass`, `Softmax`), sample weights, validation sets with early
  stopping and callbacks, continued fits (`init_model`), seed bags (`train_seeds`, one fit per GPU on CUDA), lockstep
  fits that share each pass over the data (`train_many`), and TreeSHAP contributions and leaf indices
  (`Booster.predict_contributions`, `Booster.predict_leaves`).
- **idun_tabm**: TabM (Gorishniy et al., ICLR 2025) on Metal, CUDA and a float64 reference: classification,
  regression and HL-Gauss regression, piecewise-linear embeddings of the numeric inputs, gradient clipping,
  learning-rate schedules, sample weights, validation with patience, checkpoints that resume bit for bit, a loss and a
  batch plan of the caller's, a temporal encoder (`encoder="gru"`, `Sequences`), bfloat16 storage and micro-batches.
- **idun_realmlp**: RealMLP-TD (Holzmüller et al., NeurIPS 2024) as pytabkit builds it, on Metal, CUDA and a float64
  reference: its schedules and learning-rate factors, the activation (`act`), `hidden_sizes`, categorical embeddings,
  pytabkit's preprocessing subsample (`init_rows=0`), sample weights, validation and seed bags.
- **JARF** (`idun.jarf`): idun_gbm behind the directions a trained idun_tabm is most sensitive to, as one estimator
  (`JARFRegressor`) and as its pieces (`idun.tabm.fit_directions`, `idun.jarf.extend`, `idun.jarf.keep`).
- **Data**: `idun.Dataset` for tables in memory (numeric, 4-bit code and categorical columns; LightGBM's `max_bin` and
  `subsample_for_bin`), and packed datasets for tables larger than memory (`idun.data.PackedWriter`,
  `idun.data.PackedDataset`), memory-mapped and wrapped by Metal without a copy.
- **Estimators**: `GBMRegressor`, `GBMClassifier`, `TabMClassifier`, `TabMRegressor`, `RealMLPClassifier`,
  `RealMLPRegressor` and `JARFRegressor` follow scikit-learn's conventions without depending on it; `idun.load` reads
  any saved estimator back.
- **idun-numerai**: Numerai's data downloaded and packed (`idun_numerai.data.build_cache`, with the target, the refit
  dates and the purge as arguments), walk-forward refits, labels, Numerai's per-era scoring functions in NumPy
  (`idun_numerai.score`), and model uploads that run with NumPy alone (`idun_numerai.upload`).
- **Platforms**: Python 3.12 to 3.14. Metal on Apple silicon with macOS 15 or later; CUDA on Linux with an NVIDIA GPU
  of compute capability 7.5 or newer and NVRTC; the CPU everywhere.
