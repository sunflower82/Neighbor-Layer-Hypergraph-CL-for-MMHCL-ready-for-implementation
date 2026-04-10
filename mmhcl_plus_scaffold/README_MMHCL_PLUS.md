# MMHCL+ scaffold

This scaffold implements the revision43 design as a standalone package that can be dropped into the current MMHCL repository.

Included components:
- Two-stage contrastive learning trainer.
- Neighbor-layer positive pair construction.
- Expanded projector for the u2u branch.
- Chunked / temperature-free InfoNCE utilities.
- Barlow Twins and soft BYOL alignment.
- Dynamic EMA teacher and W_ema bank.
- Soft topology-aware purification with ANN abstraction.
- Profiling-guided checkpointing helpers.
- Dirichlet regularization with mini-batch sparse block support.

## Expected integration points with the current MMHCL repo

1. Replace the demo branch encoders in `train_mmhcl_plus.py` with wrappers around the current MMHCL user-user and item-item branches.
2. Route real batch tensors from the repo's dataloader into the batch dictionary used by `TwoStageTrainer.train_step`.
3. Connect the real sparse hypergraph Laplacian block getter to `dirichlet_energy_minibatch`.
4. Optionally replace the simple `FusionMLP` with the repo's original fusion operator.

## Batch dictionary expected by the trainer

```python
batch = {
    'x': input_embeddings_or_features,
    'user_online': user_branch_online_embeddings,      # optional if using repo-specific flow
    'item_online': item_branch_online_embeddings,      # optional if using repo-specific flow
    'pos_scores': pos_scores,
    'neg_scores': neg_scores,
    'node_idx': node_indices_for_dirichlet,
    'lap_getter': callable_returning_block_laplacian,
    'bridge_score': optional_bridge_score,
}
```

## Run demo

```bash
python train_mmhcl_plus.py --config configs/mmhcl_plus.yaml
```
