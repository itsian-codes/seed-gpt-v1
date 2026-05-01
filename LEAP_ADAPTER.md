# How to adapt this SEED-GPT prototype to LEAP

The runnable code in this package is self-contained and uses dense synthetic graphs so that it can be executed immediately. To connect it to LEAP:

1. Replace `seedgpt.data.GraphBatch` with the graph object used by LEAP/PyG.
2. Replace `GINEncoder.forward(x, adj, mask)` with LEAP's pretrained GNN forward call.
3. Keep the SEED-GPT modules:
   - `HyperPromptGenerator`
   - `PrincipalPolicy`
   - `AdversarialPolicy`
   - `TransitionModel`
   - `TrajectoryEvaluator`
4. During the real interaction phase, call the frozen pretrained GNN after applying `x + prompt`.
5. During the synthetic generation phase, do not call the frozen GNN. Use `TransitionModel` and `TrajectoryEvaluator`.
6. For the `w/o SEG & TE` ablation, disable `synthetic_generation_phase`.
7. For the `w/o Adversarial Policy` ablation, set `use_adversarial=False`.
8. For the `w/o ECR` ablation, set `use_ecr=False`.

The class and function names are designed to mirror the manuscript so that the implementation can be cited directly in method descriptions.
