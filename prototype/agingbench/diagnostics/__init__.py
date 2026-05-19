"""
agingbench/diagnostics — P1/P2/P3 diagnostic error partitioning.

Implements §5.2: counterfactual interventions that decompose the total
system error (1 − Acc_P1) into three mutually exclusive components:

  Utilization Error  = 1 − Acc_P3         → Revision Aging   (𝒰 failure)
  Write Error        = Acc_P3 − Acc_P2    → Compression Aging (𝒲 failure)
  Read Error         = Acc_P2 − Acc_P1    → Interference Aging (ℛ failure)
"""
