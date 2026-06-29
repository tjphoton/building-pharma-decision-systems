# Chapter 6 Exercise Solutions

Each solution stays compact and ends with the judgment required for real data.



```python
from pathlib import Path
import importlib
import sys

import pandas as pd

ROOT = Path.cwd().resolve()
while not (ROOT / "ch06_hcp").exists():
    if ROOT.parent == ROOT:
        raise FileNotFoundError("Run this notebook inside the repository.")
    ROOT = ROOT.parent

SCRIPT_DIR = ROOT / "ch06_hcp" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

analysis_module = importlib.import_module("run_analysis")
targeting = importlib.import_module("targeting")
referral = importlib.import_module("referral_network")
segmentation = importlib.import_module("segmentation")

results = analysis_module.run_analysis(ROOT)

```

## Exercise 1: Change the referral window



```python
episodes_45 = referral.prepare_referral_episodes(
    results["referral_episodes"], transition_days=45
)
graph_45, _ = referral.build_referral_graph(episodes_45)
metrics_45 = referral.referral_centrality(
    graph_45, analysis_module.load_inputs(ROOT)["hcp_account_affiliations"]
)
top_45 = set(metrics_45.head(20).npi)
top_60 = set(results["referral_metrics"].head(20).npi)
print(f"Top-20 overlap: {len(top_45 & top_60)}/20")
print(f"Only at 45 days: {sorted(top_45 - top_60)}")

```

    Top-20 overlap: 18/20
    Only at 45 days: ['9000000211', '9000000631']


**Methods note:** Window sensitivity belongs in the pathway brief. In real data, confirm that the transition window matches the clinical pathway and source latency before changing an HCP plan.


## Exercise 2: Refit k-means with k = 3



```python
inputs = analysis_module.load_inputs(ROOT)
features, matrix, _ = segmentation.prepare_segmentation_features(
    results["hcp_features"], inputs["engagement_signals"]
)
evaluation = segmentation.evaluate_cluster_counts(matrix)
comparison = evaluation.loc[evaluation.k.isin([3, 4]), [
    "k", "silhouette", "minimum_cluster_size",
    "seed_stability_ari", "bootstrap_stability_ari",
]]
print(comparison)

```

       k  silhouette  minimum_cluster_size  seed_stability_ari  \
    0  3    0.641224                    14                 1.0   
    1  4    0.763294                     9                 1.0   
    
       bootstrap_stability_ari  
    0                   0.9474  
    1                   1.0000  


**Methods note:** The 4-cluster model has stronger silhouette and bootstrap stability than the 3-cluster model. A real deployment also needs blinded business review of the centroid profiles.


## Exercise 3: Review one KOL candidate



```python
candidate = results["kol_profiles"].query("npi == '9000000363'")
transparency = results["kol_transparency_review"].query(
    "npi == '9000000363'"
)
print(candidate[[
    "npi", "research_percentile", "leadership_percentile",
    "practice_expertise_percentile", "peer_connection_percentile",
    "proposed_role", "role_fit_score", "review_status",
]].reset_index(drop=True))
print(transparency.reset_index(drop=True))

```

              npi  research_percentile  leadership_percentile  \
    0  9000000363                100.0                  100.0   
    
       practice_expertise_percentile  peer_connection_percentile  \
    0                          100.0                   66.666667   
    
                          proposed_role  role_fit_score  \
    0  Evidence-generation collaborator           100.0   
    
                         review_status  
    0  Medical-affairs review required  
              npi                     proposed_role  \
    0  9000000363  Evidence-generation collaborator   
    
                       review_status_x  total_payment_amount  payment_records  \
    0  Medical-affairs review required                   0.0                0   
    
      payment_categories  latest_payment_year review_status_y  \
    0                NaN                  NaN             NaN   
    
                                        transparency_use  
    0  Disclosure context only; excluded from scienti...  


**Judgment:** The scientific role requires medical-affairs confirmation. The commercial action and speaker-program eligibility require their own governed workflows. In real data, request source-level identity-match evidence before acting.

