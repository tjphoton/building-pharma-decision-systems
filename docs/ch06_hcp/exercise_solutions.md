# Chapter 6 Exercise Solutions

Each solution stays compact and ends with the judgment required for real data.



```python
from pathlib import Path
import importlib
import sys

import pandas as pd
from IPython.display import display

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
display(comparison)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>silhouette</th>
      <th>minimum_cluster_size</th>
      <th>seed_stability_ari</th>
      <th>bootstrap_stability_ari</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0.641224</td>
      <td>14</td>
      <td>1.0</td>
      <td>0.9474</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>0.763294</td>
      <td>9</td>
      <td>1.0</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>


**Methods note:** The 4-cluster model has stronger silhouette and bootstrap stability than the 3-cluster model. A real deployment also needs blinded business review of the centroid profiles.


## Exercise 3: Review one KOL candidate



```python
candidate = results["kol_profiles"].query("npi == '9000000363'")
transparency = results["kol_transparency_review"].query(
    "npi == '9000000363'"
)
display(candidate[[
    "npi", "research_percentile", "leadership_percentile",
    "practice_expertise_percentile", "peer_connection_percentile",
    "proposed_role", "role_fit_score", "review_status",
]])
display(transparency)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>npi</th>
      <th>research_percentile</th>
      <th>leadership_percentile</th>
      <th>practice_expertise_percentile</th>
      <th>peer_connection_percentile</th>
      <th>proposed_role</th>
      <th>role_fit_score</th>
      <th>review_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>9000000363</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>66.666667</td>
      <td>Evidence-generation collaborator</td>
      <td>100.0</td>
      <td>Medical-affairs review required</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>npi</th>
      <th>proposed_role</th>
      <th>review_status_x</th>
      <th>total_payment_amount</th>
      <th>payment_records</th>
      <th>payment_categories</th>
      <th>latest_payment_year</th>
      <th>review_status_y</th>
      <th>transparency_use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>9000000363</td>
      <td>Evidence-generation collaborator</td>
      <td>Medical-affairs review required</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Disclosure context only; excluded from scienti...</td>
    </tr>
  </tbody>
</table>
</div>


**Judgment:** The scientific role requires medical-affairs confirmation. The commercial action and speaker-program eligibility require their own governed workflows. In real data, request source-level identity-match evidence before acting.

