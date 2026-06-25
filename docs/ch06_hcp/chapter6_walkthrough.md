# Chapter 6 Walkthrough: HCP Targeting

This executed notebook builds the Chapter 6 artifacts at the December 31, 2024 cutoff. Run `ch06_hcp/scripts/generate_ch06_data.py` before rebuilding the notebook.



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

headline = pd.Series({
    "Journey patients": results["attribution_comparison"].patient_id.nunique(),
    "Eligible-roster patients": results["patient_hcp"].patient_id.nunique(),
    "Eligible HCPs": results["hcp_features"].npi.nunique(),
})
display(headline.to_frame("count"))

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
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Journey patients</th>
      <td>6393</td>
    </tr>
    <tr>
      <th>Eligible-roster patients</th>
      <td>1556</td>
    </tr>
    <tr>
      <th>Eligible HCPs</th>
      <td>158</td>
    </tr>
  </tbody>
</table>
</div>


The eligible roster covers 1,556 patients and 158 HCPs. The remaining journey patients stay outside this field-planning artifact.


## 1. Attribution sensitivity



```python
agreement = results["attribution_summary"].copy()
agreement["agreement_rate"] = agreement["agreement_rate"].map(
    lambda value: f"{value:.1%}"
)
display(agreement)
display(
    results["attribution_comparison"].query(
        "patient_id == 'PAT02034'"
    )
)

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
      <th>comparison</th>
      <th>patients_with_both</th>
      <th>same_hcp</th>
      <th>agreement_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Index vs plurality</td>
      <td>6393</td>
      <td>4399</td>
      <td>68.8%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Index vs latest</td>
      <td>6393</td>
      <td>4088</td>
      <td>63.9%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>All 3 rules</td>
      <td>6393</td>
      <td>4005</td>
      <td>62.6%</td>
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
      <th>patient_id</th>
      <th>index_npi</th>
      <th>plurality_npi</th>
      <th>latest_npi</th>
      <th>all_rules_agree</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>635</th>
      <td>PAT02034</td>
      <td>9000000280</td>
      <td>9000000280</td>
      <td>9000000280</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


All 3 attribution rules agree for 63.4% of patients. PAT02034 remains assigned to HCP0280 under every rule.


## 2. HCP evidence and concentration



```python
columns = [
    "npi", "account_id", "cohort_patients", "treated_patients",
    "roventra_starts", "competitor_treated", "untreated_mature",
    "review_opportunity", "contact_permission_status",
]
display(
    results["hcp_features"].sort_values(
        ["cohort_patients", "npi"], ascending=[False, True]
    )[columns].head(10)
)
display(results["decile_summary"].head())

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
      <th>account_id</th>
      <th>cohort_patients</th>
      <th>treated_patients</th>
      <th>roventra_starts</th>
      <th>competitor_treated</th>
      <th>untreated_mature</th>
      <th>review_opportunity</th>
      <th>contact_permission_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>93</th>
      <td>9000000430</td>
      <td>ACC189</td>
      <td>36</td>
      <td>9</td>
      <td>2</td>
      <td>7</td>
      <td>25</td>
      <td>32</td>
      <td>Allowed</td>
    </tr>
    <tr>
      <th>105</th>
      <td>9000000469</td>
      <td>ACC121</td>
      <td>34</td>
      <td>10</td>
      <td>9</td>
      <td>1</td>
      <td>21</td>
      <td>22</td>
      <td>Opt-out</td>
    </tr>
    <tr>
      <th>35</th>
      <td>9000000162</td>
      <td>ACC062</td>
      <td>33</td>
      <td>13</td>
      <td>8</td>
      <td>5</td>
      <td>19</td>
      <td>24</td>
      <td>Opt-out</td>
    </tr>
    <tr>
      <th>99</th>
      <td>9000000447</td>
      <td>ACC216</td>
      <td>32</td>
      <td>12</td>
      <td>5</td>
      <td>7</td>
      <td>17</td>
      <td>24</td>
      <td>Opt-out</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9000000026</td>
      <td>ACC226</td>
      <td>28</td>
      <td>11</td>
      <td>8</td>
      <td>3</td>
      <td>15</td>
      <td>18</td>
      <td>Allowed</td>
    </tr>
    <tr>
      <th>121</th>
      <td>9000000537</td>
      <td>ACC079</td>
      <td>28</td>
      <td>6</td>
      <td>2</td>
      <td>4</td>
      <td>18</td>
      <td>22</td>
      <td>Opt-out</td>
    </tr>
    <tr>
      <th>49</th>
      <td>9000000217</td>
      <td>ACC224</td>
      <td>27</td>
      <td>8</td>
      <td>4</td>
      <td>4</td>
      <td>16</td>
      <td>20</td>
      <td>Allowed</td>
    </tr>
    <tr>
      <th>117</th>
      <td>9000000516</td>
      <td>ACC167</td>
      <td>27</td>
      <td>9</td>
      <td>6</td>
      <td>3</td>
      <td>18</td>
      <td>21</td>
      <td>Allowed</td>
    </tr>
    <tr>
      <th>102</th>
      <td>9000000460</td>
      <td>ACC219</td>
      <td>26</td>
      <td>10</td>
      <td>7</td>
      <td>3</td>
      <td>15</td>
      <td>18</td>
      <td>Allowed</td>
    </tr>
    <tr>
      <th>86</th>
      <td>9000000389</td>
      <td>ACC155</td>
      <td>24</td>
      <td>8</td>
      <td>4</td>
      <td>4</td>
      <td>15</td>
      <td>19</td>
      <td>Allowed</td>
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
      <th>opportunity_decile</th>
      <th>hcps</th>
      <th>cohort_patients</th>
      <th>review_opportunity</th>
      <th>cumulative_hcp_share</th>
      <th>cumulative_opportunity_share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>12</td>
      <td>279</td>
      <td>216</td>
      <td>0.107143</td>
      <td>0.265683</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>11</td>
      <td>157</td>
      <td>123</td>
      <td>0.205357</td>
      <td>0.416974</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>11</td>
      <td>127</td>
      <td>100</td>
      <td>0.303571</td>
      <td>0.539975</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>11</td>
      <td>109</td>
      <td>84</td>
      <td>0.401786</td>
      <td>0.643296</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>11</td>
      <td>83</td>
      <td>68</td>
      <td>0.500000</td>
      <td>0.726937</td>
    </tr>
  </tbody>
</table>
</div>


The highest-volume rows include opt-outs. The first 30% of HCPs capture 55.2% of review opportunity.


![Top 20 HCPs ranked by review opportunity. Blue bars show review opportunity for Allowed HCPs, red bars for Opt-out HCPs, gray for Unknown. Light blue shows remaining attributed patients.](assets/figures/figure_6_1_volume_diagnostic.svg)

*Figure 6.1. Review opportunity ranked highest to lowest, colored by contact permission. An HCP near the top with a red bar holds substantial opportunity but cannot be worked through the field channel in this cycle. Synthetic data.*

![Line chart starting at the origin showing cumulative review opportunity captured as contactable HCP share increases, ranked by review opportunity, with a dashed diagonal reference line.](assets/figures/figure_6_2_cumulative_capture.svg)

*Figure 6.2. The curve starts at (0%, 0%) and rises steeply. The top 30% of contactable HCPs by review opportunity account for 54% of total contactable opportunity. The dashed diagonal shows what equal distribution would look like. Synthetic data.*


## 3. Referral pathways



```python
display(results["referral_edges"].head(10))
stable = results["referral_metrics"].merge(
    results["referral_stability"][[
        "npi", "bootstrap_top20_frequency", "window_rank_range",
    ]],
    on="npi",
)
display(stable.head(15))

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
      <th>source_npi</th>
      <th>destination_npi</th>
      <th>source_specialty</th>
      <th>destination_specialty</th>
      <th>source_account_id</th>
      <th>destination_account_id</th>
      <th>region</th>
      <th>unique_patients</th>
      <th>referral_episodes</th>
      <th>median_transition_days</th>
      <th>first_referral_date</th>
      <th>last_referral_date</th>
      <th>path_cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9000000578</td>
      <td>9000000258</td>
      <td>Primary Care</td>
      <td>Endocrinology</td>
      <td>ACC142</td>
      <td>ACC164</td>
      <td>Midwest</td>
      <td>22</td>
      <td>22</td>
      <td>25.0</td>
      <td>2024-01-09</td>
      <td>2024-10-29</td>
      <td>0.045455</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9000000417</td>
      <td>9000000164</td>
      <td>Primary Care</td>
      <td>Endocrinology</td>
      <td>ACC126</td>
      <td>ACC109</td>
      <td>West</td>
      <td>20</td>
      <td>20</td>
      <td>40.0</td>
      <td>2024-01-22</td>
      <td>2024-11-28</td>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9000000460</td>
      <td>9000000567</td>
      <td>Primary Care</td>
      <td>Endocrinology</td>
      <td>ACC219</td>
      <td>ACC030</td>
      <td>South</td>
      <td>20</td>
      <td>20</td>
      <td>24.5</td>
      <td>2024-01-29</td>
      <td>2024-12-18</td>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9000000033</td>
      <td>9000000302</td>
      <td>Primary Care</td>
      <td>Endocrinology</td>
      <td>ACC044</td>
      <td>ACC090</td>
      <td>South</td>
      <td>19</td>
      <td>19</td>
      <td>32.0</td>
      <td>2024-02-02</td>
      <td>2024-12-24</td>
      <td>0.052632</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9000000265</td>
      <td>9000000409</td>
      <td>Primary Care</td>
      <td>Endocrinology</td>
      <td>ACC148</td>
      <td>ACC164</td>
      <td>Midwest</td>
      <td>19</td>
      <td>19</td>
      <td>27.0</td>
      <td>2024-02-13</td>
      <td>2024-12-23</td>
      <td>0.052632</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9000000520</td>
      <td>9000000127</td>
      <td>Primary Care</td>
      <td>Endocrinology</td>
      <td>ACC110</td>
      <td>ACC073</td>
      <td>South</td>
      <td>19</td>
      <td>19</td>
      <td>29.0</td>
      <td>2024-01-16</td>
      <td>2024-12-26</td>
      <td>0.052632</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9000000020</td>
      <td>9000000409</td>
      <td>Primary Care</td>
      <td>Endocrinology</td>
      <td>ACC068</td>
      <td>ACC164</td>
      <td>Midwest</td>
      <td>18</td>
      <td>18</td>
      <td>37.0</td>
      <td>2024-02-25</td>
      <td>2024-11-28</td>
      <td>0.055556</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9000000128</td>
      <td>9000000567</td>
      <td>Primary Care</td>
      <td>Endocrinology</td>
      <td>ACC160</td>
      <td>ACC030</td>
      <td>South</td>
      <td>18</td>
      <td>18</td>
      <td>31.5</td>
      <td>2024-02-24</td>
      <td>2024-12-26</td>
      <td>0.055556</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9000000470</td>
      <td>9000000217</td>
      <td>Primary Care</td>
      <td>Endocrinology</td>
      <td>ACC068</td>
      <td>ACC224</td>
      <td>Midwest</td>
      <td>18</td>
      <td>18</td>
      <td>29.0</td>
      <td>2024-01-21</td>
      <td>2024-10-26</td>
      <td>0.055556</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9000000565</td>
      <td>9000000217</td>
      <td>Primary Care</td>
      <td>Endocrinology</td>
      <td>ACC099</td>
      <td>ACC224</td>
      <td>Midwest</td>
      <td>18</td>
      <td>18</td>
      <td>32.5</td>
      <td>2024-02-19</td>
      <td>2024-12-26</td>
      <td>0.055556</td>
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
      <th>specialty</th>
      <th>account_id</th>
      <th>region</th>
      <th>unique_sources</th>
      <th>unique_destinations</th>
      <th>patients_received</th>
      <th>patients_referred</th>
      <th>betweenness_centrality</th>
      <th>pathway_patient_volume</th>
      <th>pathway_breadth</th>
      <th>bootstrap_top20_frequency</th>
      <th>window_rank_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9000000217</td>
      <td>Endocrinology</td>
      <td>ACC224</td>
      <td>Midwest</td>
      <td>6</td>
      <td>2</td>
      <td>72</td>
      <td>15</td>
      <td>0.000626</td>
      <td>87</td>
      <td>8</td>
      <td>1.0000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9000000567</td>
      <td>Endocrinology</td>
      <td>ACC030</td>
      <td>South</td>
      <td>5</td>
      <td>2</td>
      <td>66</td>
      <td>14</td>
      <td>0.000521</td>
      <td>80</td>
      <td>7</td>
      <td>1.0000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9000000127</td>
      <td>Endocrinology</td>
      <td>ACC073</td>
      <td>South</td>
      <td>7</td>
      <td>2</td>
      <td>57</td>
      <td>13</td>
      <td>0.000730</td>
      <td>70</td>
      <td>9</td>
      <td>1.0000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9000000170</td>
      <td>Endocrinology</td>
      <td>ACC132</td>
      <td>Northeast</td>
      <td>8</td>
      <td>2</td>
      <td>56</td>
      <td>13</td>
      <td>0.000834</td>
      <td>69</td>
      <td>10</td>
      <td>0.9875</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9000000204</td>
      <td>Endocrinology</td>
      <td>ACC153</td>
      <td>South</td>
      <td>6</td>
      <td>2</td>
      <td>55</td>
      <td>9</td>
      <td>0.000521</td>
      <td>64</td>
      <td>8</td>
      <td>1.0000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9000000215</td>
      <td>Endocrinology</td>
      <td>ACC183</td>
      <td>South</td>
      <td>7</td>
      <td>1</td>
      <td>56</td>
      <td>8</td>
      <td>0.000313</td>
      <td>64</td>
      <td>8</td>
      <td>0.9875</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9000000207</td>
      <td>Endocrinology</td>
      <td>ACC094</td>
      <td>Northeast</td>
      <td>4</td>
      <td>2</td>
      <td>46</td>
      <td>16</td>
      <td>0.000417</td>
      <td>62</td>
      <td>6</td>
      <td>1.0000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9000000258</td>
      <td>Endocrinology</td>
      <td>ACC164</td>
      <td>Midwest</td>
      <td>4</td>
      <td>2</td>
      <td>49</td>
      <td>12</td>
      <td>0.000417</td>
      <td>61</td>
      <td>6</td>
      <td>1.0000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9000000550</td>
      <td>Endocrinology</td>
      <td>ACC179</td>
      <td>West</td>
      <td>5</td>
      <td>2</td>
      <td>50</td>
      <td>9</td>
      <td>0.000469</td>
      <td>59</td>
      <td>7</td>
      <td>1.0000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9000000636</td>
      <td>Endocrinology</td>
      <td>ACC059</td>
      <td>Northeast</td>
      <td>7</td>
      <td>1</td>
      <td>47</td>
      <td>11</td>
      <td>0.000313</td>
      <td>58</td>
      <td>8</td>
      <td>0.9875</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9000000115</td>
      <td>Endocrinology</td>
      <td>ACC225</td>
      <td>South</td>
      <td>6</td>
      <td>2</td>
      <td>44</td>
      <td>12</td>
      <td>0.000573</td>
      <td>56</td>
      <td>8</td>
      <td>0.9625</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11</th>
      <td>9000000409</td>
      <td>Endocrinology</td>
      <td>ACC164</td>
      <td>Midwest</td>
      <td>4</td>
      <td>1</td>
      <td>45</td>
      <td>6</td>
      <td>0.000209</td>
      <td>51</td>
      <td>5</td>
      <td>0.9750</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>9000000218</td>
      <td>Endocrinology</td>
      <td>ACC204</td>
      <td>Midwest</td>
      <td>5</td>
      <td>2</td>
      <td>38</td>
      <td>12</td>
      <td>0.000417</td>
      <td>50</td>
      <td>7</td>
      <td>0.9500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>9000000363</td>
      <td>Endocrinology</td>
      <td>ACC022</td>
      <td>Midwest</td>
      <td>5</td>
      <td>2</td>
      <td>39</td>
      <td>11</td>
      <td>0.000521</td>
      <td>50</td>
      <td>7</td>
      <td>0.8750</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>9000000174</td>
      <td>Endocrinology</td>
      <td>ACC032</td>
      <td>Midwest</td>
      <td>5</td>
      <td>2</td>
      <td>34</td>
      <td>12</td>
      <td>0.000469</td>
      <td>46</td>
      <td>7</td>
      <td>0.8125</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


The referral output is a pathway-context artifact. Stability comes from transition-window comparison and patient-level bootstrap resampling.


![Schematic referral graph illustrating directed edges, patient counts, and betweenness centrality.](assets/figures/figure_6_3_referral_schematic.svg)

*Figure 6.3. Conceptual illustration of the referral graph structure used in this chapter. Nodes A-C are Primary Care physicians (blue), node D is the Endocrinologist hub (gold), and nodes E-F are Cardiologists (green). Arrow width reflects patient count on each edge. Node D has the highest betweenness centrality because it bridges multiple upstream sources to downstream specialists.*

![Directed account-centered referral network with patient counts on each edge.](assets/figures/figure_6_4_referral_network.svg)

*Figure 6.4. The ego network shows the highest-betweenness HCP and the ten strongest referral edges connected to that physician. Patient count labels each edge. Synthetic data.*


## 4. Scientific role evidence



```python
candidates = results["kol_profiles"].loc[
    results["kol_profiles"]["kol_candidate"]
]
display(candidates[[
    "npi", "specialty_1", "research_percentile",
    "leadership_percentile", "practice_expertise_percentile",
    "peer_connection_percentile", "proposed_role",
    "role_fit_score", "evidence_confidence",
]].head(15))
display(results["kol_validation"])
display(results["kol_transparency_review"].head())

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
      <th>specialty_1</th>
      <th>research_percentile</th>
      <th>leadership_percentile</th>
      <th>practice_expertise_percentile</th>
      <th>peer_connection_percentile</th>
      <th>proposed_role</th>
      <th>role_fit_score</th>
      <th>evidence_confidence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9000000105</td>
      <td>Cardiology</td>
      <td>100.000000</td>
      <td>80.000000</td>
      <td>25.000000</td>
      <td>55.000000</td>
      <td>National scientific leader</td>
      <td>89.0</td>
      <td>High</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9000000206</td>
      <td>Endocrinology</td>
      <td>100.000000</td>
      <td>8.333333</td>
      <td>34.782609</td>
      <td>15.384615</td>
      <td>Evidence-generation collaborator</td>
      <td>77.2</td>
      <td>High</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9000000211</td>
      <td>Endocrinology</td>
      <td>100.000000</td>
      <td>36.363636</td>
      <td>80.000000</td>
      <td>70.833333</td>
      <td>Evidence-generation collaborator</td>
      <td>93.0</td>
      <td>High</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9000000237</td>
      <td>Primary Care</td>
      <td>100.000000</td>
      <td>75.000000</td>
      <td>77.777778</td>
      <td>91.666667</td>
      <td>Evidence-generation collaborator</td>
      <td>92.2</td>
      <td>High</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9000000363</td>
      <td>Endocrinology</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>66.666667</td>
      <td>Evidence-generation collaborator</td>
      <td>100.0</td>
      <td>High</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9000000441</td>
      <td>Primary Care</td>
      <td>100.000000</td>
      <td>84.615385</td>
      <td>77.777778</td>
      <td>26.190476</td>
      <td>Evidence-generation collaborator</td>
      <td>92.2</td>
      <td>High</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9000000512</td>
      <td>Cardiology</td>
      <td>100.000000</td>
      <td>31.250000</td>
      <td>94.444444</td>
      <td>92.500000</td>
      <td>Evidence-generation collaborator</td>
      <td>98.1</td>
      <td>High</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9000000562</td>
      <td>Cardiology</td>
      <td>100.000000</td>
      <td>40.000000</td>
      <td>89.473684</td>
      <td>75.000000</td>
      <td>Evidence-generation collaborator</td>
      <td>96.3</td>
      <td>High</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9000000633</td>
      <td>Primary Care</td>
      <td>100.000000</td>
      <td>92.592593</td>
      <td>4.545455</td>
      <td>59.375000</td>
      <td>National scientific leader</td>
      <td>95.9</td>
      <td>High</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9000000366</td>
      <td>Endocrinology</td>
      <td>96.153846</td>
      <td>4.166667</td>
      <td>39.130435</td>
      <td>15.384615</td>
      <td>Evidence-generation collaborator</td>
      <td>76.2</td>
      <td>High</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9000000277</td>
      <td>Cardiology</td>
      <td>94.736842</td>
      <td>25.000000</td>
      <td>63.157895</td>
      <td>40.909091</td>
      <td>Evidence-generation collaborator</td>
      <td>83.7</td>
      <td>High</td>
    </tr>
    <tr>
      <th>11</th>
      <td>9000000258</td>
      <td>Endocrinology</td>
      <td>92.307692</td>
      <td>41.666667</td>
      <td>30.434783</td>
      <td>96.153846</td>
      <td>Evidence-generation collaborator</td>
      <td>70.7</td>
      <td>High</td>
    </tr>
    <tr>
      <th>12</th>
      <td>9000000446</td>
      <td>Primary Care</td>
      <td>90.322581</td>
      <td>48.148148</td>
      <td>63.636364</td>
      <td>28.125000</td>
      <td>Evidence-generation collaborator</td>
      <td>81.0</td>
      <td>High</td>
    </tr>
    <tr>
      <th>13</th>
      <td>9000000235</td>
      <td>Cardiology</td>
      <td>89.473684</td>
      <td>70.000000</td>
      <td>15.789474</td>
      <td>40.909091</td>
      <td>National scientific leader</td>
      <td>78.8</td>
      <td>High</td>
    </tr>
    <tr>
      <th>14</th>
      <td>9000000008</td>
      <td>Cardiology</td>
      <td>88.888889</td>
      <td>25.000000</td>
      <td>61.111111</td>
      <td>65.000000</td>
      <td>Evidence-generation collaborator</td>
      <td>79.2</td>
      <td>High</td>
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
      <th>validation_measure</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KOL candidates</td>
      <td>83.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Reviewed candidates</td>
      <td>79.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Proposed role match rate</td>
      <td>0.582278</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Reviewer confirmation rate</td>
      <td>0.639241</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Reviewer decision kappa</td>
      <td>0.426150</td>
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
      <th>0</th>
      <td>9000000105</td>
      <td>National scientific leader</td>
      <td>Medical-affairs review required</td>
      <td>85575.71</td>
      <td>7</td>
      <td>Education/Training | Research Grants | Speakin...</td>
      <td>2024.0</td>
      <td>Transparency review only</td>
      <td>Disclosure context only; excluded from scienti...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9000000206</td>
      <td>Evidence-generation collaborator</td>
      <td>Medical-affairs review required</td>
      <td>0.00</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Disclosure context only; excluded from scienti...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9000000211</td>
      <td>Evidence-generation collaborator</td>
      <td>Medical-affairs review required</td>
      <td>0.00</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Disclosure context only; excluded from scienti...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9000000237</td>
      <td>Evidence-generation collaborator</td>
      <td>Medical-affairs review required</td>
      <td>0.00</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Disclosure context only; excluded from scienti...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9000000363</td>
      <td>Evidence-generation collaborator</td>
      <td>Medical-affairs review required</td>
      <td>0.00</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Disclosure context only; excluded from scienti...</td>
    </tr>
  </tbody>
</table>
</div>


Scientific role fit and payment transparency remain separate. Medical affairs owns the role review.


![Four scatter plots, one per proposed scientific role, showing candidate positions on that role's two primary evidence dimensions. Gray dots show candidates from other roles for context.](assets/figures/figure_6_5_kol_evidence_matrix.svg)

*Figure 6.5. Each panel focuses on one proposed role and plots the two dimensions that dominate its fit formula. Gray dots are candidates assigned to other roles. The same candidate can look strong or weak depending on which role lens is applied. Synthetic data.*


## 5. K-means engagement profiles



```python
display(results["cluster_evaluation"])
display(results["segment_profiles"])
display(results["segment_policy_comparison"])

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
      <th>inertia</th>
      <th>silhouette</th>
      <th>minimum_cluster_size</th>
      <th>minimum_cluster_share</th>
      <th>seed_stability_ari</th>
      <th>bootstrap_stability_ari</th>
      <th>selection_score</th>
      <th>operational_size_pass</th>
      <th>selected</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>13.929700</td>
      <td>0.641224</td>
      <td>14</td>
      <td>0.250000</td>
      <td>1.000000</td>
      <td>0.947400</td>
      <td>1.228074</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>3.515723</td>
      <td>0.763294</td>
      <td>9</td>
      <td>0.160714</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.363294</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>2.902478</td>
      <td>0.712550</td>
      <td>4</td>
      <td>0.071429</td>
      <td>1.000000</td>
      <td>0.918132</td>
      <td>1.239702</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>2.490435</td>
      <td>0.508611</td>
      <td>4</td>
      <td>0.071429</td>
      <td>0.925109</td>
      <td>0.810973</td>
      <td>0.990251</td>
      <td>False</td>
      <td>False</td>
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
      <th>cluster_id</th>
      <th>hcp_count</th>
      <th>cohort_patients</th>
      <th>opportunity_rate</th>
      <th>roventra_share</th>
      <th>access_signal_rate</th>
      <th>recent_contacts</th>
      <th>productive_contact_rate</th>
      <th>evidence_need_score</th>
      <th>access_resource_score</th>
      <th>digital_response_rate</th>
      <th>field_response_rate</th>
      <th>segment_name</th>
      <th>engagement_pattern</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>9</td>
      <td>14.111111</td>
      <td>0.670029</td>
      <td>0.596152</td>
      <td>0.005556</td>
      <td>1.000000</td>
      <td>0.394180</td>
      <td>0.800000</td>
      <td>0.589333</td>
      <td>0.770111</td>
      <td>0.236667</td>
      <td>C0: Digital evidence seeker</td>
      <td>Approved digital evidence, then field review</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>14</td>
      <td>14.571429</td>
      <td>0.755483</td>
      <td>0.495040</td>
      <td>0.001984</td>
      <td>1.857143</td>
      <td>0.461905</td>
      <td>0.309214</td>
      <td>0.347214</td>
      <td>0.210071</td>
      <td>0.813929</td>
      <td>C1: Field maintenance</td>
      <td>Maintenance field follow-up</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>22</td>
      <td>13.181818</td>
      <td>0.777571</td>
      <td>0.456602</td>
      <td>0.004132</td>
      <td>1.227273</td>
      <td>0.357035</td>
      <td>0.793909</td>
      <td>0.619273</td>
      <td>0.225773</td>
      <td>0.820773</td>
      <td>C2: Field evidence builder</td>
      <td>Field evidence discussion</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>11</td>
      <td>11.000000</td>
      <td>0.702138</td>
      <td>0.556061</td>
      <td>0.000000</td>
      <td>0.909091</td>
      <td>0.513420</td>
      <td>0.292818</td>
      <td>0.330455</td>
      <td>0.766909</td>
      <td>0.233909</td>
      <td>C3: Digital maintenance</td>
      <td>Digital maintenance, then field review</td>
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
      <th>segment_name</th>
      <th>Access-resource need</th>
      <th>Balanced follow-up</th>
      <th>Digital evidence seeker</th>
      <th>Established adopter</th>
      <th>Field evidence builder</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C0: Digital evidence seeker</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C1: Field maintenance</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C2: Field evidence builder</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C3: Digital maintenance</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


The selected 4-cluster solution has silhouette 0.763, seed ARI 1.000, bootstrap ARI 1.000, and minimum cluster size 9.


![Line chart comparing silhouette, seed ARI, and bootstrap ARI for candidate cluster counts.](assets/figures/figure_6_6_cluster_validation.svg)

*Figure 6.6. k=4 achieves the best silhouette score, seed ARI, and bootstrap ARI among solutions that pass the minimum cluster-size gate. Synthetic data.*

![2x2 small-multiples bar charts showing each engagement profile's evidence-need, access-need, digital-response, and field-response scores.](assets/figures/figure_6_7_segment_profiles.svg)

*Figure 6.7. Each panel is one engagement profile. The dashed line marks 0.5 (mid-range). C0 and C2 both show high evidence-need bars but diverge on which response channel is tall; C1 and C3 both show lower evidence-need bars but split the same way on channel. Synthetic data.*


## 6. HCP call plan



```python
display(results["call_plan"])
display(results["plan_comparison"])

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
      <th>cycle_start</th>
      <th>cycle_end</th>
      <th>territory</th>
      <th>account_id</th>
      <th>parent_account_id</th>
      <th>account_name</th>
      <th>npi</th>
      <th>specialty</th>
      <th>account_action</th>
      <th>hcp_action</th>
      <th>engagement_pattern</th>
      <th>segment_name</th>
      <th>recommended_calls</th>
      <th>hcp_review_opportunity</th>
      <th>recent_contacts</th>
      <th>permission_status</th>
      <th>reason_code</th>
      <th>reason</th>
      <th>territory_cycle_capacity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2025-01-01</td>
      <td>2025-01-28</td>
      <td>T01</td>
      <td>ACC224</td>
      <td>SYS-MID-09</td>
      <td>Michigan Care 224</td>
      <td>9000000217</td>
      <td>Endocrinology</td>
      <td>Increase priority</td>
      <td>Prioritize</td>
      <td>Maintenance field follow-up</td>
      <td>C1: Field maintenance</td>
      <td>2</td>
      <td>20</td>
      <td>1</td>
      <td>Allowed</td>
      <td>PRIORITIZE_REVIEW_OPPORTUNITY</td>
      <td>Permitted review opportunity and adoption belo...</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2025-01-01</td>
      <td>2025-01-28</td>
      <td>T01</td>
      <td>ACC056</td>
      <td>SYS-NOR-09</td>
      <td>Pennsylvania Care 056</td>
      <td>9000000136</td>
      <td>Primary Care</td>
      <td>Increase priority</td>
      <td>Prioritize</td>
      <td>Field evidence discussion</td>
      <td>C2: Field evidence builder</td>
      <td>2</td>
      <td>11</td>
      <td>1</td>
      <td>Allowed</td>
      <td>PRIORITIZE_REVIEW_OPPORTUNITY</td>
      <td>Permitted review opportunity and adoption belo...</td>
      <td>48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2025-01-01</td>
      <td>2025-01-28</td>
      <td>T03</td>
      <td>ACC034</td>
      <td>SYS-SOU-11</td>
      <td>Florida Care 034</td>
      <td>9000000273</td>
      <td>Primary Care</td>
      <td>Increase priority</td>
      <td>Prioritize</td>
      <td>Field review</td>
      <td>Not clustered</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>Allowed</td>
      <td>PRIORITIZE_REVIEW_OPPORTUNITY</td>
      <td>Permitted review opportunity and adoption belo...</td>
      <td>56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2025-01-01</td>
      <td>2025-01-28</td>
      <td>T04</td>
      <td>ACC155</td>
      <td>SYS-WES-12</td>
      <td>Arizona Care 155</td>
      <td>9000000389</td>
      <td>Cardiology</td>
      <td>Increase priority</td>
      <td>Prioritize</td>
      <td>Digital maintenance, then field review</td>
      <td>C3: Digital maintenance</td>
      <td>2</td>
      <td>19</td>
      <td>1</td>
      <td>Allowed</td>
      <td>PRIORITIZE_REVIEW_OPPORTUNITY</td>
      <td>Permitted review opportunity and adoption belo...</td>
      <td>48</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2025-01-01</td>
      <td>2025-01-28</td>
      <td>T04</td>
      <td>ACC219</td>
      <td>SYS-SOU-04</td>
      <td>Florida Care 219</td>
      <td>9000000460</td>
      <td>Primary Care</td>
      <td>Maintain</td>
      <td>Maintain</td>
      <td>Field evidence discussion</td>
      <td>C2: Field evidence builder</td>
      <td>1</td>
      <td>18</td>
      <td>0</td>
      <td>Allowed</td>
      <td>MAINTAIN_ESTABLISHED</td>
      <td>Permitted evidence with adoption at or above t...</td>
      <td>48</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2025-01-01</td>
      <td>2025-01-28</td>
      <td>T05</td>
      <td>ACC124</td>
      <td>SYS-NOR-05</td>
      <td>New York Care 124</td>
      <td>9000000035</td>
      <td>Primary Care</td>
      <td>Increase priority</td>
      <td>Prioritize</td>
      <td>Field evidence discussion</td>
      <td>C2: Field evidence builder</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>Allowed</td>
      <td>PRIORITIZE_REVIEW_OPPORTUNITY</td>
      <td>Permitted review opportunity and adoption belo...</td>
      <td>52</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2025-01-01</td>
      <td>2025-01-28</td>
      <td>T06</td>
      <td>ACC189</td>
      <td>SYS-MID-10</td>
      <td>Michigan Care 189</td>
      <td>9000000430</td>
      <td>Cardiology</td>
      <td>Increase priority</td>
      <td>Prioritize</td>
      <td>Maintenance field follow-up</td>
      <td>C1: Field maintenance</td>
      <td>2</td>
      <td>32</td>
      <td>0</td>
      <td>Allowed</td>
      <td>PRIORITIZE_REVIEW_OPPORTUNITY</td>
      <td>Permitted review opportunity and adoption belo...</td>
      <td>56</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2025-01-01</td>
      <td>2025-01-28</td>
      <td>T06</td>
      <td>ACC109</td>
      <td>SYS-WES-02</td>
      <td>Arizona Care 109</td>
      <td>9000000164</td>
      <td>Endocrinology</td>
      <td>Increase priority</td>
      <td>Prioritize</td>
      <td>Approved digital evidence, then field review</td>
      <td>C0: Digital evidence seeker</td>
      <td>2</td>
      <td>13</td>
      <td>0</td>
      <td>Allowed</td>
      <td>PRIORITIZE_REVIEW_OPPORTUNITY</td>
      <td>Permitted review opportunity and adoption belo...</td>
      <td>56</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2025-01-01</td>
      <td>2025-01-28</td>
      <td>T06</td>
      <td>ACC005</td>
      <td>SYS-SOU-06</td>
      <td>Florida Care 005</td>
      <td>9000000498</td>
      <td>Cardiology</td>
      <td>Increase priority</td>
      <td>Prioritize</td>
      <td>Field evidence discussion</td>
      <td>C2: Field evidence builder</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>Allowed</td>
      <td>PRIORITIZE_REVIEW_OPPORTUNITY</td>
      <td>Permitted review opportunity and adoption belo...</td>
      <td>56</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2025-01-01</td>
      <td>2025-01-28</td>
      <td>T06</td>
      <td>ACC005</td>
      <td>SYS-SOU-06</td>
      <td>Florida Care 005</td>
      <td>9000000051</td>
      <td>Cardiology</td>
      <td>Increase priority</td>
      <td>Prioritize</td>
      <td>Field evidence discussion</td>
      <td>C2: Field evidence builder</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>Allowed</td>
      <td>PRIORITIZE_REVIEW_OPPORTUNITY</td>
      <td>Permitted review opportunity and adoption belo...</td>
      <td>56</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2025-01-01</td>
      <td>2025-01-28</td>
      <td>T07</td>
      <td>ACC190</td>
      <td>SYS-WES-11</td>
      <td>Washington Care 190</td>
      <td>9000000366</td>
      <td>Endocrinology</td>
      <td>Increase priority</td>
      <td>Prioritize</td>
      <td>Field review</td>
      <td>Not clustered</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>Allowed</td>
      <td>PRIORITIZE_REVIEW_OPPORTUNITY</td>
      <td>Permitted review opportunity and adoption belo...</td>
      <td>48</td>
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
      <th>plan</th>
      <th>selected_hcps</th>
      <th>contact_permitted</th>
      <th>held_or_unknown</th>
      <th>review_opportunity</th>
      <th>recent_contacts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Top 30 by patient volume</td>
      <td>30</td>
      <td>30</td>
      <td>0</td>
      <td>397</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gated 4-week field plan</td>
      <td>11</td>
      <td>11</td>
      <td>0</td>
      <td>143</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


The HCP call plan contains 11 permitted HCPs and 20 recommended calls. Each row keeps site account context for routing.


## 8. Export the evidence package



```python
output_dir = ROOT / "ch06_hcp" / "assets" / "generated_outputs"
analysis_module.write_outputs(results, output_dir, ROOT)
print(f"Wrote {len(results)} CSV artifacts and manifest.json")

```

    Wrote 32 CSV artifacts and manifest.json


The package carries analysis date, source hashes, rule version, decision boundaries, and output contracts.

