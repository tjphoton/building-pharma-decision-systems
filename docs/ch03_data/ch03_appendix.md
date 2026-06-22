# Appendix 3A: Common Data Sources and Synthetic Data Field Reference


*Table 3A.1. Common pharmaceutical data sources and analytical uses.*

| Data category | Analytical use |
| --- | --- |
| Medical claims | Disease prevalence, diagnosed populations, utilization, sites of care, comorbidities, provider volume, market sizing (Chapter 4); patient journeys (Chapter 5); referral network construction from `referring_npi` (Chapter 6) |
| Pharmacy claims | Treatment adoption, refills, persistence, switching, rejection analysis, prescriber activity, market share (Chapter 5) |
| Coverage and eligibility | Denominators, observation windows, lookback periods, treatment gaps, incidence cohorts (Chapters 4 and 5) |
| EHR and clinical data | Biomarkers, disease stage, cohort design, prior and concomitant therapy |
| Formulary and access | Coverage barriers, restriction burden, historical access assignment, reachable opportunity (Chapters 4 and 7) |
| Patient services / specialty pharmacy | Fulfillment pathway, authorization delays, abandonment analysis, support demand (Chapter 5) |
| CRM and field activity | Call patterns, reach, frequency, channel mix, targeting, next-best-action (Chapter 6) |
| HCP digital engagement | Content response, channel preference, engagement scoring, omnichannel sequencing (Chapter 6) |
| Provider and account reference | Provider matching, specialty segmentation, targeting, territory alignment, referral networks (Chapters 4 and 6) |
| HCP social network | Platform-level influence scoring, HCP-to-HCP engagement graph, social KOL identification (Chapter 6); vendors: Doceree DataIQ, CarePrecise, Swoop/Real Chemistry, OptimizeRx, PulsePoint |
| Publication and conference data | Co-authorship network, h-index proxy, congress prominence, scientific KOL scoring (Chapter 6); sources: OpenAlex, PubMed/NCBI, Alpha Sophia, IQVIA KOL Data |
| Product and code reference | Product mapping, treatment baskets, competitive groupings, quality checks (Chapters 4 and 5) |
| Public reference (CMS) | Provider volume benchmarking, market sizing, specialty segmentation, competitive context (Chapter 4); financial KOL signal via Open Payments (Chapter 6) |
| Sales and supply | Demand analysis, revenue tracking, forecast performance |

---

*Table 3A.2. Generated data file field reference.*

Each sub-table lists one group of output files: the folder, the grain of one record, and every field it contains.

**Table 3A.2.1: Reference tables** (`reference/`)

| File | Grain | Fields |
| --- | --- | --- |
| `patients.csv` | One patient | `patient_id`, `state`, `region`, `age_band`, `sex`, `true_launch_condition` |
| `patient_enrollments.csv` | One eligibility period per patient | `patient_id`, `eligibility_start_date`, `eligibility_end_date`, `payer_id`, `payer_type`, `has_medical_coverage`, `has_pharmacy_coverage`, `product_type` |
| `providers.csv` | One prescriber or facility | `npi`, `specialty_1`, `specialty_2`, `provider_state`, `provider_type`, `credential`, `primary_facility_npi` |
| `hcp_targets.csv` | One HCP in the internal commercial universe (~42% of all providers) | `npi`, `account_id`, `territory`, `state`, `region`, `specialty_1` |
| `accounts.csv` | One healthcare account (hospital, group practice, clinic) | `account_id`, `account_name`, `account_type`, `city`, `state`, `region`, `territory`, `capacity` |
| `payers.csv` | One payer or plan | `payer_id`, `payer_name`, `payer_type`, `region` |
| `ndc_codes.csv` | One product NDC code | `ndc`, `brand_generic`, `drug_name`, `ingredient` |

**Table 3A.2.2: Medical claims** (`claims_medical/`)

Two files with identical columns: `medical_claims.csv` (early snapshot, claims received within five days of month close) and `medical_claims_mature.csv` (full snapshot, all encounters regardless of lag). Use the mature file for most analyses; use the early file only to illustrate claim maturity.

| File | Grain | Fields |
| --- | --- | --- |
| `medical_claims[_mature].csv` | One billed encounter | `encounter_id`, `patient_id`, `claim_type`, `claim_date`, `admitting_diagnosis` (inpatient-only, ~1% populated), `diagnosis_1`–`diagnosis_10`, `icd_procedure_1`–`icd_procedure_3`, `patient_gender`, `patient_state`, `coverage_type`, `rendering_npi`, `attending_npi`, `referring_npi` (populated on specialist encounters when a primary care provider initiated the referral; ~49% of specialist claims; blank on all Primary Care visits and pharmacy claims), `facility_npi`, `payer_id` |
| `service_lines.csv` | One procedure line within an encounter | `encounter_id`, `patient_id`, `line_number`, `service_from`, `service_to`, `procedure_code`, `place_of_service`, `line_diagnosis_1`, `line_diagnosis_2`, `units`, `line_charge` |

**Table 3A.2.3: Pharmacy and lab claims** (`claims_pharmacy/`, `claims_lab/`)

| File | Grain | Fields |
| --- | --- | --- |
| `pharmacy_claims.csv` | One pharmacy transaction | `claim_id`, `patient_id`, `patient_state`, `prescriber_npi`, `primary_care_npi`, `date_of_service`, `rx_written_date`, `transaction_type`, `ndc`, `ndc_prescribed`, `refills_authorized`, `diagnosis_code`, `qty_prescribed`, `qty_dispensed`, `fill_number`, `days_supply`, `patient_pay`, `plan_pay`, `reject_code`, `payer_id` |
| `lab_results.csv` | One lab test result | `lab_id`, `patient_id`, `service_date`, `loinc_code`, `test_name`, `result`, `result_unit`, `ref_low`, `ref_high`, `abnormal_flag`, `ordering_npi`, `diagnosis_1` |

**Table 3A.2.4: Formulary** (`formulary/`)

| File | Grain | Fields |
| --- | --- | --- |
| `formulary_status.csv` | One plan-product current policy | `formulary_id`, `plan_id`, `plan_name`, `product_name`, `tier`, `prior_authorization`, `step_therapy`, `quantity_limit`, `specialty_pharmacy`, `effective_start`, `effective_end` |
| `formulary_history.csv` | One plan-product policy-change event | `history_id`, `plan_id`, `plan_name`, `product_name`, `quarter`, `effective_date`, `prior_tier`, `new_tier`, `prior_prior_authorization`, `new_prior_authorization`, `prior_step_therapy`, `new_step_therapy`, `prior_quantity_limit`, `new_quantity_limit`, `prior_specialty_pharmacy`, `new_specialty_pharmacy`, `change_type` |

**Table 3A.2.5: Specialty pharmacy** (`specialty_pharmacy/`)

| File | Grain | Fields |
| --- | --- | --- |
| `sp_events.csv` | One hub referral case | `event_id`, `patient_id`, `prescriber_npi`, `product_ndc`, `referral_date`, `hub_status`, `status_date`, `ship_date`, `days_supply`, `dispense_status`, `copay_assistance`, `discontinue_reason` |

**Table 3A.2.6: CRM and field activity** (`crm_veeva/`)

| File | Grain | Fields |
| --- | --- | --- |
| `crm_interactions.csv` | One recorded field interaction | `interaction_id`, `interaction_date`, `rep_id`, `hcp_npi`, `account_id`, `territory`, `product_name`, `channel`, `detail_topic`, `call_outcome`, `duration_min`, `sample_qty`, `consent_status` |
| `territory_alignment.csv` | One territory assignment | `territory`, `rep_id`, `region`, `n_accounts_aligned`, `n_hcps_aligned`, `target_calls_per_month`, `target_coverage_pct` |

**Table 3A.2.7: Digital engagement** (`digital_engagement/`)

| File | Grain | Fields |
| --- | --- | --- |
| `digital_engagement.csv` | One digital engagement event | `digital_event_id`, `event_date`, `hcp_npi`, `account_id`, `territory`, `product_name`, `channel`, `content_topic`, `open_flag`, `click_flag`, `webinar_attended` |

**Table 3A.2.8: Public-proxy datasets** (`open_payments/`, `cms_part_d/`)

These files mirror the structure of public CMS datasets but contain synthetic values. Do not combine them with the official public extracts.

| File | Grain | Fields |
| --- | --- | --- |
| `open_payments.csv` | One payment record to an HCP or institution | `npi`, `company_name`, `payment_year`, `payment_category`, `payment_amount`, `is_kol` |
| `prescriber_summary.csv` | One prescriber-drug annual summary (Part D style) | `prscrbr_npi`, `prscrbr_last_org_name`, `prscrbr_first_name`, `prscrbr_city`, `prscrbr_state_abrvtn`, `prscrbr_type`, `drug_name`, `generic_name`, `drug_class`, `tot_clms`, `tot_benes`, `tot_day_suply`, `tot_drug_cst`, `year` |

---

**Table 3A.2.9: KOL and network datasets** (`ch06_hcp/data/social/`, `ch06_hcp/data/publications/`)

These files are used in Chapter 6 for referral network analysis, social influence scoring, and co-authorship network construction. They are synthetic proxies modelled on vendor and public data source schemas described below. They are not part of the core ch03 data generation; run `ch06_hcp/scripts/generate_kol_network_data.py` to produce them.

**Referral network** is derived from `medical_claims_mature.csv` directly, with no separate file. Filter rows where `referring_npi` is non-blank; each row is a directed edge from `referring_npi` (source, typically Primary Care) to `rendering_npi` (destination, typically specialist).

**Social network** (`ch06_hcp/data/social/`): modelled on data formats from Doceree DataIQ, CarePrecise, Swoop/Real Chemistry Healthcare Social Graph, OptimizeRx, and PulsePoint Authenticated NPI identity graph. In production, NPI-to-social-handle mapping is obtained from CarePrecise (minimum order ~$250) or Doceree DataIQ (HIPAA-certified, multi-platform).

| File | Grain | Fields |
| --- | --- | --- |
| `social_profiles.csv` | One HCP × platform | `npi`, `platform`, `handle`, `display_name`, `verified`, `follower_count`, `following_count`, `bio_specialty_match`, `account_age_years`, `post_count_90d`, `avg_likes_per_post`, `avg_shares_per_post`, `engagement_rate`, `data_source`, `extract_date` |
| `social_posts.csv` | One social post | `post_id`, `npi`, `platform`, `post_date`, `content_type`, `topic_tags`, `likes`, `shares`, `comments`, `impressions`, `mentions_npi` |
| `social_interactions.csv` | One HCP-to-HCP engagement | `interaction_id`, `from_npi`, `to_npi`, `platform`, `interaction_date`, `interaction_type`, `content_topic` |

**Publication and conference data** (`ch06_hcp/data/publications/`): modelled on OpenAlex (free REST API and bulk S3 snapshots at openalex.org) and PubMed/NCBI E-utilities (free API). In production, NPI-to-author matching follows the HHS DDOD Use Case 24 strategy: author name search on PubMed or OpenAlex, then institution cross-walk. Commercial KOL databases such as Alpha Sophia and IQVIA KOL Data provide pre-built NPI-to-publication mappings. Congress speaker lists (ASCO, ACC, ADA, ACR, ATS, ENDO, ASH, CHEST) are publicly available on congress websites and can be licensed from IQVIA or scraped and name-matched to NPPES.

| File | Grain | Fields |
| --- | --- | --- |
| `publications.csv` | One publication or congress abstract | `pub_id`, `title`, `journal`, `pub_year`, `pub_type`, `doi_synthetic`, `citation_count`, `therapeutic_area`, `conference_flag`, `data_source` |
| `pub_authors.csv` | One author on one publication | `pub_id`, `npi`, `author_position`, `institution`, `is_corresponding` |
| `conference_appearances.csv` | One HCP role at one congress | `appearance_id`, `npi`, `conference_name`, `conf_year`, `role`, `presentation_title`, `session_topic`, `audience_size_band`, `data_source` |
