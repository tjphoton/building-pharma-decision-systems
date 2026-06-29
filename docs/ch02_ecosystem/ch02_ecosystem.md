# Chapter 2: The Commercialization Operating System

On June 12, 2024, HCP0280 writes a Roventra prescription order for PAT02034 at ACC089. Treatment begins only after the office, PAY002, and the specialty pharmacy complete their parts of the path.

You will map the external checkpoints and the internal launch organization around them, then use PAT02034's path to classify an observed event, identify the external decision owner, route addressable friction to the right internal function.

## 2.1 Prescription to Treatment Path

The prescription-to-treatment path begins when an HCP selects Roventra and ends when the patient starts treatment. A missing document, coverage restriction, dispensing delay, affordability problem, or patient decision, any of these 5 external checkpoints, can stop progress.

![The same fictional patient moves through clinical choice, office workflow, coverage review, specialty-pharmacy fulfillment, and treatment.](figures/one-prescription-many-handoffs.png)

*Figure 2.1. PAT02034 moves through 5 checkpoints after HCP0280 selects Roventra. Progress depends on office workflow, PAY002 coverage, specialty-pharmacy fulfillment, and her decision to begin treatment. Fictional market.*

*Table 2.1. The external decision owners on the prescription-to-treatment path.*

| Checkpoint | External decision owner | What must happen | Common source of delay |
| --- | --- | --- | --- |
| Clinical choice | HCP and patient | The HCP and patient agree that Roventra fits the approved indication and treatment plan | Clinical fit, treatment preference, or selection of Nexoral or Vexpro |
| Office workflow | Account | Required records and forms reach the appropriate destination | Missing chart notes, insurance information, test results, or signatures |
| Coverage | Payer | The request satisfies the benefit and utilization-management rules | Formulary status, prior authorization, step therapy, quantity limit, or administrative exception |
| Fulfillment | Specialty pharmacy | The benefit transaction clears and the medicine is dispensed and delivered through the required channel | Benefit coordination, affordability, inventory, dispensing, or delivery |
| Treatment | Patient and treating HCP | The patient begins and continues the agreed treatment | Cost, treatment concerns, adverse effects, or a later clinical decision |

The broader launch path starts earlier. It includes diagnosis and disease awareness before treatment selection, then continues through documentation, coverage, fulfillment, initiation, and persistence. The 5-checkpoint path locates a specific prescription; the 6-stage path places the full launch organization from diagnosis through persistence.

## 2.2 The Launch Organization Around the Path

Internal functions support the path through approved information, process guidance, evidence, patient services, supply, and measurement. Table 2.2 maps the broader launch path to the functions that can address operational friction.

*Table 2.2. Launch functions mapped to patient-path stages, addressable friction, and observable outputs.*

| Path stage | External decision owner | Internal functions supporting the stage | Addressable friction | Observable output |
| --- | --- | --- | --- | --- |
| Diagnosis and disease awareness | HCP and patient | Disease-State Education, Medical Affairs, Market Research | Gaps in approved disease education or audience understanding | Education delivery, research findings, diagnosis records |
| Treatment selection | HCP and patient | Field Sales, Brand and Omnichannel Marketing, Medical Affairs | Gaps in approved product information or scientific exchange | Approved engagement and medication order |
| Prescription and documentation | Account | Field Reimbursement, Key Account Management, Patient Services | Missing forms, unclear process, or incomplete routing | Referral, submitted documentation, case status |
| Coverage | Payer | Market Access, Pricing and Contracting, Health Economics and Outcomes Research, Policy and Government Affairs | Value-evidence needs, contracting options, and process education | Effective-dated policy, authorization status, claim outcome |
| Fulfillment | Specialty pharmacy | Trade and Distribution, Patient Services, Supply Chain | Coordination, inventory, affordability, dispensing, or delivery delay | Paid transaction, dispense status, shipment |
| Initiation and persistence | Patient and treating HCP | Patient Services, Nurse Educators, Patient Marketing | Approved education, navigation, affordability support, or follow-up | Treatment start, refill, support status |

Decision Science and Analytics spans the 6 stages. It combines permitted records into evidence about where friction occurs, which accounts or populations warrant review, and whether an action changed a prespecified outcome. Operations, brand leadership, and cross-functional partners complete the operating map in [Appendix 2A](ch02_appendix.md).

![Selected launch functions support different points along the fictional patient path.](figures/commercial-teams-around-patient-path.png)

*Figure 2.2. Selected customer-facing and execution functions support specific stages of the path. Table 2.2 and Appendix 2A provide the fuller operating map, including analytics, leadership, and cross-functional partners. Fictional market.*

## 2.3 What Each Event Leaves Behind

The path creates partial records across separate operational systems. The same event may appear in more than one source.

*Table 2.3. Event-level records and their evidence boundaries.*

| Event | Possible record | What the record supports | Evidence boundary |
| --- | --- | --- | --- |
| Medication ordered | EHR medication order | Clinical intent was recorded on a date | The order may not have reached a dispensing pharmacy |
| Referral opened | Hub or specialty-pharmacy case | A case entered the support or fulfillment workflow | Referral does not establish authorization, payment, or shipment |
| Authorization reviewed | Hub, payer, or account status | A review status, when captured, and its status date | A commercial dataset may omit the payer's full reasoning and submitted clinical detail |
| Claim submitted | Pharmacy transaction | A benefit transaction occurred with a recorded outcome | `PAID`, `PENDED`, and `REVERSED` rows must be reconciled before counting completed fills |
| Medicine shipped | Specialty-pharmacy event | The pharmacy recorded fulfillment and shipment | Shipment does not prove that the patient took the medicine |
| Field engagement completed | CRM activity | A permitted interaction occurred with an HCP or account | HCP-level activity cannot usually be assigned to a named de-identified patient |
| Expected refill absent | Derived treatment episode | A refill was not observed within a prespecified window | The gap does not identify the reason for discontinuation |

PAT02034's path illustrates the separation. The EHR order is dated June 12, 2024. The specialty-pharmacy referral opens June 13. Authorization status is recorded as approved on June 18. A pharmacy claim is pended on July 2, paid on July 9, and followed by shipment on July 10. These dates describe distinct operational events in 2024.

No single source covers the full path or the full patient population. Coverage scope, arrival schedule, and identifier format differ across EHR, pharmacy, hub, CRM, and formulary files; joining them into a patient-path view requires explicit rules. The synthetic data package in the next chapter builds these records.

## 2.4 Summary

PAT02034's Roventra order moved through a clinical account, PAY002 benefit review, and specialty-pharmacy fulfillment before shipment. Each stage had a different external decision owner, internal support function, observable event, and evidence boundary.

1. Use the 5-checkpoint path to locate a prescription after treatment selection.
2. Use the broader 6-stage path to map the launch organization from diagnosis through persistence.
3. Route addressable friction to the function equipped to support it while preserving external clinical, coverage, dispensing, and patient decisions.
4. Keep medication orders, referrals, authorization statuses, claim transactions, shipments, CRM activity, and derived persistence records distinct.
5. Treat a paid claim and shipment as evidence that a case advanced. Use repeated cases and a credible comparison to measure whether an action caused the result.

## 2.5 Exercises

1. **Explain a metric change.** Paid new-patient starts increased 12% last month while CRM activity at the field team's targeted accounts remained flat. Give 2 checkpoint-level explanations. For each explanation, name the record needed to test it.

2. **Preserve the decision boundary.** An account submits complete documentation, and the payer denies coverage because the patient has not completed required step therapy. State what Field Reimbursement, Market Access, and the account may each do. End with the judgment an analyst should record before recommending action on real data.

## 2.6 Exercise Solutions

**1.** A formulary improvement may convert existing prescription attempts into paid starts without changing CRM activity. Test this with effective-dated formulary records and pharmacy transaction chains. A specialty-pharmacy backlog may also release cases referred in an earlier month. Test this with referral, approval, paid-claim, and shipment dates.

**2.** On the internal side: the account prepares complete clinical documentation; Field Reimbursement explains the current process and supports permitted routing or appeal steps; Market Access assesses the affected population, policy evidence, and contracting options. PAY002 retains the coverage decision; HCP0280 retains the treatment decision. Before recommending action, the analyst should verify the effective policy, patient eligibility, completed step history, documentation status, permitted support route, and number of similarly affected cases.
