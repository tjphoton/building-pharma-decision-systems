const phase = document.body.dataset.phase;
let activeRunId = null;
let activeCaseId = null;
let activeMode = 'saved';
let editVisible = false;

function detailMessage(payload, fallback) {
  if (!payload) return fallback;
  if (typeof payload.detail === 'string') return payload.detail;
  if (payload.detail && typeof payload.detail.message === 'string') return payload.detail.message;
  if (Array.isArray(payload.detail)) return payload.detail.map((item) => item.msg).join('; ');
  return fallback;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch (_error) {
    payload = null;
  }
  if (!response.ok) throw new Error(detailMessage(payload, `Request failed (${response.status}).`));
  return payload;
}

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab, .view').forEach((item) => item.classList.remove('active'));
    tab.classList.add('active');
    document.querySelector(`#view-${tab.dataset.view}`).classList.add('active');
  });
});

const runStatus = document.querySelector('#run-status');
const runMeter = document.querySelector('#run-meter');
const approvalStatus = document.querySelector('#approval-status');
const outcomeStatus = document.querySelector('#outcome-status');

function setText(selector, value) {
  const element = document.querySelector(selector);
  if (element) element.textContent = value;
}

function setMeter(state) {
  if (!state || !state.run_metadata) return;
  const metadata = state.run_metadata;
  const elapsed = Number(metadata.elapsed_ms || 0) / 1000;
  setText('[data-meter="node"]', metadata.current_node || '-');
  setText('[data-meter="elapsed"]', `${elapsed.toFixed(1)} s`);
  setText('[data-meter="llm"]', metadata.llm_steps || 0);
  setText('[data-meter="tokens"]', `${metadata.input_tokens || 0} / ${metadata.output_tokens || 0}`);
  setText('[data-meter="tools"]', metadata.tool_calls || 0);
  setText('[data-meter="queries"]', metadata.ad_hoc_queries || 0);
  setText('[data-meter="cost"]', `$${Number(metadata.estimated_cost_usd || 0).toFixed(4)}`);
  setText('[data-meter="pause"]', metadata.pause_reason || '-');
  setText('#overview-run-state', state.status || 'running');
  setText('#overview-run-id', metadata.run_id || 'registered');
  setText('#overview-mode', activeMode);
  setText('#overview-node', metadata.current_node || 'starting');
  setText('#overview-revisions', `${state.revision_count || 0} revisions`);
  if (state.human) setText('#overview-human', state.human.decision);
}

async function refreshTimeline() {
  if (!activeCaseId) return;
  const payload = await api(`/api/cases/${activeCaseId}`);
  const list = document.querySelector('#case-timeline');
  list.replaceChildren();
  payload.runs.forEach((run) => {
    const item = document.createElement('li');
    const title = document.createElement('strong');
    const meta = document.createElement('small');
    title.textContent = `${run.mode} run: ${run.status}`;
    meta.textContent = `${run.run_id} | ${run.current_node || 'registered'}`;
    item.append(title, meta);
    list.append(item);
  });
  if (payload.case.outcome_event) {
    const item = document.createElement('li');
    const title = document.createElement('strong');
    const meta = document.createElement('small');
    title.textContent = `Outcome: ${payload.case.outcome_event.observed_incremental_nrx} incremental NRx`;
    meta.textContent = payload.case.outcome_event.measurement_window;
    item.append(title, meta);
    list.append(item);
    document.querySelector('#reopen-case').disabled = false;
  }
}

async function pollRun(runId) {
  for (let attempt = 0; attempt < 400; attempt += 1) {
    const state = await api(`/api/runs/${runId}`);
    setMeter(state);
    if (state.status === 'interrupted') {
      runStatus.textContent = `Run interrupted: ${state.run_metadata.pause_reason || 'runtime limit'}.`;
      await refreshTimeline();
      return state;
    }
    if (state.human || state.review) {
      const disposition = state.review ? state.review.disposition : 'pending';
      runStatus.textContent = disposition === 'pass'
        ? 'Paused for human approval. Record a reviewer and reason below.'
        : `Independent review: ${disposition}. Human disposition is required.`;
      document.querySelector('#resume-run').hidden = false;
      await refreshTimeline();
      return state;
    }
    runStatus.textContent = `Running: ${state.run_metadata.current_node || 'starting'}.`;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  runStatus.textContent = 'The run is still active. Use the checkpoint control to poll again.';
  return null;
}

document.querySelector('#intake-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const submit = document.querySelector('#start-run');
  submit.disabled = true;
  activeMode = document.querySelector('#run-mode').value;
  setText('#run-mode-label', activeMode);
  setText('#overview-mode', activeMode);
  try {
    if (activeMode === 'saved') {
      const saved = await api(`/api/saved/${phase}`);
      setMeter(saved);
      runStatus.textContent = 'Loaded the committed saved trace. Choose mock or live to create a new case.';
      return;
    }
    const evidenceDate = document.querySelector('#evidence-date').value;
    runStatus.textContent = 'Evaluating the signal and opening the case.';
    const signal = await api('/api/signals/evaluate', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({evidence_date: evidenceDate}),
    });
    if (!signal.signal) throw new Error('The selected date does not produce a candidate signal.');
    const caseData = await api('/api/cases', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({evidence_date: evidenceDate}),
    });
    activeCaseId = caseData.case_id;
    runStatus.textContent = `Starting ${activeMode} run for ${activeCaseId}.`;
    const run = await api(`/api/cases/${activeCaseId}/runs`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode: activeMode}),
    });
    activeRunId = run.run_id;
    await refreshTimeline();
    await pollRun(activeRunId);
  } catch (error) {
    runStatus.textContent = `Error: ${error.message}`;
  } finally {
    submit.disabled = false;
  }
});

document.querySelector('#resume-run').addEventListener('click', async () => {
  if (!activeRunId) return;
  try {
    runStatus.textContent = 'Resuming from the durable checkpoint.';
    const result = await api(`/api/runs/${activeRunId}/resume`, {method: 'POST'});
    runStatus.textContent = `Checkpoint status: ${result.status}.`;
    await pollRun(activeRunId);
  } catch (error) {
    runStatus.textContent = `Error: ${error.message}`;
  }
});

function editedOption() {
  const isExperiment = document.querySelector('#edit-experiment').checked;
  return {
    name: document.querySelector('#edit-name').value,
    description: 'Reviewer-entered bounded option from the workbench.',
    budget_moved_usd: Number(document.querySelector('#edit-budget').value),
    audience: document.querySelector('#edit-audience').value,
    geography: document.querySelector('#edit-geography').value,
    duration_weeks: Number(document.querySelector('#edit-duration').value),
    reversibility: isExperiment ? 'high' : 'staged',
    is_experiment: isExperiment,
    measurement_design: isExperiment ? 'matched_market' : 'outcome_monitor',
  };
}

document.querySelectorAll('[data-decision]').forEach((button) => {
  button.addEventListener('click', async () => {
    const decision = button.dataset.decision;
    if (decision === 'edit' && !editVisible) {
      editVisible = true;
      document.querySelector('#edit-option').hidden = false;
      approvalStatus.textContent = 'Edit the structured option, then press Edit again to revalidate it.';
      return;
    }
    if (!activeRunId) {
      approvalStatus.textContent = 'Start a mock or live run before recording a disposition.';
      return;
    }
    const reviewer = document.querySelector('#reviewer').value.trim();
    const reason = document.querySelector('#review-reason').value.trim();
    if (!reviewer || !reason) {
      approvalStatus.textContent = 'Reviewer and reason are required.';
      return;
    }
    const body = {decision, reviewer, reason};
    if (decision === 'edit') body.edited_option = editedOption();
    try {
      const result = await api(`/api/runs/${activeRunId}/disposition`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      approvalStatus.textContent = `${decision} recorded by ${reviewer}. Run status: ${result.status}.`;
      setText('#overview-human', decision);
      document.querySelector('#reopen-case').disabled = decision !== 'approve';
      const state = await api(`/api/runs/${activeRunId}`);
      setMeter(state);
      await refreshTimeline();
    } catch (error) {
      approvalStatus.textContent = `Error: ${error.message}`;
    }
  });
});

document.querySelector('#outcome-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!activeCaseId) {
    outcomeStatus.textContent = 'Create and approve a case first.';
    return;
  }
  const form = new FormData(event.currentTarget);
  try {
    const result = await api(`/api/cases/${activeCaseId}/outcomes`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        observed_incremental_nrx: Number(form.get('observed')),
        confidence_low: Number(form.get('low')),
        confidence_high: Number(form.get('high')),
      }),
    });
    outcomeStatus.textContent = result.message;
    document.querySelector('#reopen-case').disabled = false;
    await refreshTimeline();
  } catch (error) {
    outcomeStatus.textContent = `Error: ${error.message}`;
  }
});

document.querySelector('#reopen-case').addEventListener('click', async () => {
  if (!activeCaseId) return;
  const mode = activeMode === 'live' ? 'live' : 'mock';
  try {
    outcomeStatus.textContent = 'Reopening the same case with the mature outcome.';
    const run = await api(`/api/cases/${activeCaseId}/reopen`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode}),
    });
    activeRunId = run.run_id;
    await refreshTimeline();
    await pollRun(activeRunId);
  } catch (error) {
    outcomeStatus.textContent = `Error: ${error.message}`;
  }
});

document.querySelector('#scenario-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const isExperiment = form.has('is_experiment');
  const target = document.querySelector('#scenario-result');
  target.classList.remove('error');
  target.innerHTML = '<p class="section-label">Scenario result</p><h3>Calculating</h3>';
  try {
    const result = await api('/api/scenarios', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        name: 'Reader what-if', description: 'Structured scenario from the local workbench.',
        budget_moved_usd: Number(form.get('budget_moved_usd')),
        audience: form.get('audience'), geography: form.get('geography'),
        duration_weeks: Number(form.get('duration_weeks')),
        reversibility: isExperiment ? 'high' : 'staged', is_experiment: isExperiment,
        measurement_design: isExperiment ? 'matched_market' : 'outcome_monitor',
        date_phase: phase,
      }),
    });
    target.replaceChildren();
    const label = document.createElement('p');
    const heading = document.createElement('h3');
    const summary = document.createElement('p');
    label.className = 'section-label';
    label.textContent = 'Scenario result';
    heading.textContent = result.feasible ? 'Inside approved controls' : 'Blocked by controls';
    summary.textContent = `Expected incremental NRx: ${result.expected_incr_nrx_low} to ${result.expected_incr_nrx_high}. Audience: ${result.audience_hcp_count.toLocaleString()} HCPs.`;
    target.append(label, heading, summary);
  } catch (error) {
    target.classList.add('error');
    target.textContent = `Scenario error: ${error.message}`;
  }
});
