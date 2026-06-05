# P0a Verification #3 #4: Reuse Judgment + SW Death UI Safety

> Branch: `p0a-verify-3-4` | Date: 2026-05-29 | Based on p0a-fix-fingerprint

---

## #3: Failed Agent Not Reused

### Problem

Does the persistent error written by `mergeCheckpointError` cause the checkpoint resume to skip that Agent? (It should re-call LLM, not skip.)

### Evidence

`lib/agents/runner.js:138`:

```js
const hasPartial = (role) => checkpoint && checkpoint.partials && checkpoint.partials[role];
```

Only checks `checkpoint.partials[role]`, not `errors`.

`lib/agents/runner.js:101-113` (`mergeCheckpointError`):

```js
await chrome.storage.local.set({
  [key]: {
    v: CHECKPOINT_VERSION,
    ts: Date.now(),
    fp: fingerprint,
    partials: { ...prev.partials },          // <- does not write role
    errors: { ...prev.errors, [role]: errorMsg },  // <- only writes errors
  },
});
```

Error written to `errors[role]`, not `partials[role]`.

`lib/agents/runner.js:150-153`:

```js
const promises = agentDefs.map(({ agent, role }) => {
  if (hasPartial(role)) {
    return Promise.resolve(checkpoint.partials[role]);  // only reuse if partials exist
  }
  return agent.run(ctx, opts).then(/* ... */);
});
```

### Conclusion

**Pass.** Failed Agent is not reused — `hasPartial(role)` returns falsy for error-only role; Agent will re-call `agent.run()`.

### New Test

`test/agents/runner.test.js`:
- Preset checkpoint: `partials: { bull }` + `errors: { bear: 'LLM timeout' }`
- Assert: bull reused (text='bull checkpoint cached'), bear re-called (fetch = 3 times, includes bear+predictor+judge), bear.text != 'bear checkpoint cached'

---

## #4: SW Termination -> UI Retryable Failure State (not infinite spinner)

### Problem

After SW is terminated by Chrome mid-debate, does the content.js message channel always land in a "show error, user can retry" terminal state, or could it get stuck in infinite loading spinner?

### Evidence

`content.js:383-422` (`sendAndShow` function):

```js
const resp = await chrome.runtime.sendMessage({
  type: 'ANALYZE', url: location.href, force, pageEvents,
});
if (!resp) {
  setBody('<div class="error">Service Worker not responding; extension may not be loaded</div>');
  return;
}
if (!resp.ok) {
  setBody(`<div class="error">${escapeHtml(resp.error || 'Unknown error')}</div>`);
  return;
}
// ... normal rendering ...
} catch (err) {
  setBody(`<div class="error">Communication error: ${escapeHtml(err.message || String(err))}</div>`);
} finally {
  hideThinkingStream();
  busy = false;
  fabEl.disabled = false;
  reanalyzeEl.disabled = false;
}
```

Three error paths:

| Path | Trigger Condition | Result |
|------|---------|------|
| `!resp` | SW not responding (terminated/not loaded) | Shows "Service Worker not responding" |
| `!resp.ok` | SW returns error | Shows resp.error text |
| `catch` | Communication exception (channel broken etc.) | Shows "Communication error: ..." |

`finally` block executes on all three paths: hides loading animation, resets busy flag, restores
button enabled state. User sees error message and can close panel to retry.

No "message sent then SW dies, content side waits forever" path —
`chrome.runtime.sendMessage` returns `undefined` when SW does not respond (triggers
`!resp` branch), does not throw exception or hang.

### Conclusion

**Pass.** SW death -> `!resp` -> shows error text -> `finally` cleans up state. No
infinite spinner.

---

## Untouched

| Module | Status |
|------|------|
| Agent prompt (bull/bear/predictor/judge) | Untouched |
| LLM provider | Untouched |
| score-fusion | Untouched |
| Structured output parsing | Untouched |
| content.js error handling | Read-only verification, not modified |
| runDebate return structure | Unchanged |

## Tests

```
252 tests | 0 fail | 0 skip
```

Added 1 #3 verification test (pred error -> not reused), total 7 checkpoint-specific tests.

## Manual Review Checklist

- [ ] During trading hours: Stop SW (`chrome://serviceworker-internals` -> Stop)
- [ ] Click retry analysis button
- [ ] Confirm UI shows "Service Worker not responding" (not infinite spinner)
- [ ] Stop again -> retry -> console confirm completed Agent was reused
- [ ] `node --test test/agents/runner.test.js` -> 11/11 pass
