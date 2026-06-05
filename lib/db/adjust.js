// Adjustment factor management (Phase 8 v1 stub)
// Future versions will implement adjustment factor calculation and forward adjustment
// Currently: adjust_events table created but not yet populated

/** Stub: to be implemented in future versions */
export async function fetchAdjustEvents(code) {
  // TODO: Fetch dividend/bonus events from eastmoney API
  return [];
}

/** Stub: to be implemented in future versions */
export function applyForwardAdjust(klines, code, refDate) {
  // TODO: Calculate forward-adjusted prices from adjust_events table
  // First version not implemented, returns raw klines directly
  return klines;
}
