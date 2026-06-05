// Agent factory + list

import { bullAgent } from './bull.js';
import { bearAgent } from './bear.js';
import { predictorAgent } from './predictor.js';
import { judgeAgent } from './judge.js';

export { bullAgent, bearAgent, predictorAgent, judgeAgent };

export function listAgents() {
  return [bullAgent, bearAgent, predictorAgent, judgeAgent];
}
