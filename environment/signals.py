"""Observation channel spec — the fly's sensory world, 16 channels.

Order matters: encoder/populations.py maps channel index -> real
sensory neuron population.

  0 funding      money signals (may be inflated by liar accounts)
  1 hiring       headcount growth / open roles
  2 intent       website/product intent
  3 job_change   champion/exec movement
  4 negative     bad news, layoffs, bad reviews
  5 trigger      trigger events (news, tech change)
  6 budget_frac  remaining spend today [0,1]
  7 day_frac     position in episode [0,1]
  8 researched   RESEARCH has been bought for this account
  9 enriched     ENRICH has been bought for this account
 10 research_price  today's RESEARCH cost / base, clipped [0,2]
 11 enrich_price    ..
 12 email_price     ..
 13 escalate_price  ..
 14 budget_level    today's daily budget / base, clipped [0,2]
 15 pain         latent pain revealed by RESEARCH (0.5 = unknown)
"""

N_CHANNELS = 16

CHANNEL_NAMES = [
    "funding", "hiring", "intent", "job_change", "negative", "trigger",
    "budget_frac", "day_frac", "researched", "enriched",
    "research_price", "enrich_price", "email_price", "escalate_price",
    "budget_level", "pain",
]
