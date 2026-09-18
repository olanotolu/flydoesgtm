import numpy as np

# ------------------------------------------------------------------
# Clayfly GTM ecosystem — ported from C. Claygans' world.py.
# The fly sees noisy signals; hidden state decides outcomes. Actions
# cost real budget; outcomes are probabilistic and delayed; accounts
# fatigue. One new adversarial feature: `inflate` — a fraction of
# low-intent accounts claim funding they don't have.
# ------------------------------------------------------------------

ACTIONS = ["WAIT", "OBSERVE", "RESEARCH", "ENRICH",
           "EMAIL", "ESCALATE", "IGNORE"]
WAIT, OBSERVE, RESEARCH, ENRICH, EMAIL, ESCALATE, IGNORE = range(7)

# hard daily-budget costs ($)
ACTION_COST = np.array([0, 0.05, 2, 1, 5, 15, 0], dtype=np.float32)

# immediate reward shaping (points)
ACTION_PENALTY = np.array(
    [0, -0.01, -0.25, -0.10, -1.0, -5.0, 0], dtype=np.float32)

OUTCOME_REWARD = {
    "reply": 2.0, "meeting": 8.0, "opportunity": 25.0, "closed": 100.0,
    "neg_reply": -2.0, "unsubscribe": -8.0, "spam": -30.0,
    "expired": -3.0,
}

REWARD_PROFILES = {
    "balanced": {},
    "gentle_spam": {"spam": -15.0, "unsubscribe": -5.0},
    "high_value": {"meeting": 15.0, "opportunity": 40.0,
                    "closed": 200.0},
}

HIDDEN_DIM = 5  # intent, icp_fit, urgency, champion, competitor_dissat


class World:
    BASE_COSTS = ACTION_COST.copy()
    BASE_BUDGET = 100.0

    def __init__(self, n_accounts=300, days=60, daily_budget=100.0,
                 seed=0, lambda_cost=0.0, conserve_coef=0.0,
                 conv_mult=1.0, cost_mult=1.0, intent_shift=0.0,
                 vary_costs=False, liar_frac=0.10,
                 reward_profile="balanced", reward_overrides=None,
                 rent=0.0, kill_bonus=0.0, kill_penalty=0.0,
                 expire_all=False):
        self.rng = np.random.default_rng(seed)
        self.n = n_accounts
        self.days = days
        self.daily_budget = daily_budget
        self.lambda_cost = lambda_cost
        self.conserve_coef = conserve_coef
        self.rent = rent
        self.kill_bonus = kill_bonus
        self.kill_penalty = kill_penalty
        self.expire_all = expire_all
        if reward_profile not in REWARD_PROFILES:
            raise ValueError(f"unknown reward profile: {reward_profile}")
        self.outcome_reward = dict(OUTCOME_REWARD)
        self.outcome_reward.update(REWARD_PROFILES[reward_profile])
        self.outcome_reward.update(reward_overrides or {})

        # per-day price/regime schedule (see c-claygans world.py)
        self.cost_sched = np.tile(
            self.BASE_COSTS * np.asarray(cost_mult, dtype=np.float32),
            (days, 1))
        self.conv_sched = np.full(days, conv_mult, dtype=np.float32)
        self.budget_sched = np.full(days, daily_budget, dtype=np.float32)
        if vary_costs:
            self._sample_regimes()

        self.costs = self.cost_sched[0]
        self.conv_mult = float(self.conv_sched[0])

        n, rng = self.n, self.rng
        self.intent = np.clip(rng.random(n) + intent_shift, 0, 1)
        self.icp = rng.random(n)
        self.urgency = rng.random(n)
        self.champion = rng.random(n)
        self.competitor = rng.random(n)
        # latent pain: only RESEARCH can reveal it (obs ch15). It
        # materially raises outreach EV when discovered — that's what
        # gives expensive cognition option value over cheap watching.
        self.pain = np.clip(
            0.5 * self.competitor + 0.3 * self.urgency
            + 0.2 * rng.random(n), 0, 1).astype(np.float32)

        # claimed-funding inflation: a slice of weak accounts lie about
        # money. Visible on the funding channel, uncorrelated with intent
        # — the thing the fly has to learn to distrust.
        self.inflate = np.where(
            (rng.random(n) < liar_frac) & (self.intent < 0.5),
            rng.uniform(0.3, 0.6, n), 0.0).astype(np.float32)

        # buying window: many companies only become purchasable mid-episode
        starts = rng.integers(0, max(2, days - 15), n)
        lengths = rng.integers(10, max(11, days - starts.max() // 2), n)
        self.win_start = starts
        self.win_end = np.minimum(starts + lengths, days - 1)

        # dynamic state
        self.fatigue = np.zeros(n, dtype=np.float32)
        self.researched = np.zeros(n, dtype=bool)
        self.enriched = np.zeros(n, dtype=bool)
        self.pursued = np.zeros(n, dtype=bool)
        self.last_pursue = np.full(n, -10_000, dtype=np.int64)
        self.active = np.ones(n, dtype=bool)
        self.sigma = np.full(n, 0.45, dtype=np.float32)
        self.last_slot = -np.ones(n, dtype=np.int64)

        # credit ledger: decision slot -> reward (immediate + delayed +
        # cost-pressure + conservation dividend). `econ` is the parallel
        # pure-economics ledger (outcomes + action penalties only) used
        # for benchmark reporting so training shaping stays out of it.
        self.credit = {}
        self.econ = {}
        self._credit_events = []
        self._econ_events = []
        self.pending = []          # (resolve_day, slot, reward)
        self.day_slots = []        # slots used today (dividend split)
        self.spend_today = 0.0
        self.stats = {
            "meetings": 0, "replies": 0, "opportunities": 0,
            "closed": 0, "unsubscribes": 0, "spam": 0,
            "emails": 0, "escalations": 0, "researches": 0,
            "enriches": 0, "spent": 0.0,
            "ignored": 0, "rent_paid": 0.0,
        }

    def _add_ledger(self, slot, credit_amount, econ_amount=None):
        """Add one auditable event to the shaped and pure-economics ledgers."""
        slot = int(slot)
        credit_amount = float(credit_amount)
        econ_amount = credit_amount if econ_amount is None else float(econ_amount)
        self.credit[slot] = self.credit.get(slot, 0.0) + credit_amount
        self.econ[slot] = self.econ.get(slot, 0.0) + econ_amount
        self._credit_events.append((slot, credit_amount))
        self._econ_events.append((slot, econ_amount))

    def ledger_invariants(self, *, atol=1e-5):
        """Return conservation checks for every settled ledger event."""
        credit_expected = sum(amount for _, amount in self._credit_events)
        econ_expected = sum(amount for _, amount in self._econ_events)
        return {
            "credit_conserved": bool(np.isclose(
                sum(self.credit.values()), credit_expected, atol=atol)),
            "econ_conserved": bool(np.isclose(
                sum(self.econ.values()), econ_expected, atol=atol)),
            "credit_total": float(sum(self.credit.values())),
            "econ_total": float(sum(self.econ.values())),
            "credit_events": len(self._credit_events),
            "econ_events": len(self._econ_events),
        }

    # ---------------- regime schedule ----------------

    def _sample_regimes(self):
        """Fill cost/conv/budget schedules with blocky economic regimes."""
        rng = self.rng
        lo, hi = np.log(0.02), np.log(4.0)
        day = 0
        while day < self.days:
            block = int(rng.integers(8, 18))
            end = min(day + block, self.days)
            r = rng.random()
            mult = np.ones(7, dtype=np.float32)
            conv, bud = 1.0, self.BASE_BUDGET
            if r < 0.55:          # normal, jittered prices
                mult[1:4] = np.exp(rng.uniform(lo, hi, 3))   # info
                mult[4:6] = np.exp(rng.uniform(np.log(0.3),
                                               np.log(3.0), 2))
            elif r < 0.70:        # cheap intelligence shock
                mult[1:4] = rng.uniform(0.005, 0.05, 3)
            elif r < 0.85:        # pricey intel + weak demand
                mult[1:4] = rng.uniform(2.0, 8.0, 3)
                conv = rng.uniform(0.4, 0.7)
                bud = rng.uniform(40.0, 80.0)
            else:                 # capital crunch or boom
                if rng.random() < 0.5:
                    bud = rng.uniform(15.0, 35.0)      # crunch
                else:
                    conv = rng.uniform(1.2, 1.5)       # boom
                    bud = rng.uniform(120.0, 160.0)
            self.cost_sched[day:end] = self.BASE_COSTS * mult
            self.conv_sched[day:end] = conv
            self.budget_sched[day:end] = bud
            day = end

    # ---------------- observation ----------------

    def _visible_signals(self, i):
        """Noisy projection of hidden state into the 6 sensory channels."""
        rng = self.rng
        s = self.sigma[i]
        obs = np.array([
            np.clip(0.55 * self.intent[i] + 0.45 * self.urgency[i]
                    + self.inflate[i], 0, 1),      # funding (can lie)
            0.50 * self.urgency[i] + 0.50 * self.icp[i],      # hiring
            0.75 * self.intent[i] + 0.25 * self.urgency[i],   # intent
            1.0 - self.champion[i],                           # job_change
            0.55 * (1 - self.intent[i]) + 0.45 * self.fatigue[i],  # negative
            0.60 * self.urgency[i] + 0.40 * self.competitor[i],    # trigger
        ], dtype=np.float32)
        return np.clip(obs + rng.normal(0, s, 6), 0, 1)

    def observe(self, day):
        idxs = np.where(self.active)[0]
        m = len(idxs)
        obs = np.zeros((m, 16), dtype=np.float32)
        for k, i in enumerate(idxs):
            obs[k, :6] = self._visible_signals(i)
        # budget_frac is captured at day start (apply() runs once per
        # day over all accounts), so the fly can't see within-day
        # depletion — the hard cap is enforced by env fallback anyway.
        obs[:, 6] = 1.0 - self.spend_today / max(self.daily_budget, 1e-9)
        obs[:, 7] = day / max(1, self.days - 1)                 # day frac
        obs[:, 8] = self.researched[idxs].astype(np.float32)
        obs[:, 9] = self.enriched[idxs].astype(np.float32)
        # price senses: today's cost relative to base, clipped [0, 2]
        rel = np.clip(self.costs / np.where(self.BASE_COSTS == 0, 1,
                                            self.BASE_COSTS), 0, 2)
        obs[:, 10] = rel[RESEARCH]
        obs[:, 11] = rel[ENRICH]
        obs[:, 12] = rel[EMAIL]
        obs[:, 13] = rel[ESCALATE]
        obs[:, 14] = self.daily_budget / self.BASE_BUDGET  # budget level
        # hidden context: pain revealed only by RESEARCH; 0.5 = unknown
        obs[:, 15] = np.where(self.researched[idxs],
                              self.pain[idxs], 0.5)
        return obs, idxs

    # ---------------- world dynamics ----------------

    def start_day(self, day):
        self.day = day

        # settle yesterday's conservation dividend before resetting spend:
        # leftover budget is worth conserve_coef/day, split across that
        # day's decision slots so the reward varies within the episode
        if self.day_slots and self.conserve_coef > 0:
            div = self.conserve_coef * (self.daily_budget
                                        - self.spend_today)
            share = div / len(self.day_slots)
            for s in self.day_slots:
                self._add_ledger(s, share, 0.0)
        self.day_slots = []
        self.spend_today = 0.0

        # today's regime takes effect (after the dividend used
        # yesterday's budget)
        self.costs = self.cost_sched[day]
        self.conv_mult = float(self.conv_sched[day])
        self.daily_budget = float(self.budget_sched[day])

        # flush delayed outcomes due today
        still = []
        for resolve_day, slot, reward in self.pending:
            if resolve_day <= day:
                self._add_ledger(slot, reward)
            else:
                still.append((resolve_day, slot, reward))
        self.pending = still

        rng = self.rng
        # random market events mutate hidden state
        n = self.n
        hot = rng.random(n) < 0.01
        self.intent = np.clip(self.intent + hot * 0.4, 0, 1)
        self.urgency = np.clip(self.urgency + hot * 0.5, 0, 1)
        churn = (rng.random(n) < 0.008)
        self.champion = np.where(churn, 0.0, self.champion)
        self.intent = np.clip(self.intent - churn * 0.25, 0, 1)
        news = rng.random(n) < 0.01
        self.intent = np.clip(self.intent - news * 0.2, 0, 1)

        # slow drift + fatigue decay
        self.fatigue *= 0.95

        # buying-window expiry on active-but-ignored accounts; with
        # expire_all every unpursued account past its window expires —
        # parking a dead account forever stops being free
        expired = (day > self.win_end) & self.active & ~self.pursued \
            & ((self.intent > 0.6) | self.expire_all)
        for i in np.where(expired)[0]:
            slot = int(self.last_slot[i])
            if slot >= 0:
                self._add_ledger(slot, self.outcome_reward["expired"])
            self.active[i] = False

    def apply(self, day, idxs, actions, slot_of):
        """Apply one action per active account. `slot_of(k)` maps the k-th
        decision to a global credit slot. Rewards land in self.credit
        (immediate) and self.pending (delayed). Returns the actually
        executed actions, including budget fallbacks to WAIT."""
        rng = self.rng
        executed = np.empty(len(idxs), dtype=np.int64)
        for k, i in enumerate(idxs):
            a = int(actions[k])
            slot = slot_of(k)
            self.last_slot[i] = slot
            self.day_slots.append(slot)

            cost = self.costs[a]
            if self.spend_today + cost > self.daily_budget:
                a = WAIT  # can't afford -> forced inaction
                cost = self.costs[WAIT]
            executed[k] = a

            self.spend_today += cost
            self.stats["spent"] += cost
            self.stats["rent_paid"] += self.rent
            self._add_ledger(
                slot,
                ACTION_PENALTY[a] - self.lambda_cost * cost - self.rent,
                ACTION_PENALTY[a] - self.rent,
            )

            if a == WAIT or a == OBSERVE:
                if a == OBSERVE:
                    # cheap passive watching: shrinks noise but can
                    # NEVER make you certain — information has tiers.
                    self.sigma[i] = max(0.35, self.sigma[i] * 0.95)
            elif a == RESEARCH:
                # paid cognition: reveals hidden context (pain) and
                # unlocks the outreach boosts below.
                self.researched[i] = True
                self.sigma[i] = 0.07
                self.stats["researches"] += 1
            elif a == ENRICH:
                self.enriched[i] = True
                self.stats["enriches"] += 1
            elif a == IGNORE:
                self.active[i] = False
                self.stats["ignored"] += 1
                if self.kill_bonus or self.kill_penalty:
                    # calibrated write-off: pays most on dead accounts,
                    # least on hot ones — uses the same hidden quality
                    # signal as outcome rewards, not observation leakage.
                    # kill_penalty scales with intent, so at
                    # penalty == bonus a hot kill costs what a cold
                    # kill pays.
                    self._add_ledger(
                        slot, self.kill_bonus * (1.0 - self.intent[i])
                        - self.kill_penalty * self.intent[i])
            elif a in (EMAIL, ESCALATE):
                self._outreach(day, i, slot, escalate=(a == ESCALATE))
        return executed

    def _outreach(self, day, i, slot, escalate):
        rng = self.rng
        self.pursued[i] = True
        self.last_pursue[i] = day
        if escalate:
            self.stats["escalations"] += 1
        else:
            self.stats["emails"] += 1

        in_window = self.win_start[i] <= day <= self.win_end[i]
        timing = 1.0 if in_window else 0.3
        boost = 0.8 * self.researched[i] + 0.3 * self.enriched[i]
        if self.researched[i]:
            # hidden context only pays off once discovered
            if self.pain[i] > 0.55:
                boost += 0.5          # pain hypothesis confirmed
            if self.competitor[i] > 0.6:
                boost += 0.4          # incumbent is vulnerable
        boost += 0.3 if escalate else 0.0

        p_reply = 1.0 / (1.0 + np.exp(-(
            2.0 * self.intent[i]
            + 1.2 * self.icp[i]
            + 0.8 * timing
            + boost
            - 1.3 * self.fatigue[i]
            - 3.5))) * self.conv_mult

        self.fatigue[i] += 0.25 if not escalate else 0.15

        # bad outcomes first — annoying a cold account
        p_bad = np.clip(0.35 * self.fatigue[i]
                        + 0.5 * (1 - self.intent[i]), 0, 0.9)
        if rng.random() < p_bad * 0.5:
            r = self.outcome_reward["spam"] if self.fatigue[i] > 0.7 \
                else self.outcome_reward["unsubscribe"]
            key = "spam" if self.fatigue[i] > 0.7 else "unsubscribes"
            self.stats[key] += 1
            self.pending.append((day + int(rng.integers(1, 3)),
                                 slot, r))
            if self.fatigue[i] > 0.7:
                self.active[i] = False  # burned
            return

        if rng.random() >= p_reply:
            return  # silence

        self.stats["replies"] += 1
        d = int(rng.integers(1, 4))
        self.pending.append((day + d, slot, self.outcome_reward["reply"]))
        day += d

        p_meet = np.clip(0.15 + 0.5 * self.intent[i]
                         + 0.15 * self.enriched[i], 0, 0.9)
        if rng.random() >= p_meet:
            return
        self.stats["meetings"] += 1
        d = int(rng.integers(4, 9))
        self.pending.append((day + d, slot, self.outcome_reward["meeting"]))
        day += d

        p_opp = np.clip(0.2 + 0.5 * self.icp[i]
                        + 0.2 * self.researched[i], 0, 0.9)
        if rng.random() >= p_opp:
            return
        self.stats["opportunities"] += 1
        d = int(rng.integers(5, 13))
        self.pending.append((day + d, slot,
                             self.outcome_reward["opportunity"]))
        day += d

        p_close = np.clip(0.2 + 0.4 * self.intent[i]
                          + 0.3 * self.icp[i], 0, 0.95)
        if rng.random() < p_close:
            self.stats["closed"] += 1
            self.pending.append(
                (day + int(rng.integers(8, 18)), slot,
                 self.outcome_reward["closed"]))
            self.active[i] = False  # won; out of the pipeline

    def end_episode(self):
        """Settle the final day's dividend + flush pending rewards."""
        if self.day_slots and self.conserve_coef > 0:
            div = self.conserve_coef * (self.daily_budget
                                        - self.spend_today)
            share = div / len(self.day_slots)
            for s in self.day_slots:
                self._add_ledger(s, share, 0.0)
        self.day_slots = []
        for _, slot, reward in self.pending:
            self._add_ledger(slot, reward)
        self.pending = []
        checks = self.ledger_invariants()
        if not checks["credit_conserved"] or not checks["econ_conserved"]:
            raise RuntimeError(f"ledger conservation failed: {checks}")
