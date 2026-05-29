from __future__ import annotations

import time
import os
import sys

from markets.market_engine import compute_edge_zone, detect_edge
from alerts.alert_engine import check_and_alert
from agents.role_based_decision import build_role_votes, weighted_vote
from agents.debate import run_local_debate
from prediction.regime_router import route_model, describe_regime
from learning.decision_logger import append_decision_log
from signals.run_signal_scan import run_signal_scan

# Важно: `future-intelligence-engine` кладём в sys.path ПОСЛЕ импорта root-модулей,
# иначе имя пакета `agents` будет резолвиться в FIE-версию и затрёт `agents/agent_swarm.py`.
_FIE_PATH = os.path.join(os.path.dirname(__file__), "..", "future-intelligence-engine")
if _FIE_PATH not in sys.path:
    sys.path.insert(0, os.path.abspath(_FIE_PATH))

from events.event_discovery import discover_events  # noqa: E402
from simulation.scenario_engine import generate_scenarios  # noqa: E402
from simulation.timeline_engine import build_timeline  # noqa: E402
from database.store_prediction import save_prediction as store_prediction  # noqa: E402


def autonomous_loop() -> None:
    while True:
        print("\n🌍 FIE scanning world...")

        # ── Signal Layer ──────────────────────────────────────────────────
        run_signal_scan()
        events = discover_events()

        for event in events:
            print("\n⚡ Event detected:", event)

            # ── Scenario Layer ────────────────────────────────────────────
            scenarios = generate_scenarios(event)
            timeline = build_timeline(event, scenarios)

            # ── Regime Filter ─────────────────────────────────────────────
            zone = compute_edge_zone()
            model_id = route_model(zone)
            print(f"\n🔀 Regime: {describe_regime(zone)}")

            if model_id is None:
                prediction = {
                    "probability": 0.5,
                    "confidence": 0.0,
                    "agents_count": 0,
                    "skipped": True,
                    "skip_reason": zone.reason,
                    "model": None,
                    "state": zone.state,
                    "market_state": zone.market_state,
                    "regime_key": zone.regime_key,
                    "scen_accumulation": zone.scen_accumulation,
                    "scen_breakout_confirmed": zone.scen_breakout_confirmed,
                    "scen_breakout_suspicious": zone.scen_breakout_suspicious,
                    "scen_trend_flow": zone.scen_trend_flow,
                    "breakout_up": zone.breakout_up,
                    "funding_stress": zone.funding_stress,
                    "liq_spike": zone.liq_spike,
                    "momentum_up": zone.momentum_up,
                    "oi_strength": zone.oi_strength,
                    "gate_score": zone.gate_score,
                    "gate_threshold": zone.gate_threshold,
                    "agents_votes": [],
                    "debate_log": [],
                    "disagreement": 1.0,
                    "agents_disagree": True,
                }
            else:
                # ── Agent Layer ───────────────────────────────────────────
                # 1. Initial votes (Smart Whale style reasoning)
                agent_votes = build_role_votes(event, zone)

                # 2. Debate round: агенты слышат друг друга и корректируют
                agent_votes, debate_log = run_local_debate(agent_votes, zone)

                # 3. Weighted consensus
                voted = weighted_vote(agent_votes)

                # ── Prediction ────────────────────────────────────────────
                min_conf = float(os.environ.get("FIE_MIN_AGENTS_CONFIDENCE", "0.6") or "0.6")
                conf = float(voted["confidence"])
                position_size = conf if conf >= min_conf else conf * 0.5

                prediction = {
                    "probability": voted["final_probability"],
                    "confidence": voted["confidence"],
                    "position_size": round(position_size, 3),
                    "agents_count": len(agent_votes),
                    "skipped": False,
                    "skip_reason": None,
                    "model": model_id,
                    "state": zone.state,
                    "market_state": zone.market_state,
                    "regime_key": zone.regime_key,
                    "scen_accumulation": zone.scen_accumulation,
                    "scen_breakout_confirmed": zone.scen_breakout_confirmed,
                    "scen_breakout_suspicious": zone.scen_breakout_suspicious,
                    "scen_trend_flow": zone.scen_trend_flow,
                    "breakout_up": zone.breakout_up,
                    "funding_stress": zone.funding_stress,
                    "liq_spike": zone.liq_spike,
                    "momentum_up": zone.momentum_up,
                    "oi_strength": zone.oi_strength,
                    "gate_score": zone.gate_score,
                    "gate_threshold": zone.gate_threshold,
                    "agents_votes": agent_votes,
                    "debate_log": debate_log,
                    "disagreement": voted["disagreement"],
                    "agents_disagree": voted["agents_disagree"],
                }

                # Print agent debate for observability
                print("\n🧠 Agent Debate:")
                for line in debate_log:
                    print(" ", line)
                print("\n🗳️  Agent Votes (post-debate):")
                for v in agent_votes:
                    print(f"  [{v['agent']}] p={v['probability']:.3f} | {v['reasoning'][:120]}")

            print("\n📊 Prediction:", prediction)
            if not bool(prediction.get("skipped", False)):
                store_prediction(
                    event,
                    float(prediction.get("probability", 0.5)),
                    0.5,  # TODO: подключить реальный market probability benchmark
                    float(zone.market_state.get("confidence", 0.0)),
                )

            append_decision_log(
                {
                    "event": event,
                    "regime": zone.regime_key,
                    "model": prediction.get("model"),
                    "state": zone.market_state,
                    "scenarios": {
                        "scen_accumulation": zone.scen_accumulation,
                        "scen_breakout_confirmed": zone.scen_breakout_confirmed,
                        "scen_breakout_suspicious": zone.scen_breakout_suspicious,
                        "scen_trend_flow": zone.scen_trend_flow,
                        "breakout_up": zone.breakout_up,
                        "funding_stress": zone.funding_stress,
                        "liq_spike": zone.liq_spike,
                        "momentum_up": zone.momentum_up,
                        "oi_strength": zone.oi_strength,
                    },
                    "debate_log": prediction.get("debate_log", []),
                    "agents_votes": prediction.get("agents_votes", []),
                    "final_prob": prediction.get("probability", 0.5),
                    "confidence": prediction.get("confidence", 0.0),
                    "outcome": prediction.get("outcome"),
                }
            )

            market_signal = detect_edge(event, prediction.get("probability"), zone=zone)
            print("\n💰 Market Signal:", market_signal)

            check_and_alert(event, market_signal)

        print("\n⏳ Sleeping before next scan...\n")
        time.sleep(300)
