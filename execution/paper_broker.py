from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _debug_edge_pipeline() -> bool:
    v = os.environ.get("FIE_DEBUG_EDGE_PIPELINE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _edge_enter_log() -> bool:
    v = os.environ.get("FIE_EDGE_ENTER_LOG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class _Position:
    entry_price: float
    side: int           # +1 long, -1 short
    size: float
    notional: float
    tp: float           # take-profit price level
    sl: float           # stop-loss price level
    steps_remaining: int
    total_steps: int
    meta: dict
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0


@dataclass
class PaperBroker:
    """
    Paper broker с реализованным PnL + TP/SL.

    Выход происходит при ПЕРВОМ из условий:
      1. current_price достиг TP или SL
      2. holding_steps >= max_holding

    side = +1 (long, BUY_YES):   TP выше entry, SL ниже entry
    side = -1 (short, BUY_NO):   TP ниже entry, SL выше entry

    tp_pct / sl_pct — процентный отступ от entry_price (default 0.002 = 0.2%).
    """

    capital: float = 1.0
    hold_steps: int = 5
    tp_pct: float = 0.002
    sl_pct: float = 0.002
    # Жёсткий потолок убытка по открытой позиции: выход если unrealized PnL < -notional * frac (0 = выкл).
    hard_loss_cap_frac: float = 0.0
    open_positions: list[_Position] = field(default_factory=list)

    def enter(
        self,
        action: str,
        size: float,
        entry_price: float,
        hold_steps: Optional[int] = None,
        tp_pct: Optional[float] = None,
        sl_pct: Optional[float] = None,
        meta: Optional[dict] = None,
    ) -> None:
        """Открыть позицию. PnL не считается до выхода."""
        if action == "HOLD":
            return

        side = +1 if action == "BUY_YES" else -1
        size = _clamp(float(size), 0.0, 1.0)
        entry_price = _clamp(float(entry_price), 0.0, 1.0)
        notional = self.capital * size
        steps = hold_steps if hold_steps is not None else self.hold_steps

        tp_p = tp_pct if tp_pct is not None else self.tp_pct
        sl_p = sl_pct if sl_pct is not None else self.sl_pct

        # Long:  TP выше, SL ниже
        # Short: TP ниже, SL выше
        tp = entry_price * (1 + tp_p * side)
        sl = entry_price * (1 - sl_p * side)

        # Копия dict: чтобы meta с edge_real не мутировала снаружи до закрытия
        _meta = dict(meta) if meta else {}
        if _debug_edge_pipeline() or _edge_enter_log():
            print(
                f"[EDGE-PIPE broker.enter] meta_keys={sorted(_meta.keys())} "
                f"edge_real={_meta.get('edge_real')} p_model={_meta.get('p_model')}",
                flush=True,
            )
            print(f"[BROKER META] {_meta}", flush=True)

        self.open_positions.append(
            _Position(
                entry_price=entry_price,
                side=side,
                size=size,
                notional=notional,
                tp=tp,
                sl=sl,
                steps_remaining=steps,
                total_steps=steps,
                meta=_meta,
            )
        )

    def step(self, current_price: float) -> list[dict]:
        """
        Вызывать каждый тик. Возвращает список закрытых позиций с realized PnL.
        Позиция закрывается если:
          - достигнут TP или SL (по текущей цене)
          - истекло max_holding шагов
        """
        current_price = _clamp(float(current_price), 0.0, 1.0)
        closed: list[dict] = []
        still_open: list[_Position] = []

        for pos in self.open_positions:
            pos.steps_remaining -= 1
            steps_taken = pos.total_steps - pos.steps_remaining

            excursion = (current_price - pos.entry_price) * pos.side
            pos.max_favorable_excursion = max(pos.max_favorable_excursion, excursion)
            pos.max_adverse_excursion = min(pos.max_adverse_excursion, excursion)

            unrealized = excursion * pos.notional
            if self.hard_loss_cap_frac > 0:
                max_loss = pos.notional * float(self.hard_loss_cap_frac)
                if unrealized < -max_loss:
                    pnl = unrealized
                    self.capital += pnl
                    if _debug_edge_pipeline():
                        print(f"[DEBUG CLOSE META] {pos.meta}", flush=True)
                    _fill = dict(pos.meta)
                    _fill.update(
                        {
                            "entry_price": pos.entry_price,
                            "exit_price": current_price,
                            "side": "long" if pos.side == 1 else "short",
                            "size": pos.size,
                            "tp": pos.tp,
                            "sl": pos.sl,
                            "holding_steps": steps_taken,
                            "hold_min": float(steps_taken),
                            "exit_reason": "loss_cap",
                            "exit_signal": "loss_cap",
                            "timeout_hit": False,
                            "reverse_hit": False,
                            "mfe": float(pos.max_favorable_excursion),
                            "mae": float(abs(pos.max_adverse_excursion)),
                            "pnl": pnl,
                            "capital": self.capital,
                        }
                    )
                    closed.append(_fill)
                    continue

            hit_tp = False
            hit_sl = False

            if pos.side == 1:   # long
                hit_tp = current_price >= pos.tp
                hit_sl = current_price <= pos.sl
            else:               # short
                hit_tp = current_price <= pos.tp
                hit_sl = current_price >= pos.sl

            hit_timeout = pos.steps_remaining <= 0

            if hit_tp or hit_sl or hit_timeout:
                exit_reason = "tp" if hit_tp else ("sl" if hit_sl else "timeout")
                pnl = (current_price - pos.entry_price) * pos.side * pos.notional
                self.capital += pnl
                if _debug_edge_pipeline():
                    print(f"[DEBUG CLOSE META] {pos.meta}", flush=True)
                _fill = dict(pos.meta)
                _fill.update(
                    {
                        "entry_price": pos.entry_price,
                        "exit_price": current_price,
                        "side": "long" if pos.side == 1 else "short",
                        "size": pos.size,
                        "tp": pos.tp,
                        "sl": pos.sl,
                        "holding_steps": steps_taken,
                        "hold_min": float(steps_taken),
                        "exit_reason": exit_reason,
                        "exit_signal": exit_reason,
                        "timeout_hit": bool(hit_timeout),
                        "reverse_hit": False,
                        "mfe": float(pos.max_favorable_excursion),
                        "mae": float(abs(pos.max_adverse_excursion)),
                        "pnl": pnl,
                        "capital": self.capital,
                    }
                )
                closed.append(_fill)
            else:
                still_open.append(pos)

        self.open_positions = still_open
        return closed
