########################################
# sumo_env.py – Gym wrapper for SUMO‑RL #
########################################
"""Build a SUMO‑RL environment from PeMS CSVs.

*   Compatible with SUMO ≤ 1.14  (Apple‑Silicon source build)
*   Works with sumo‑rl 1.4.x     (single‑TLS interface)

The helper injects exactly one traffic‑light if netgenerate omits it.
"""

from __future__ import annotations
import os, random, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import gymnasium as gym
import pandas as pd
from gymnasium import spaces

# ---------- SUMO setup -----------------------------------------------------
if "SUMO_HOME" not in os.environ:
    guess = Path.home() / "sumo-local/share/sumo"
    if not guess.exists():
        raise EnvironmentError("SUMO_HOME not set and default path missing.")
    os.environ["SUMO_HOME"] = str(guess)

TOOLS = Path(os.environ["SUMO_HOME"]) / "tools"
sys.path.append(str(TOOLS))

from sumo_rl import SumoEnvironment   # after TOOLS in path

# ---------- paths ----------------------------------------------------------
NET_DIR, ROUTE_DIR = Path("nets"), Path("routes")
NET_DIR.mkdir(exist_ok=True); ROUTE_DIR.mkdir(exist_ok=True)
NET_FILE = NET_DIR / "4way.net.xml"

# ---------- network generation --------------------------------------------
def _generate_network() -> None:
    subprocess.run(
        ["netgenerate", "--grid", "--grid.number=2", "--output-file", NET_FILE],
        check=True
    )
    tree = ET.parse(NET_FILE); net = tree.getroot()

    if not any(j.get("type") == "traffic_light" for j in net.iter("junction")):
        j = next(net.iter("junction"))
        j.set("type", "traffic_light")
        tl = ET.SubElement(net, "tlLogic", id=j.get("id"),
                           type="static", programID="0", offset="0")
        # ------- 4‑phase program (NS‑green / yellow / EW‑green / yellow) ----
        ET.SubElement(tl, "phase", duration="31", state="GrGr")
        ET.SubElement(tl, "phase", duration="6",  state="yryr")
        ET.SubElement(tl, "phase", duration="31", state="rGrG")
        ET.SubElement(tl, "phase", duration="6",  state="ryry")
        print(f"[sumo_env] Patched {j.get('id')} → traffic_light")
    tree.write(NET_FILE)


def _edges_from_net() -> List[str]:
    tree = ET.parse(NET_FILE)
    return [e.get("id") for e in tree.getroot().iter("edge")
            if e.get("function") is None]

# ---------- route generation ----------------------------------------------
def _csv_to_route(csv_path: Path) -> Path:
    out = ROUTE_DIR / f"{csv_path.stem}.rou.xml"
    if out.exists():
        return out

    edges = _edges_from_net()
    df = pd.read_csv(csv_path)

    with out.open("w") as f:
        f.write("<routes>\n")
        veh_id, depart = 0, 0.0
        # for _, row in df.iterrows():
        for _, row in df.iloc[:120].iterrows():   # 120 × 5‑s rows ≈ first 10 min
            arrivals = int(row.get("Total Flow", 0) // 12)  # ≈ veh / 5 s
            for _ in range(arrivals):
                edge = random.choice(edges)                # **single edge**
                f.write(f'  <vehicle id="veh{veh_id}" depart="{depart:.1f}">\n')
                f.write(f'    <route edges="{edge}"/>\n  </vehicle>\n')
                veh_id += 1
            depart += 5.0
        f.write("</routes>\n")
    print(f"[sumo_env] Built {out}")
    return out

# ---------- Gym wrapper ----------------------------------------------------
class _SingleTLWrapper(gym.Env):
    def __init__(self, raw: SumoEnvironment):
        if not raw.ts_ids:
            raise RuntimeError("sumo_rl reports no traffic‑light.")
        self.raw, self.tls_id = raw, raw.ts_ids[0]

        if isinstance(getattr(raw, "action_spaces"), dict):
            self.action_space = raw.action_spaces[self.tls_id]
            obs_space         = raw.observation_spaces[self.tls_id]
        else:                                # older API: methods
            self.action_space = raw.action_spaces(self.tls_id)
            obs_space         = raw.observation_spaces(self.tls_id)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=obs_space.shape, dtype=float
        )

    def reset(self, **kw):
        obs = self.raw.reset(**kw)
        if isinstance(obs, tuple): obs = obs[0]
        return obs[self.tls_id], {}

    def step(self, action: int):
        obs, rew, done, info = self.raw.step({self.tls_id: int(action)})
        return (
            obs[self.tls_id],          # observation   → ndarray
            rew[self.tls_id],          # reward        → float
            done[self.tls_id],         # terminated    → bool
            False,                     # truncated     → False (no time‑limit)
            info,
        )
    def render(self,*a,**k): return self.raw.render(*a,**k)
    def close(self): self.raw.close()

# ---------- public factory -------------------------------------------------
def make_demand_env(csv_root: str | Path):
    csv_root = Path(csv_root)
    csvs = sorted(csv_root.glob("*.csv*")) if csv_root.is_dir() else [csv_root]
    _generate_network()

    class _CsvEnv(SumoEnvironment):
        def __init__(self):
            super().__init__(net_file=str(NET_FILE),
                             route_file=str(_csv_to_route(csvs[0])),
                             use_gui=False, num_seconds=3600)
        def reset(self,*a,**k):
            self.route_file = str(_csv_to_route(random.choice(csvs)))
            return super().reset(*a,**k)
    return _SingleTLWrapper(_CsvEnv())
