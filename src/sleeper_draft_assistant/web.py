from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from typing import Any, Literal
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, model_validator

from .cache import cache_directory
from .cli import DEFAULT_USERNAME, LEAGUE_IDS
from .models import LeagueContext, Recommendation
from .native_engine import create_recommendation_engine
from .rankings import load_consensus_board
from .simulator import SimulationReport
from .sleeper import SleeperClient
from .valuation import build_player_values
from .management import TeamService
from .team_ui import TEAM_HTML


LOGGER = logging.getLogger(__name__)
COOKIE_NAME = "draft_assist_session"
SESSION_SECONDS = 12 * 60 * 60
DISPLAY_RECOMMENDATIONS = 5


class RecommendationRequest(BaseModel):
    league: str = "shield-ai"
    candidates: int = Field(default=15, ge=1, le=25)
    runs_per_candidate: int = Field(default=10_000, ge=100, le=25_000)
    workers: Literal["auto"] | int = "auto"

    @model_validator(mode="after")
    def validate_request(self) -> RecommendationRequest:
        if self.league not in LEAGUE_IDS:
            raise ValueError("unknown saved league")
        if isinstance(self.workers, int) and not 1 <= self.workers <= 64:
            raise ValueError("workers must be 'auto' or an integer from 1 to 64")
        if self.candidates * self.runs_per_candidate > 500_000:
            raise ValueError("the web service is limited to 500,000 total rollouts")
        return self


class TeamRequest(BaseModel):
    league: str = "shield-ai"
    league_id: str | None = Field(default=None, pattern=r"^[0-9]{1,24}$")
    username: str = Field(default=DEFAULT_USERNAME, min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    action: Literal["lineup", "waivers", "trades"] = "lineup"
    week: int | None = Field(default=None, ge=1, le=18)
    season: int | None = Field(default=None, ge=2020, le=2100)
    mode: Literal["projection", "tiers"] = "projection"
    limit: int = Field(default=10, ge=1, le=25)
    refresh: bool = False
    confirm_tiers_week: bool = False

    @model_validator(mode="after")
    def validate_league(self) -> TeamRequest:
        if self.league_id is None and self.league not in LEAGUE_IDS:
            raise ValueError("unknown saved league")
        if self.confirm_tiers_week and (self.week is None or self.season is None):
            raise ValueError("confirm tiers for an explicit season and week")
        return self


@dataclass(slots=True)
class LeagueRuntime:
    context: LeagueContext
    simulator: Any
    engine_name: str
    draft_id: str
    signature: str
    lock: threading.RLock = field(default_factory=threading.RLock)


class DraftService:
    """Keeps immutable model data warm and only refreshes Sleeper picks."""

    def __init__(self, client: SleeperClient | None = None) -> None:
        self.client = client or SleeperClient()
        self._runtimes: dict[str, LeagueRuntime] = {}
        self._runtime_lock = threading.RLock()

    @staticmethod
    def _pick_signature(picks: list[dict[str, Any]]) -> str:
        encoded = "|".join(
            f"{int(pick['pick_no'])}:{pick.get('player_id', '')}" for pick in picks
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def _create_runtime(self, league_name: str) -> LeagueRuntime:
        context = self.client.sync(LEAGUE_IDS[league_name], DEFAULT_USERNAME)
        cache_dir = cache_directory()
        board = load_consensus_board(cache_dir)
        values = build_player_values(
            board,
            context.rules,
            cache_dir,
            int(context.league["season"]),
        )
        biases = self.client.manager_position_biases(
            context.league,
            max_seasons=3,
            additional_league_ids=LEAGUE_IDS.values(),
        )
        requested_engine = os.environ.get("DRAFT_ENGINE", "auto")
        simulator, engine_name = create_recommendation_engine(
            requested_engine,
            context,
            values.players,
            biases,
            seed=2026 + len(context.picks),
        )
        return LeagueRuntime(
            context=context,
            simulator=simulator,
            engine_name=engine_name,
            draft_id=str(context.draft["draft_id"]),
            signature=self._pick_signature(context.picks),
        )

    def _runtime(self, league_name: str) -> LeagueRuntime:
        with self._runtime_lock:
            runtime = self._runtimes.get(league_name)
            if runtime is None:
                runtime = self._create_runtime(league_name)
                self._runtimes[league_name] = runtime
            return runtime

    def _refresh_picks(self, runtime: LeagueRuntime) -> list[dict[str, Any]]:
        picks = self.client.picks(runtime.draft_id)
        signature = self._pick_signature(picks)
        if signature != runtime.signature:
            runtime.context.picks = picks
            runtime.simulator.update_picks(picks, 2026 + len(picks))
            runtime.signature = signature
        return picks

    @staticmethod
    def _last_pick(picks: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not picks:
            return None
        pick = picks[-1]
        metadata = pick.get("metadata") or {}
        name = metadata.get("full_name") or " ".join(
            part
            for part in (metadata.get("first_name"), metadata.get("last_name"))
            if part
        )
        return {
            "pick": int(pick["pick_no"]),
            "name": name or str(pick.get("player_id") or "Unknown player"),
            "position": metadata.get("position") or "",
            "team": metadata.get("team") or "",
        }

    def status(self, league_name: str) -> dict[str, Any]:
        runtime = self._runtime(league_name)
        with runtime.lock:
            picks = self._refresh_picks(runtime)
            return {
                "league": league_name,
                "league_name": runtime.context.league["name"].strip(),
                "draft_id": runtime.draft_id,
                "completed_picks": len(picks),
                "total_picks": runtime.context.rules.teams * runtime.context.rules.rounds,
                "signature": runtime.signature,
                "last_pick": self._last_pick(picks),
            }

    def recommend(self, request: RecommendationRequest) -> dict[str, Any]:
        total_started = time.perf_counter()
        already_prepared = request.league in self._runtimes
        runtime = self._runtime(request.league)
        prepared = time.perf_counter()
        with runtime.lock:
            picks = self._refresh_picks(runtime)
            synced = time.perf_counter()
            total_rollouts = request.candidates * request.runs_per_candidate
            report = runtime.simulator.recommend(
                total_rollouts,
                request.candidates,
                request.workers,
            )
            simulated = time.perf_counter()
            payload = self._report_payload(runtime, report, picks)
        payload["timing"] = {
            "cold_preparation_seconds": round(prepared - total_started, 3),
            "board_sync_seconds": round(synced - prepared, 3),
            "simulation_seconds": round(simulated - synced, 3),
            "total_seconds": round(simulated - total_started, 3),
            "model_was_warm": already_prepared,
        }
        payload["requested"] = {
            "candidates": request.candidates,
            "runs_per_candidate": request.runs_per_candidate,
            "total_rollouts": total_rollouts,
            "workers": request.workers,
        }
        return payload

    def _report_payload(
        self,
        runtime: LeagueRuntime,
        report: SimulationReport,
        picks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        teams = runtime.context.rules.teams
        round_number = (report.next_user_pick - 1) // teams + 1
        pick_in_round = (report.next_user_pick - 1) % teams + 1
        recommendations = [
            _recommendation_payload(rank, item)
            for rank, item in enumerate(
                report.recommendations[:DISPLAY_RECOMMENDATIONS], start=1
            )
        ]
        return {
            "league": runtime.context.league_id,
            "league_name": runtime.context.league["name"].strip(),
            "draft_id": runtime.draft_id,
            "draft_slot": runtime.context.draft_slot,
            "completed_picks": len(picks),
            "total_picks": teams * runtime.context.rules.rounds,
            "signature": runtime.signature,
            "last_pick": self._last_pick(picks),
            "on_clock": report.on_clock,
            "next_pick": report.next_user_pick,
            "next_pick_label": f"{round_number}.{pick_in_round:02d}",
            "engine": runtime.engine_name,
            "actual_rollouts": report.total_rollouts,
            "recommendations": recommendations,
        }


def _recommendation_payload(rank: int, item: Recommendation) -> dict[str, Any]:
    return {
        "rank": rank,
        "player_id": item.player.sleeper_id,
        "name": item.player.name,
        "position": item.player.position,
        "team": item.player.team,
        "bye": item.player.bye,
        "model_score": round(item.mean_score, 2),
        "top_roster_rate": round(item.top_roster_rate, 6),
        "availability_rate": round(item.availability_rate, 6),
        "selection_rate": round(item.selection_rate, 6),
        "lineup_gain": round(item.starter_gain, 2),
        "next_turn_availability": (
            round(item.next_pick_availability, 6)
            if item.next_pick_availability is not None
            else None
        ),
        "wait_cost": round(item.wait_cost, 2),
        "samples": item.samples,
    }


def _password_digest(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _auth_settings() -> tuple[str, str]:
    return (
        os.environ.get("APP_PASSWORD_HASH", "").strip().lower(),
        os.environ.get("SESSION_SECRET", "").strip(),
    )


def _session_token(timestamp: int | None = None) -> str:
    _, signing_secret = _auth_settings()
    issued = timestamp if timestamp is not None else int(time.time())
    payload = str(issued)
    signature = hmac.new(
        signing_secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return f"{payload}.{signature}"


def _valid_session(token: str | None, timestamp: int | None = None) -> bool:
    expected_hash, signing_secret = _auth_settings()
    if not token or len(expected_hash) != 64 or not signing_secret:
        return False
    try:
        issued_text, supplied_signature = token.split(".", 1)
        issued = int(issued_text)
    except (ValueError, TypeError):
        return False
    now = timestamp if timestamp is not None else int(time.time())
    if issued > now + 60 or now - issued > SESSION_SECONDS:
        return False
    expected_signature = _session_token(issued).split(".", 1)[1]
    return secrets.compare_digest(expected_signature, supplied_signature)


def _require_auth(request: Request) -> None:
    if not _valid_session(request.cookies.get(COOKIE_NAME)):
        raise HTTPException(status_code=401, detail="Sign in to use DraftAssist")


app = FastAPI(title="Sleeper DraftAssist", docs_url=None, redoc_url=None)
service = DraftService()
team_service = TeamService()


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; form-action 'self'"
    )
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if _valid_session(request.cookies.get(COOKIE_NAME)):
        return RedirectResponse("/", status_code=303)
    expected_hash, signing_secret = _auth_settings()
    configured = len(expected_hash) == 64 and bool(signing_secret)
    return HTMLResponse(_login_html(configured=configured))


@app.post("/login")
async def login(request: Request):
    expected_hash, signing_secret = _auth_settings()
    if len(expected_hash) != 64 or not signing_secret:
        raise HTTPException(status_code=503, detail="Login is not configured")
    body = (await request.body())[:4096].decode("utf-8", errors="replace")
    password = parse_qs(body).get("password", [""])[0]
    if not secrets.compare_digest(_password_digest(password), expected_hash):
        await asyncio.sleep(0.4)
        return HTMLResponse(_login_html(configured=True, error=True), status_code=401)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        COOKIE_NAME,
        _session_token(),
        max_age=SESSION_SECONDS,
        httponly=True,
        secure=os.environ.get("COOKIE_SECURE", "true").lower() != "false",
        samesite="strict",
        path="/",
    )
    return response


@app.post("/logout")
def logout() -> RedirectResponse:
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if not _valid_session(request.cookies.get(COOKIE_NAME)):
        return RedirectResponse("/login", status_code=303)
    return HTMLResponse(HOME_HTML)


@app.get("/team", response_class=HTMLResponse)
def team_home(request: Request):
    if not _valid_session(request.cookies.get(COOKIE_NAME)):
        return RedirectResponse("/login", status_code=303)
    return HTMLResponse(TEAM_HTML)


@app.get("/api/team/state")
async def api_team_state(request: Request):
    _require_auth(request)
    try:
        state = await asyncio.to_thread(team_service.client.nfl_state)
        return {key: state.get(key) for key in ("season", "week", "season_type")}
    except Exception as exc:
        LOGGER.exception("Unable to load NFL week")
        raise HTTPException(status_code=502, detail="Could not load the NFL week from Sleeper.") from exc


@app.post("/api/team")
async def api_team(request: Request, options: TeamRequest):
    _require_auth(request)
    try:
        return await asyncio.to_thread(
            team_service.advise,
            league_id=options.league_id or LEAGUE_IDS[options.league],
            username=options.username, action=options.action, week=options.week,
            mode=options.mode, limit=options.limit, refresh=options.refresh,
            confirm_tiers_week=options.confirm_tiers_week, expected_season=options.season,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Unable to produce team advice")
        raise HTTPException(status_code=502, detail="Could not load team advice. Try refreshing the data.") from exc


@app.get("/api/status")
async def api_status(request: Request, league: str = "shield-ai"):
    _require_auth(request)
    if league not in LEAGUE_IDS:
        raise HTTPException(status_code=400, detail="Unknown saved league")
    try:
        return await asyncio.to_thread(service.status, league)
    except Exception as exc:
        LOGGER.exception("Unable to refresh Sleeper draft status")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/recommend")
async def api_recommend(request: Request, options: RecommendationRequest):
    _require_auth(request)
    try:
        return await asyncio.to_thread(service.recommend, options)
    except Exception as exc:
        LOGGER.exception("Unable to produce draft recommendations")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _login_html(configured: bool, error: bool = False) -> str:
    if not configured:
        message = "The service administrator must configure login before use."
    elif error:
        message = "That password was not accepted. Try again."
    else:
        message = "Private access to your live Sleeper draft assistant."
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#111827"><title>Sign in · DraftAssist</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;background:#08111f;color:#edf4ff;font:16px/1.45 system-ui,-apple-system,sans-serif}}
.card{{width:min(100%,390px);padding:30px;border:1px solid #24344d;border-radius:24px;background:#111d30;box-shadow:0 24px 70px #0008}} .mark{{width:52px;height:52px;display:grid;place-items:center;border-radius:15px;background:#75e0b4;color:#06261a;font-size:27px;font-weight:900}}
h1{{margin:22px 0 7px;font-size:29px}} p{{margin:0 0 24px;color:#a9b9cf}} label{{display:block;margin-bottom:8px;font-weight:700}} input{{width:100%;padding:15px;border:1px solid #3b4b63;border-radius:12px;background:#091321;color:white;font-size:17px;outline:none}} input:focus{{border-color:#75e0b4;box-shadow:0 0 0 3px #75e0b42b}}
button{{width:100%;margin-top:14px;padding:15px;border:0;border-radius:12px;background:#75e0b4;color:#06261a;font-size:16px;font-weight:900}} .error{{color:#ffaaa5}}
</style></head><body><main class="card"><div class="mark">D</div><h1>DraftAssist</h1>
<p class="{'error' if error or not configured else ''}">{message}</p>
<form method="post" action="/login"><label for="password">Draft-day password</label>
<input id="password" name="password" type="password" autocomplete="current-password" required {'disabled' if not configured else ''}>
<button type="submit" {'disabled' if not configured else ''}>Open draft room</button></form></main></body></html>"""


HOME_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#08111f"><title>Sleeper DraftAssist</title>
<style>
:root{--bg:#07101d;--panel:#101c2e;--line:#263750;--text:#edf5ff;--muted:#9dafc7;--mint:#75e0b4;--blue:#78a9ff;--red:#ff817a;--gold:#ffd166}
*{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at 80% -10%,#19395c 0,transparent 35%),var(--bg);color:var(--text);font:15px/1.42 system-ui,-apple-system,sans-serif;-webkit-font-smoothing:antialiased}
button,input,select{font:inherit}.shell{width:min(100%,760px);margin:auto;padding:18px 14px 60px}.top{display:flex;align-items:center;justify-content:space-between;margin:5px 2px 20px}.brand{font-size:20px;font-weight:900;letter-spacing:-.4px}.brand b{color:var(--mint)}.logout{padding:7px 10px;border:0;background:none;color:var(--muted)}
.hero{padding:20px;border:1px solid var(--line);border-radius:22px;background:linear-gradient(145deg,#14243a,#0d1727);box-shadow:0 20px 60px #0005}.eyebrow{color:var(--mint);font-size:12px;font-weight:900;letter-spacing:1.2px;text-transform:uppercase}.hero h1{margin:6px 0 2px;font-size:29px;line-height:1.1;letter-spacing:-1px}.sub{color:var(--muted)}
.controls{display:grid;grid-template-columns:1fr auto;gap:10px;margin-top:18px}select,.field input{width:100%;padding:13px;border:1px solid #354861;border-radius:12px;background:#091422;color:var(--text)}.primary{padding:13px 18px;border:0;border-radius:12px;background:var(--mint);color:#06261a;font-weight:900}.primary:disabled{opacity:.5}.watch{display:flex;gap:8px;align-items:center;margin-top:13px;color:var(--muted)}.watch input{accent-color:var(--mint);width:17px;height:17px}
.status{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:12px 0}.stat{padding:13px 12px;border:1px solid var(--line);border-radius:14px;background:#0c1727}.stat span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.7px}.stat strong{display:block;margin-top:3px;font-size:17px}.clock{color:var(--gold)}
.section-title{display:flex;justify-content:space-between;align-items:end;margin:24px 2px 10px}.section-title h2{margin:0;font-size:19px}.section-title small{color:var(--muted)}.results{display:grid;gap:9px}.pick{position:relative;padding:15px 15px 14px 58px;border:1px solid var(--line);border-radius:17px;background:var(--panel)}.rank{position:absolute;left:14px;top:15px;width:31px;height:31px;display:grid;place-items:center;border-radius:10px;background:#1c3149;color:var(--mint);font-weight:900}.player{display:flex;align-items:center;justify-content:space-between;gap:8px}.player h3{margin:0;font-size:18px}.tag{padding:4px 7px;border-radius:7px;background:#213650;color:#bcd8ff;font-size:11px;font-weight:900}.meta{margin:2px 0 11px;color:var(--muted);font-size:13px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.metric{padding:8px;border-radius:9px;background:#0a1524}.metric span{display:block;color:var(--muted);font-size:10px}.metric b{font-size:14px}
.empty{padding:30px 20px;text-align:center;border:1px dashed #334760;border-radius:17px;color:var(--muted)}.advanced{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}.advanced summary{color:var(--muted);cursor:pointer}.fields{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:11px}.field label{display:block;margin-bottom:5px;color:var(--muted);font-size:12px}
.notice{display:none;margin-top:12px;padding:12px;border-radius:11px;background:#3a1f24;color:#ffc0bc}.busy{display:none;margin:16px 0 0;color:var(--muted)}.busy.on{display:flex;align-items:center;gap:9px}.spinner{width:18px;height:18px;border:2px solid #345;border-top-color:var(--mint);border-radius:50%;animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(1turn)}}
.foot{margin:22px 4px;color:#71849e;font-size:12px}.live{display:inline-block;width:7px;height:7px;margin-right:6px;border-radius:50%;background:var(--mint);box-shadow:0 0 0 4px #75e0b422}
@media(max-width:430px){.shell{padding-left:11px;padding-right:11px}.hero{padding:17px}.hero h1{font-size:25px}.controls{grid-template-columns:1fr}.primary{width:100%}.metrics{grid-template-columns:repeat(2,1fr)}.status{gap:5px}.stat{padding:11px 8px}.stat strong{font-size:15px}}
</style></head><body><main class="shell">
<header class="top"><div class="brand">Sleeper <b>DraftAssist</b></div><a href="/team" style="color:var(--mint)">Manage team</a><form method="post" action="/logout"><button class="logout">Sign out</button></form></header>
<section class="hero"><div class="eyebrow">Live draft room</div><h1 id="leagueTitle">Shield AI After Hours</h1><div class="sub">C++ Monte Carlo advice, synced directly from Sleeper.</div>
<div class="controls"><select id="league"><option value="shield-ai">Shield AI After Hours</option><option value="hooligans">Hooligans</option><option value="unemployables">The Unemployables</option></select><button id="run" class="primary">Get recommendations</button></div>
<label class="watch"><input id="watch" type="checkbox" checked> Watch for new picks every 2 seconds</label>
<details class="advanced"><summary>Simulation settings</summary><div class="fields"><div class="field"><label for="candidates">Candidates</label><input id="candidates" type="number" min="1" max="25" value="15"></div><div class="field"><label for="runs">Runs per candidate</label><input id="runs" type="number" min="100" max="25000" value="10000"></div></div></details>
<div id="busy" class="busy"><div class="spinner"></div><span>Syncing draft and running simulations…</span></div><div id="notice" class="notice"></div></section>
<section id="status" class="status"><div class="stat"><span>Draft status</span><strong>Ready</strong></div><div class="stat"><span>Your next pick</span><strong>—</strong></div><div class="stat"><span>Completed</span><strong>—</strong></div></section>
<div class="section-title"><h2>Top recommendations</h2><small id="timing"></small></div><section id="results" class="results"><div class="empty">Tap “Get recommendations” before the draft, then leave auto-watch on.</div></section>
<p class="foot"><span class="live"></span>Read-only assistant. Make the final selection in Sleeper. “Top roster” compares completed projected rosters; it is not championship probability.</p></main>
<script>
const $=id=>document.getElementById(id);let busy=false,currentSignature=null,pollTimer=null;
const pct=value=>value==null?'—':(value*100).toFixed(1)+'%';
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function setBusy(value,message){busy=value;$('run').disabled=value;$('busy').classList.toggle('on',value);if(message)$('busy').querySelector('span').textContent=message}
function showError(message){$('notice').style.display='block';$('notice').textContent=message}function clearError(){$('notice').style.display='none'}
function metric(label,value){return `<div class="metric"><span>${label}</span><b>${value}</b></div>`}
function render(data){currentSignature=data.signature;$('leagueTitle').textContent=data.league_name;const state=data.on_clock?'<span class="clock">ON THE CLOCK</span>':'Waiting';$('status').innerHTML=`<div class="stat"><span>Draft status</span><strong>${state}</strong></div><div class="stat"><span>Your next pick</span><strong>${esc(data.next_pick_label)}</strong></div><div class="stat"><span>Completed</span><strong>${data.completed_picks}/${data.total_picks}</strong></div>`;
const timing=data.timing;$('timing').textContent=`${timing.total_seconds.toFixed(1)}s · ${data.actual_rollouts.toLocaleString()} runs · ${data.engine.toUpperCase()}`;
$('results').innerHTML=data.recommendations.length?data.recommendations.map(p=>`<article class="pick"><div class="rank">${p.rank}</div><div class="player"><h3>${esc(p.name)}</h3><span class="tag">${esc(p.position)}</span></div><div class="meta">${esc(p.team||'FA')}${p.bye?' · Bye '+p.bye:''} · ${p.samples.toLocaleString()} samples</div><div class="metrics">${metric('Model score',p.model_score.toFixed(1))}${metric('Top roster',pct(p.top_roster_rate))}${metric('Lineup gain',p.lineup_gain.toFixed(1))}${metric(data.on_clock?'Next turn':'AI selects',pct(data.on_clock?p.next_turn_availability:p.selection_rate))}${metric('Wait cost',p.wait_cost.toFixed(1))}${metric('Available',pct(p.availability_rate))}</div></article>`).join(''):'<div class="empty">No legal recommendations were returned for this draft state.</div>'}
async function runRecommendations(automatic=false){if(busy)return;clearError();setBusy(true,automatic?'A pick changed — refreshing your advice…':'Syncing draft and running simulations…');try{const response=await fetch('/api/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({league:$('league').value,candidates:Number($('candidates').value),runs_per_candidate:Number($('runs').value),workers:'auto'})});if(response.status===401){location='/login';return}const data=await response.json();if(!response.ok)throw new Error(data.detail||'Recommendation failed');render(data)}catch(error){showError(error.message)}finally{setBusy(false)}}
async function poll(){if(busy||!$('watch').checked||!currentSignature)return;try{const response=await fetch('/api/status?league='+encodeURIComponent($('league').value));if(response.status===401){location='/login';return}if(!response.ok)return;const data=await response.json();if(data.signature!==currentSignature)runRecommendations(true)}catch(error){}}
$('run').addEventListener('click',()=>runRecommendations(false));$('league').addEventListener('change',()=>{currentSignature=null;$('leagueTitle').textContent=$('league').selectedOptions[0].textContent;$('results').innerHTML='<div class="empty">Tap “Get recommendations” to load this league.</div>';$('timing').textContent=''});pollTimer=setInterval(poll,2000);
</script></body></html>"""


def main() -> None:
    import uvicorn

    uvicorn.run(
        "sleeper_draft_assistant.web:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
    )


if __name__ == "__main__":
    main()
