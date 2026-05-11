# ProxyWar

ProxyWar is a competitive, execution-based evaluation framework for LLM code generation. Instead of scoring static snippets, it asks each LLM to generate game-playing agents from natural-language specifications, validates them through hierarchical testing with an iterative repair loop, and runs them against one another in automated tournaments. The resulting TrueSkill rankings, combined with Pass@1 and Repair Rate, give a multi-dimensional view of code generation quality that static benchmarks cannot capture.

This repository accompanies the ICSE 2026 paper *ProxyWar: Dynamic Assessment of LLM Code Generation in Game Arenas*.

## Architecture

ProxyWar is organized as five layers (see `main.tex`, Section 3.2):

1. **Game Environment Layer** — A unified game interface $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \mathcal{O}, s_0 \rangle$ over single-player puzzles, two-player board games, and multi-player card games. Located in `ProxyWar/games/`.
2. **Agent Layer** — Minimal contract every generated agent must satisfy: a `select_action` policy `π : O × mask → A`. Defined in `ProxyWar/agents/` and `ProxyWar/games/base.py`.
3. **Code Generation Layer** — Drives `(Model, Prompt(G)) → Code_π` and the repair loop `(Model, Code_π, Errors) → Code_π'`. Implemented in `ProxyWar/coders/` with prompts in `ProxyWar/prompts/`.
4. **Testing Layer** — Hierarchical test suite (Structure / Function / Logic / Robustness) used both as a Pass@1 gate and as feedback for the repair loop. Implemented in `ProxyWar/testers/`.
5. **Tournament Management Layer** — Schedules round-robin and sampled matches, updates TrueSkill ratings (conservative estimate $\mu - 3\sigma$), and emits per-game and aggregate reports. Implemented in `ProxyWar/evaluations/`.

## Games (9)

| Category | Game | Notes |
|---|---|---|
| Single-player puzzle | Sudoku | NP-complete |
| Single-player puzzle | 2048 | 4×4 board, perfect + random |
| Single-player puzzle | Tower of Hanoi | $n$-disk variant |
| Single-player puzzle | Maze | Grid traversal, $W \times H$ |
| Two-player board | Tic-Tac-Toe | Solved, $O(1)$ |
| Two-player board | Connect Four | PSPACE-complete |
| Two-player board | Reversi | PSPACE-complete |
| Two-player spatial | Snake | 2-player, $n \times n$ |
| Multi-player card | Texas Hold'em (Limit) | Imperfect information |

## Coders (18)

Reflects `configs/example.py` and Table 2 of the paper. All coders are accessed through OpenRouter.

**General-purpose**: Qwen2.5-72B, Claude 3.5 Sonnet, Phi-4, Gemini 2.0 Flash, DeepSeek V3 (0324), Llama 4 Maverick, GPT-4.1-Mini, GPT-4.1.

**Reasoning-enhanced**: O3-Mini, Qwen3-235B, DeepSeek-R1 (0528), Claude 4 Sonnet, Gemini 2.5 Flash, Magistral-Small.

**Code-specialized**: Qwen2.5-Coder-32B, Codestral 2501, Mercury-Coder, Codex-Mini.

## Installation

Requires Python 3.10+ and an OpenRouter API key.

```bash
git clone https://github.com/xinke-wang/ProxyWar.git
cd ProxyWar
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...   # Windows PowerShell: $env:OPENROUTER_API_KEY = "sk-or-..."
```

## Running

```bash
python run.py                              # default: configs/minimal.py (tictactoe, gpt4_1 vs gemini_2_5_flash)
python run.py --config configs/example.py  # full multi-game tournament (9 games × 18 coders)
python run.py --vis                        # also dump per-match screenshots
```

A tournament needs at least two coders (round-robin requires ≥ 2 participants), so `minimal.py` ships with two — use it as a smoke test before scaling up to `example.py`.

Reports are written under `experiments/<experiment_name>/`. Each game gets a Markdown report containing TrueSkill rankings, Pass@1, Repair Rate, and per-layer test pass rates.

## Configuration

Configuration files are plain Python modules under `configs/`. Recognized fields:

| Field | Type | Description |
|---|---|---|
| `experiment_name` | `str` | Output directory under `save_path/`. |
| `save_path` | `str` | Root for all generated artifacts. |
| `tournament_rounds` | `int` | Number of independent tournament rounds (each round re-runs the full schedule for variance estimation). |
| `movement_timeout` | `int` | Per-decision wall-clock budget for two/multi-player games, in seconds. Paper default: 45. |
| `single_player_timeout` | `int` | Cumulative wall-clock budget for one episode of a single-player game, in seconds. Paper default: 60. |
| `parallel_execution` | `bool` | Run games in parallel where the schedule allows. |
| `coders` | `list[str]` | Coder ids registered in `CODER_REGISTRY` (see `ProxyWar/coders/`). |
| `games` | `list[str]` | Game ids registered in `GAME_REGISTRY` (see `ProxyWar/games/`). |
| `prompt` | `str` | Prompt template id from `PROMPT_REGISTRY`. Paper uses `"plain"`. |
| `api` | `dict` | API access: `require_key` (bool) and `key_env_var` (env-var name holding the OpenRouter key). |

## Citation

```bibtex
@inproceedings{peng2026proxywar,
  title     = {ProxyWar: Dynamic Assessment of LLM Code Generation in Game Arenas},
  author    = {Peng, Wenjun and Wang, Xinyu and Wu, Qi},
  booktitle = {Proceedings of the 2026 IEEE/ACM 48th International Conference on Software Engineering (ICSE '26)},
  year      = {2026},
  address   = {Rio de Janeiro, Brazil},
  publisher = {ACM},
  doi       = {10.1145/3744916.3773220},
  isbn      = {979-8-4007-2025-3/26/04}
}
```
