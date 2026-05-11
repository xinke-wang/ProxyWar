# ProxyWar minimal example configuration.
# Runs a single game (tictactoe) with two coders (smallest tournament
# the framework supports — round-robin requires >= 2 participants).
# See configs/example.py for the full set of available options.

experiment_name = "proxywar_minimal"
save_path = "experiments"

games = ["tictactoe"]
coders = ["gpt4_1", "gemini_2_5_flash"]
prompt = "plain"

api = {
    "require_key": True,
    "key_env_var": "OPENROUTER_API_KEY",
}
