# ProxyWar example configuration.
# Lists every option understood by ConfigLoader along with the values
# currently registered in GAME_REGISTRY / CODER_REGISTRY / PROMPT_REGISTRY.
# Copy this file and trim it down to build your own experiment config.

# Experiment settings
experiment_name = "proxywar_multi_game_evaluation"
save_path = "experiments"

# Tournament settings
tournament_rounds = 5          # Number of tournament rounds for robustness
movement_timeout = 45          # Seconds per agent move
single_player_timeout = 60     # Cumulative seconds for single-player games
parallel_execution = True      # Run games in parallel

# Games — any subset of the registered game keys:
#   "tictactoe", "connectfour", "reversi", "snake", "hanoi",
#   "sudoku", "maze", "2048", "texas_holdem"
games = [
    "tictactoe", "connectfour", "reversi", "snake",
    "hanoi", "sudoku", "maze", "2048", "texas_holdem",
]

# Coders — any subset of the registered coder keys (see ProxyWar/coders).
coders = [
    "phi4", "gpt4_1_mini", "o3_mini", "qwen_2_5_coder_32b",
    "codestral_2501", "gemini_2_5_flash", "claude_sonnet_4", "gpt4_1",
    "llama_4_maverick", "qwen3_235b_a22b", "deepseek_chat_v3", "deepseek_r1",
    "mistral_magistral_small", "qwen_2_5_72b_instruct", "openai_codex_mini",
    "inception_mercury_coder", "claude_3_5_sonnet", "gemini_2_0_flash_001",
]

# Prompt type registered in PROMPT_REGISTRY.
prompt = "plain"

# API settings — OpenRouter is the default backend for the coders above.
api = {
    "require_key": True,
    "key_env_var": "OPENROUTER_API_KEY",
}
