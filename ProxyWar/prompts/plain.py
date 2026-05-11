"""
Plain prompt implementation for ProxyWar framework.

This module implements a straightforward prompt generator that creates
clear, structured instructions for LLM agent code generation.
"""

from typing import Dict, Any, Optional
from .base import BasePrompt
from ..games.base import BaseGame
from ..registry import GAME_REGISTRY, PROMPT_REGISTRY


@PROMPT_REGISTRY.register('plain')
class PlainPrompt(BasePrompt):
    """
    Plain prompt generator that creates straightforward, structured prompts.
    
    This prompt style focuses on clarity and completeness, providing
    comprehensive game information and clear code requirements.
    """
    
    def __init__(self):
        """Initialize PlainPrompt."""
        super().__init__("PlainPrompt")
    
    def generate_prompt(self, 
                       game: BaseGame,
                       additional_context: Optional[str] = None,
                       code_requirements: Optional[Dict[str, Any]] = None,
                       agent_name: Optional[str] = None) -> str:
        """
        Generate a plain, comprehensive prompt for agent code generation.
        
        Args:
            game: The game instance containing rules and state information
            additional_context: Optional additional context or constraints
            code_requirements: Optional specific code requirements
            
        Returns:
            Complete formatted prompt string
        """
        # Merge code requirements with defaults
        requirements = self.get_default_code_requirements()
        if code_requirements:
            requirements.update(code_requirements)
        
        # Build the prompt sections
        prompt_sections = []
        
        # 1. Task description
        prompt_sections.append(self._build_task_description())
        
        # 2. Game rules and information
        prompt_sections.append(self._build_game_information(game))
        
        # 3. Code structure requirements
        prompt_sections.append(self._build_code_requirements(requirements))
        
        # 4. Examples and format
        prompt_sections.append(self._build_code_format_example(game.game_name, agent_name))
        
        # 5. Additional context if provided
        if additional_context:
            prompt_sections.append(self._build_additional_context(additional_context))
        
        # 6. Final instructions
        prompt_sections.append(self._build_final_instructions())
        
        return "\n\n".join(prompt_sections)
    
    def _build_task_description(self) -> str:
        """Build the task description section."""
        return """You are an expert Python programmer tasked with creating an OPTIMAL game-playing agent. Your goal is to implement the most intelligent agent possible that can WIN against opponents in the specified game. The agent should use advanced strategies, game theory, and algorithmic techniques to maximize its chances of victory. This is a competitive evaluation - your agent's performance will be measured against other agents, so it must be strategically superior.

IMPORTANT: Output ONLY the Python code file. Do NOT include any explanatory text, descriptions, or comments outside the code. The response should be a complete Python file that can be directly saved and executed."""
    
    def _build_game_information(self, game: BaseGame) -> str:
        """Build the game information section."""
        game_rules = game.get_game_rules()
        obs_format = game.get_observation_format()
        
        return f"""GAME INFORMATION:
{game_rules}

GAME NAME: {game.game_name}

OBSERVATION FORMAT:
{obs_format['description']}

SAMPLE DATA:
- Sample observation: {obs_format['sample_observation']}
- Sample action_mask: {obs_format['sample_action_mask']}
- Action space size: {obs_format['action_space_size']}
- Position mapping: {obs_format['position_mapping']}"""
    
    def _build_code_requirements(self, requirements: Dict[str, Any]) -> str:
        """Build the code requirements section."""
        language = requirements['language'].title()
        
        return f"""CODE REQUIREMENTS:

1. LANGUAGE: Generate a complete, executable {language} file
2. INHERITANCE: Your agent class must inherit from {requirements['base_class']}
3. METHOD IMPLEMENTATION: Override the {requirements['method_name']} method
4. METHOD SIGNATURE: def {requirements['method_name']}(self, {', '.join(requirements['input_params'])}) -> {requirements['return_type']}
5. IMPORTS: Include 'from ProxyWar.agents.base import BaseAgent' at the top
6. STYLE: Write {requirements['style']} code
7. DOCUMENTATION: Include {requirements['comments']} and reasoning as comments

CRITICAL REQUIREMENTS:
- DO NOT define or redefine the BaseAgent class - it already exists and should be imported
- DO NOT include explanatory text outside the code
- The agent class name should match the specific agent being created
- Include only the necessary imports and the agent class implementation

METHOD BEHAVIOR:
- The method receives 'observation' (current game state) and 'action_mask' (legal actions)
- action_mask is a list of booleans where True indicates a legal action at that index
- Return the index of your chosen action (integer) or None if no action is possible
- The observation format depends on the specific game environment

FILE STRUCTURE:
- Include all necessary imports at the top (especially BaseAgent import)
- Define your agent class that inherits from BaseAgent
- The file should be directly executable and ready to use
- NO explanatory text or descriptions outside the code"""
    
    def _build_code_format_example(self, game_name: str, agent_name: Optional[str] = None) -> str:
        """Build the code format example section."""
        # Use agent_name if provided, otherwise fallback to game name
        if agent_name:
            class_name = agent_name
        else:
            # Sanitize game name to create valid Python class name
            sanitized_name = game_name.replace('-', '').replace(' ', '').replace('_', '')
            class_name = f"{sanitized_name}Agent"
        
        return f"""COMPLETE FILE EXAMPLE:

The output should be a complete, executable file like this:

```python
# {class_name}.py - Complete agent implementation
from ProxyWar.agents.base import BaseAgent
from typing import Optional, List, Any
import random  # Example: you can import any libraries you need
# Add other imports as necessary

class {class_name}(BaseAgent):
    '''
    Intelligent agent for {game_name} game.
    
    Strategy: [Describe your overall strategy and reasoning here]
    
    Key considerations:
    - [List your strategic thinking]
    - [Explain your approach]
    '''
    
    def __init__(self, name: str):
        super().__init__(name)
        # Initialize any state variables you need
        # Example: self.previous_moves = []
    
    def select_action(self, observation: Any, action_mask: List[bool]) -> Optional[int]:
        '''
        Select the best action based on current game state.
        
        Args:
            observation: Current game state information
            action_mask: Boolean list indicating legal actions
            
        Returns:
            Index of selected action or None if no legal actions
        '''
        # Step 1: Get legal actions
        legal_actions = [i for i, legal in enumerate(action_mask) if legal]
        
        if not legal_actions:
            return None
        
        # Step 2: Analyze current game state (add your reasoning as comments)
        # Example: Check for immediate winning moves
        # Example: Check for blocking opponent wins
        # Example: Implement strategic positioning
        
        # Step 3: Implement your strategy logic here
        # [Your strategic decision-making code]
        
        # Placeholder: return a strategic choice
        return legal_actions[0]
```

IMPORTANT: The entire output should be a single, complete file that can be saved and executed immediately."""
    
    def _build_additional_context(self, additional_context: str) -> str:
        """Build the additional context section."""
        return f"""ADDITIONAL CONTEXT:
{additional_context}"""
    
    def _build_final_instructions(self) -> str:
        """Build the final instructions section."""
        return """FINAL INSTRUCTIONS:

1. Generate a COMPLETE, EXECUTABLE file that can be saved and run immediately
2. Include ALL necessary imports at the top of the file, especially 'from ProxyWar.agents.base import BaseAgent'
3. You can use ANY external libraries/packages you think will help (numpy, scipy, etc.)
4. Implement an OPTIMAL, STRATEGIC agent class that inherits from BaseAgent  
5. Use any algorithms of any complexity level that you deem most effective
6. Include your thinking process and strategy as detailed comments in the code
7. Handle all edge cases (empty legal actions, invalid states, etc.)
8. Write clean, well-documented code with clear reasoning
9. DO NOT include any explanatory text, descriptions, or comments outside the Python code
10. DO NOT redefine the BaseAgent class - import it instead
11. The code must be syntactically correct and immediately executable
12. Express all your analysis, strategy, and decision-making as comments within the code
13. Output ONLY the Python code without any markdown formatting or explanatory text

Remember: Your response should be a complete Python file that starts with imports and ends with the class definition. No explanatory text before or after the code.""" 