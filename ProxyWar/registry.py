from pjtools.registry import Registry

AGENT_REGISTRY = Registry('agent')
CODER_REGISTRY = Registry('coder')
GAME_REGISTRY = Registry('game')
PROMPT_REGISTRY = Registry('prompt')


def list_registered_coders():
    """
    List all registered coders with their model categories.
    
    Returns:
        A formatted string showing all registered coders grouped by category.
    """
    if not hasattr(CODER_REGISTRY, '_modules') or not CODER_REGISTRY._modules:
        return "No coders registered."
    
    # Group coders by category
    categories = {
        'general': [],
        'reasoning': [],
        'code': []
    }
    
    for name, coder_class in CODER_REGISTRY._modules.items():
        try:
            # Create a temporary instance to get the category
            coder = coder_class()
            category = coder.model_category
            model_name = coder.model_name
            
            if category in categories:
                categories[category].append((name, model_name))
            else:
                # Handle unknown categories
                if 'unknown' not in categories:
                    categories['unknown'] = []
                categories['unknown'].append((name, model_name))
                
        except Exception as e:
            # Handle any errors when creating instances
            if 'error' not in categories:
                categories['error'] = []
            categories['error'].append((name, f"Error: {str(e)}"))
    
    # Build the output string
    output = "Registered Coders by Category:\n"
    output += "=" * 50 + "\n\n"
    
    for category, coders in categories.items():
        if coders:
            output += f"{category.upper()} MODELS ({len(coders)}):\n"
            output += "-" * 30 + "\n"
            for name, model_name in sorted(coders):
                output += f"  {name:<25} -> {model_name}\n"
            output += "\n"
    
    return output


def print_registered_coders():
    """Print all registered coders with their categories."""
    print(list_registered_coders()) 