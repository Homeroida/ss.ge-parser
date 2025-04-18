"""
Utilities module for the real estate price prediction model.
"""

from .helpers import save_model, create_prediction_function, get_example_property

__all__ = [
    'save_model',
    'create_prediction_function',
    'get_example_property'
]