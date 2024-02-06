from typing import Callable


def linear_schedule(initial_value, final_value, final_progress):
    """
    Creates a function that returns a linearly interpolated value between 'initial_value' and 'final_value'
    up to 'final_progress' fraction of the total training duration, and then 'final_value' onwards.
    
    :param initial_value: The initial learning rate.
    :param final_value: The final learning rate.
    :param final_progress: The fraction of the total timesteps at which the final value is reached.
    :return: A function that takes a progress (0 to 1) and returns the learning rate.
    """
    def func(progress):
        if progress > final_progress:
            return initial_value + (1.0 - progress) * (final_value - initial_value) / final_progress
        else:
            return final_value
    
    return func
