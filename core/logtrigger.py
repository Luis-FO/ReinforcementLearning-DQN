import math

def logarithmic_trigger(episode_id: int, base_interval: int = 10) -> bool:
    if episode_id < base_interval:
        return True
    
    interval = max(1, int(base_interval * math.log10(episode_id + 1)))


    return episode_id % interval == 0


def segmented_limit_trigger(episode_id: int) -> bool:


    if episode_id < 200:
        return episode_id % 20 == 0
    else:
        return episode_id % 10 == 0