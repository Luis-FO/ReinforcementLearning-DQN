import math

def logarithmic_trigger(episode_id: int, base_interval: int = 10) -> bool:
    # Garante que gravamos o episódio 0 e o 1

    if episode_id < base_interval:
        return True
    
    interval = max(1, int(base_interval * math.log10(episode_id + 1)))


    return episode_id % interval == 0


def segmented_limit_trigger(episode_id: int) -> bool:
    """ 
    Define a lógica de gravação decrescente e limitada.
    
    Lógica:
    - Primeiro 10 episódios (0-9): grava todos.
    - Episódios 10-99: grava a cada 5 episódios.
    - Episódios 100 em diante grava a cada 10."""

    if episode_id < 200:
        return episode_id % 20 == 0
    elif episode_id < 500:
        return episode_id % 40 == 0
    else:
        return episode_id % 10 == 0