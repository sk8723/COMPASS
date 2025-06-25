import numpy as np
from matplotlib import cm

WORD_BANKS = {
    'basic': ['sky', 'building', 'water', 'ground', 'tree', 'boat'],
    'extended': ['water', 'sky', 'ground',
                 'tree', 'bush', 'grass',
                 'rock',
                 'dock', 'boat', 'buoy',
                 'building', 'wall', 'pole', 'car']
}

LABEL_COLOR_MAP = {
    'water': (48,128,240),
    'sky': (135,222,168),
    'ground': (127,57,36),

    'tree': (21,116,57),
    'bush': (72,155,54),
    'grass': (160,187,38),

    'rock': (116,105,125),

    'dock': (95,58,162),
    'boat': (173,97,224),
    'buoy': (224,52,21),

    'building': (235,192,66),
    'wall': (129,118,75),
    'pole': (156,134,94),
    'car': (235,137,50)
}

def word_bank(name):
    return WORD_BANKS.get(name, WORD_BANKS['basic'])

# generate 'n' distinct colors
def get_default_colors(n):
    cmap = cm.get_cmap("tab20", n)
    colors = [tuple((np.array(cmap(i)[:3]) * 255).astype(int)) for i in range(n)]
    return colors

# return a color palette with colors associated with labels from the word bank
def get_label_colors(word_bank):
    default_colors = get_default_colors(len(word_bank))
    palette = []

    for i, label in enumerate(word_bank):
        color = LABEL_COLOR_MAP.get(label, default_colors[i])
        palette.append(color)

    return palette