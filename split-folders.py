import splitfolders

splitfolders.ratio("label", output="train",
    seed=1337, ratio=(.8, .2), group_prefix=None, group="stem",
    formats=None, move=False, shuffle=True) # default values