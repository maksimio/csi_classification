from .._timerun import stopwatch, HighLight as HL


class LogCombiner:
    __base_groups = [
        'kitchen', 'room', 'bathroom', 'hall', 'toilet', 'air',
        'bottle', 'thermos', 'grater', 'casserole', 'dish'
    ]
    __hl = HL()


    def __init__(self, groups: list=None) -> None:
        if groups == None:
            self.groups = LogCombiner.__base_groups
            LogCombiner.__hl.hprint(LogCombiner.__hl.WARNING, 'LogCombiner: set base_groups in groups!')
        else:
            self.groups = groups
