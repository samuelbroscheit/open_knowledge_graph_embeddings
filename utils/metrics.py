from collections import OrderedDict


class AccumulateMeter(object):
    """Stores the average and current value"""

    def __init__(self,
                 greater_is_better=True,
                 print_precision=4,
                 ):
        self.greater_is_better = greater_is_better
        self.print_precision = print_precision
        self.reset()

    def reset(self):
        self.avg = 0.
        self.val = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.avg = (self.avg * self.count + val * n ) / (self.count + n)
        self.count += n

    def __add__(self, other):
        if other.count > 0:
            self.update(other.avg, other.count)
        return self

    def avg_better_than(self, other):
        if self.greater_is_better:
            return self.avg > other.avg
        else:
            return self.avg < other.avg

    def avg_better_than_float(self, afloat):
        if self.greater_is_better:
            return self.avg > afloat
        else:
            return self.avg < afloat

    def __repr__(self):
        return f"{self.avg:.{self.print_precision}f}"


class MetricResult(OrderedDict):

    def __init__(self):
        super().__init__()
        self.update([("loss", AccumulateMeter(
            greater_is_better=False,
            print_precision=7
        ))])
        self.update([("h1", AccumulateMeter())])
        self.update([("h3", AccumulateMeter())])
        self.update([("h10", AccumulateMeter())])
        self.update([("h50", AccumulateMeter())])
        self.update([("mrr", AccumulateMeter())])
        self.update([("mr", AccumulateMeter())])

    @property
    def metrics(self):
        return list(self.values())

    @property
    def averages(self):
        return "  ".join(
            [f"{k}: {v}" for k,v in self.items()]
        )

    @property
    def averages_dict(self):
        return {f"{k}": v.avg for k,v in self.items()}

    def __add__(self, other):
        for tm, om in zip(self.values(), other.values()):
            if isinstance(om, AccumulateMeter):
                tm += om
        return self

    def reset(self):
        for tm in self.values():
            tm.reset()

    def __repr__(self):
        res = ""
        for k,v in self.items():
            res += f"{k}: {v.avg}\n"
        return res

if __name__ == "__main__":
    m1 = MetricResult()
    m2 = MetricResult()

    m1["loss"].update(10, 5)
    m1["h10"].update(10, 1)
    # m2["h10"].update(2, 1)
    # m2["h10"].update(10, 1)
    # m2["h10"].update(10, 1)

    # m1 = m1 + m2
    # m1 = m1 + m2

    print(m1["h10"].avg_better_than(m2["h10"]) )
