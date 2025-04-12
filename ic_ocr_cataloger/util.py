from collections import Counter


def count_matches(recent):
    return sum(len(f) for f in recent)


def aggregate(frame):
    return Counter({f.part.part_no: v for f, v in frame.items()})
