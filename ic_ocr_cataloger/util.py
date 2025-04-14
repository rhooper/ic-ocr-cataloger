import textwrap
from collections import Counter

from ic_ocr_cataloger.catalog import PartInfo


def count_matches(recent):
    return sum(len(f) for f in recent)


def aggregate(frame):
    return Counter({f.part.part_no: v for f, v in frame.items()})


def format_part_info(
    n_occ: int | str,
    part_string: str,
    part: PartInfo,
    wrap_desc_col: int = 55,
    columns: tuple = ("part_no", "n_occ", "pins", "part_string", "description"),
) -> str:
    description = textwrap.wrap(part.description, wrap_desc_col)
    lines = []
    strings = []
    for field in columns:
        n_cols = sum(len(i) + 3 for i in strings) - 3
        match field:
            case "part_no":
                strings += [f"{part.part_no:<14s}"]
            case "n_occ":
                if isinstance(n_occ, str):
                    strings += [f"{n_occ:<2s}"]
                else:
                    strings += [f"{n_occ:>2d}"]
            case "pins":
                if part.pins in (None, -1, 0, "None"):
                    strings += ["      "]
                else:
                    strings += [f"{str(part.pins):>6s}"]
            case "part_string":
                strings += [f"{part_string:>14s}" if part_string is not None else ""]
            case "description":
                for n, line in enumerate(description):
                    field_val = (" " * n_cols + " | " if n > 0 else "") + line
                    if len(field_val) < wrap_desc_col:
                        field_val += " " * (wrap_desc_col - len(field_val))
                    if n > 0:
                        lines += [field_val]
                    else:
                        strings += [field_val]
            case "flags":
                flags = set(part.flags or [])
                flags.discard("")
                flags.discard("None")
                flags.discard(None)  # noqa
                flags.discard("null")
                strings += [",".join([fl.strip() for fl in flags])]
    return "\n".join([(" | ".join(strings)).strip()] + lines)


def evaluate_best_match(recently_found):
    # Find the best frame by number of occurrences of parts
    sortable = [
        (
            aggregate(frame).total(),
            -len(frame),
            tuple(frame.keys()),
            frame,
        )
        for frame in recently_found
    ]
    counts = Counter(frame[2] for frame in sortable)
    ordered = list(sorted(sortable, key=lambda x: x[:2], reverse=True))
    return ordered[0][3], counts[ordered[0][2]]
