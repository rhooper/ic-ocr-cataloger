import asyncio
import csv
import json
import logging
import re
import sqlite3
from collections import Counter, deque
from functools import total_ordering
from pathlib import Path
from typing import Generator, NamedTuple, Self

from PIL.Image import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PartInfo(NamedTuple):
    part_no: str
    description: str
    pins: int
    flags: set[str]


@total_ordering
class FoundPart(NamedTuple):
    part: PartInfo | None
    chip_label: str
    prefix: str
    family: str
    suffix: str
    manufacturers: list[str] = "Unknown"

    def __hash__(self):
        return hash(repr(self.part))

    def __lt__(self, other: Self):
        if None in (self.part, other.part):
            return False
        return self.part.part_no < other.part.part_no


AMBIGUOUS = dict(zip("U0158OISB", "µOISB0158"))

AMBIGUOUS_RE = re.compile(r"[5S0O1I8B/]")

# Future: https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/


def disambiguate(todo: str, done: str = "") -> Generator[str, None, None]:
    if len(todo) == 0:
        yield done
        return
    doing, remain = todo[0], todo[1:]
    alternate = AMBIGUOUS.get(doing)
    yield from disambiguate(remain, done + doing)
    if alternate:
        yield from disambiguate(remain, done + alternate)


def parse_tsv(line, p):
    rec = line.strip(" \n").split("\t")
    if len(rec) < 4:
        rec.extend([""] * (4 - len(rec)))
    for part_no in expand_names(rec[0]):
        yield PartInfo(
            part_no,
            rec[1],
            rec[3],
            set(rec[2].split(", ") + [p.with_suffix("").name]) - {"", None},
        )


def parse_txt(line, p):
    if not len(line.strip()):
        return
    rec = line.strip().split(" ", 1)
    if len(rec) == 2:
        chip, name = rec
        for part_no in expand_names(chip):
            yield PartInfo(part_no, name, -1, {p.with_suffix("").name})
    else:
        logger.info("Invalid line format: %s %s", line, p)


class Catalog:

    def __init__(self, config):
        self.config = config
        self.prefixes = {}
        self.db = sqlite3.connect(
            str(Path(config["main"].get("catalog_db", "./catalog.db")).expanduser())
        )
        self.data_dir = Path(__file__).parent / "data"
        self.list_dirs = [self.data_dir]
        if config["main"].get("lists_dir"):
            self.list_dirs.append(
                Path(config["main"].get("additional_list_dirs")).expanduser()
            )
        self.create_tables()
        self.load_prefixes()
        self.parts = {}
        self.parts.update(self.load_parts_db())
        self.recent_lookups = deque(maxlen=40)
        self.families: list[dict] = list(
            csv.DictReader((self.data_dir / "families.tsv").open("rt"), delimiter="\t")
        )

    def normalize_7400(
        self,
        src: Generator[PartInfo, None, None],
    ) -> Generator[PartInfo, None, None]:
        family_re = re.compile(
            r"^(A-Z){0,4}(74|54|64)(|"
            + "|".join(family["code"][2:] for family in self.families)
            + r")(\d{2,5})([A-Z]*[0-9]?)$"
        )
        for part in src:
            if matched := family_re.match(part.part_no):
                maker, variant, family, number, suffix = matched.groups()
                # print(maker, variant, family, number, suffix)
                yield part._replace(
                    part_no=f"{74}{number}", flags=part.flags | {family}
                )
            yield part

    def create_tables(self):
        cursor = self.db.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS parts (
            part_no TEXT PRIMARY KEY COLLATE NOCASE,
            pins int,
            description TEXT,
            flags JSON)
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS inventory_entry (
                id INTEGER PRIMARY KEY,
                part_no TEXT,
                qty INTEGER,
                part_label TEXT,
                prefix TEXT,
                family TEXT,
                suffix TEXT,
                date_code TEXT,
                location TEXT,
                manufacturer TEXT,
                date_added  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                FOREIGN KEY(part_no) REFERENCES parts(part_no) ON DELETE no action
                );
                """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS inventory_count (
                part_no TEXT PRIMARY KEY,
                qty INTEGER
            )
            """
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS inventory_log (id INTEGER PRIMARY KEY, inventory_entry int, raw_text TEXT, date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        self.db.commit()

    def add_part_to_db(self, input_part: PartInfo):
        c = self.db.cursor()
        row = c.execute(
            "SELECT part_no, pins, description, flags FROM parts WHERE part_no=?",
            (input_part.part_no,),
        ).fetchone()

        if row:
            existing_part_no, existing_pins, existing_desc, existing_flags = row
            existing_flags = set(json.loads(existing_flags)) - {"", None, "None"}
            tidy_row_pin = (
                existing_pins
                if row[1] not in (None, "", -1, "-1", None, 0, "0")
                else input_part.pins
            )
            tidy_part_pin = (
                input_part.pins
                if input_part.pins not in (None, "", -1, "-1", None, 0, "0")
                else existing_pins
            )
            pins = tidy_part_pin or tidy_row_pin
            in_desc_clean = input_part.description.rstrip(".,.;:'*^~-`\" ")
            description = (
                in_desc_clean
                if len(in_desc_clean) > len(existing_desc) < 30
                else existing_desc
            )
            flags = input_part.flags | existing_flags
            if not any(
                [
                    input_part.pins != pins,
                    description != existing_desc,
                    input_part.flags != flags,
                ]
            ):
                return 0, None
            ret = c.execute(
                "UPDATE parts SET pins=?,description=?,flags=? WHERE part_no=?",
                (pins, description, json.dumps(list(flags)), existing_part_no),
            )
            self.db.commit()
            return -1, ret.lastrowid

        ret = c.execute(
            """
            INSERT INTO parts (part_no, pins, description, flags) VALUES (?, ?, ?, ?) ON CONFLICT(part_no)
            DO NOTHING""",
            (
                input_part.part_no,
                input_part.pins,
                input_part.description,
                json.dumps(list(input_part.flags)),
            ),
        )
        self.db.commit()
        return ret.rowcount, ret.lastrowid

    def add_inventory_item(
        self,
        part: FoundPart,
        qty,
        location=None,
        manufacturer=None,
        date_code=None,
        image_path=None,
        raw_text=None,
    ):
        self.db.execute(
            """
            INSERT INTO inventory_entry (id, part_no, qty, part_label, prefix, family, suffix, date_code, location, manufacturer, image_path)
            VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                part.part.part_no,
                qty,
                part.chip_label,
                part.prefix,
                part.family,
                part.suffix,
                date_code,
                location,
                manufacturer,
                image_path,
            ),
        )
        ret = self.db.execute(
            """INSERT INTO inventory_count (part_no, qty) VALUES (?, ?) ON CONFLICT(part_no) DO UPDATE SET qty=qty+excluded.qty""",
            (part.part.part_no, qty),
        )
        self.db.execute(
            "INSERT INTO inventory_log (inventory_entry, raw_text) VALUES (?, ?)",
            (ret.lastrowid, raw_text),
        )
        self.db.commit()
        return ret.lastrowid

    def find_part_in_catalog(
        self, prefix: str = "", series_in: str = "", family: str = "", number: str = ""
    ) -> PartInfo | None:
        if family is None:
            family = ""
        if number is None:
            number = ""
        if series_in is None:
            series_in = ""
        series = [series_in]
        # if series_in == "74":
        #     series.append("54")
        if series_in == "54":
            series.append("74")
        possibilities = set()
        for series_no in series:
            possibilities |= {
                f"{prefix}{series_no}{family}{number}",
                f"{prefix}{series_no}{number}",
                f"{series_no}{number}",
            }
        for possibility in possibilities:
            self.recent_lookups.append(possibility)
            if part := self.parts.get(possibility):
                if len(part.part_no) < 4:
                    if prefix:
                        continue
                return part

    def lookup_part_from_text(
        self, raw_part_no: str, do_disambiguate=False
    ) -> FoundPart | None:
        raw_part_no = raw_part_no.replace(" ", "").replace("/", "1")
        if len(raw_part_no) < 3:
            return None

        # Parser notes:
        # Prefixes 1-3 characters  | SN SNJ HEF LN SG LM MC (See prefixes.json)
        # 2 char series number     | Logic 54 (mil) 74 (com) 4x
        # Family                   | HC HCT AC ACT F ALS AS LS S ABT
        #

        # Blacklist:
        bl = [
            re.compile(b)
            for b in (
                r"^RCA[A-Z]?.?(\d{3,4})",
                r"QQ[89012]\d{3}",
                "DALLAS",
                "JAPAN",
                "RCA",
                "TI",
                "TEXAS",
                "PHILLIPS",
                "NATIONAL",
                "NEC",
                "TAIWAN",
                "INDONESIA",
            )
        ]
        # for part_no in [disambiguate(raw_part_no.strip()):
        for part_no in disambiguate(raw_part_no) if do_disambiguate else [raw_part_no]:
            for b in bl:
                if b.match(part_no):
                    # logger.debug("Blacklisted part: %s", part_no)
                    return None
            found = re.match(
                r"([A-Z]*)([0-9]+)([A-Z]+)?([0-9]+)?(-[A-Z0-9-]+|[A-Z][A-Z0-9]*)?",
                raw_part_no,
            )
            if not found:
                return None
            prefix, series_no, family, number, suffix = found.groups()

            # Exact match
            if part := self.find_part_in_catalog("", "", "", part_no):
                return FoundPart(
                    part=part,
                    chip_label=found.group(0),
                    prefix="",
                    family="",
                    suffix="",
                    manufacturers=[],
                )

            if suffix and suffix.startswith("I"):
                suffix = suffix[1:]
                number = f"{number}1"

            match found.groups():
                case (
                    "CA",
                    series_no,
                    family,
                    number,
                    suffix,
                ) if part := self.find_part_in_catalog("LM", series_no, family, number):
                    return FoundPart(
                        part=part,
                        chip_label=found.group(0),
                        prefix="LM",
                        family=family,
                        suffix=suffix,
                        manufacturers=[],
                    )
                case (
                    "SN",
                    series_no,
                    family,
                    number,
                    suffix,
                ) if part := self.find_part_in_catalog("", series_no, family, number):
                    return FoundPart(
                        part=part,
                        chip_label=found.group(0),
                        prefix="SN",
                        family=family,
                        suffix=suffix,
                        manufacturers=[],
                    )
                case (_, series_no, *_) if len(series_no) > 2 and re.search(
                    r"[A-Z]+\d+[A-Z]$", part_no
                ):
                    if part := self.find_part_in_catalog(prefix, series_no, "", ""):
                        return FoundPart(
                            part=part,
                            chip_label=found.group(0),
                            prefix=prefix,
                            family=family,
                            suffix=suffix,
                            manufacturers=self.prefixes.get(prefix, []),
                        )
                case _ if part := self.find_part_in_catalog(found.group(0)):
                    return FoundPart(
                        part=part,
                        chip_label=found.group(0),
                        prefix="",
                        family="",
                        suffix="",
                        manufacturers=[],
                    )
                case _ if part := self.find_part_in_catalog(
                    prefix, series_no, family, number
                ):
                    return FoundPart(
                        part=part,
                        chip_label=found.group(0),
                        prefix=prefix,
                        family=family,
                        suffix=suffix,
                        manufacturers=self.prefixes.get(prefix, []),
                    )
                case _:
                    return FoundPart(
                        part=None,
                        chip_label=found.group(0),
                        prefix=prefix,
                        family=family,
                        suffix=suffix,
                        manufacturers=self.prefixes.get(prefix, []),
                    )

    def get_max(self):
        return self.db.execute("SELECT MAX(id) FROM inventory_entry").fetchone()[0] or 0

    def save_part(
        self,
        ui_image: Image,
        orig_image: Image,
        opath: Path,
        parts_found: dict[FoundPart, int],
        raw_text: str,
    ):
        part = list(parts_found.keys())[0]
        n = self.get_max() + 1
        fn = f"{opath}/frame-{n:05d}-{part.chip_label}.jpg"
        ui_image.save(fn, "JPEG", quality=99)
        orig_image.save(fn[:-4] + ".orig.jpg", "JPEG", quality=99)
        for part, qty in parts_found.items():
            self.add_inventory_item(
                part,
                qty,
                image_path=fn,
                raw_text=raw_text,
            )
        return fn

    def load_prefixes(self):
        self.prefixes = json.loads((self.data_dir / "prefixes.json").read_text())

    def load_parts_db(self):
        c = self.db.cursor()
        items = c.execute(
            "SELECT part_no, pins, description, flags FROM parts"
        ).fetchall()
        n = 0
        for n, (part_no, pins, description, flags) in enumerate(items):
            # print(part_no, pins, description, flags)
            if len(part_no.strip()) < 3:
                logging.warn("Invalid part number: %s", part_no)
            info = PartInfo(
                part_no.upper(),
                description,
                pins,
                set(json.loads(flags)) if flags else set(),
            )
            if part_no.startswith("µ"):
                yield part_no.replace("µ", "U").upper(), info._replace(
                    flags=info.flags | {"U=µ"}
                )
            if split_parts := re.match(
                r"([µA-Z]{0,2})(74|54)([A-Z]{1,5})(\d{2,})([A-Z]*)",
                part_no,
                re.IGNORECASE,
            ):
                pre, series, fam, num2, suffix = split_parts.groups()
                yield f"{series}{num2}", info._replace(
                    flags=info.flags | {pre, fam, suffix} - {"", None}
                )
            elif part_suffix := re.match(
                r"([A-Z0-9µ]*[0-9]+[0-9A-Z]*[0-9]+)([A-Z]+|-\S+|[A-Z]{1,5}[0-9]?)",
                part_no,
            ):
                part, suffix = part_suffix.groups()
                yield f"{part}", info._replace(flags=info.flags | {suffix} - {"", None})
            yield part_no.upper().strip(), info
        logger.info("Read %d parts from database", n)

    def search(
        self, search_for: str, limit=100, offset=0
    ) -> Generator[tuple[str, PartInfo], None, None]:
        c = self.db.cursor()
        c.execute(
            "SELECT part_no, pins, description, flags FROM parts WHERE part_no LIKE ? ORDER BY length(part_no)-? LIMIT ? OFFSET ?",
            (search_for, len(search_for.replace("%", "")), limit, offset),
        )
        for part_no, pins, description, flags in c.fetchall():
            yield part_no, PartInfo(
                part_no,
                description,
                pins,
                set(json.loads(flags)) if flags else set(),
            )

    async def reimport(self):
        file_dirs = self.list_dirs
        stats = Counter()

        p: Path
        expanded = []
        for d in file_dirs:
            expanded.extend(d.glob("*.txt"))
            expanded.extend(d.glob("*.tsv"))
        for p in expanded:
            if p.name in ("prefixes.json", "families.tsv"):
                logging.info("Skipping %s", p)
                continue
            logger.info("Parse %s", p)
            try:
                stats["file"] += 1
                stats.update({k: v for k, v in (await self.import_file(p)).items()})
            except Exception as e:
                logging.exception("Error parsing %s: %s", p, e)
                continue
        self.parts = dict(self.load_parts_db())
        return stats

    async def import_file(self, p: Path) -> Counter:
        counter = Counter()
        with p.open("rt") as fp:
            for line in fp:
                line = line.strip(" \n")
                if line[:1] in ("", ";", "#"):
                    continue
                counter.update(["line"])
                if p.suffix == ".tsv":
                    src = parse_tsv(line, p)
                else:
                    src = parse_txt(line, p)
                for part in self.normalize_7400(src):
                    await asyncio.sleep(0)
                    counter.update(["expanded"])
                    if not re.search(
                        r"^[A-Z0-9µ][-A-Z0-9]+", part.part_no, re.IGNORECASE
                    ):
                        logger.info("Ignoring part %s in %s", part.part_no, p)
                        counter.update(["ignored"])
                    if len(part.part_no) >= 3:
                        num, rid = self.add_part_to_db(part)
                        if num > 0:
                            counter.update(["added"])
                            logger.debug("Added - %s: maxid=", (num, rid, part))
                        elif num < 0:
                            counter.update(["updated"])
                            logger.debug("Update - %s: maxid=", (num, rid, part))
        return counter


def expand_names(part_name, no_first=False):
    # Examples:
    # MC54/74HC00A = MC54HC08A MC74HC08A
    # MC4558,AC,C = MC4558 MC4558AC MC4558C
    # MC4741,C = MC4741 MC4741C
    # ICL7104/1CL8068 = ICL7104 ICL8068
    # MC10/100H607 = MC10H607 MC100H607
    # UM8048/35/49/39 = UM8048 UM8035 UM8039 UM8049
    # UM8051/31 = UM8051 UM8031
    # MK4104-3/-33 = MK4104-3 MK4104-33

    if "/" in part_name:
        parts = part_name.split("/")
        if len(parts) > 2:
            yield parts[0]
            for subparts in parts[1:]:
                yield from expand_names(parts[0] + "/" + subparts, no_first=True)
            return
        if len(parts) != 2:
            raise ValueError(f"Invalid part name {part_name}")
        part_a, part_b = parts
        chomp_a = re.match(r"^([A-Z]*)(\d*)(.*)$", part_a)
        chomp_b = re.match(r"^([A-Z]*)(\d*)(.*)$", part_b)
        # print(f"chomp_a: {chomp_a.groups()} chomp_b: {chomp_b.groups()}")
        match list(chomp_a.groups()), list(chomp_b.groups()):
            # Handle /NN(N) format
            case [pre, num, rest], ["", num_b, ""] if len(num_b) < len(num):
                if not no_first:
                    yield part_a
                yield f"{pre}{num[:-len(num_b)]}{num_b}{rest}"
            # For no letters on 2nd part but same number
            case [pre, num, _], ["", num_b, _] if num == num_b:
                if not no_first:
                    yield part_a
                yield f"{pre}{part_b}"
            # For same length
            case _ if len(part_a) == len(part_b):
                if not no_first:
                    yield part_a
                yield part_b
            # For MC54/74HC00A format
            case [pre, num, rest], ["", num_b, rest_b]:
                yield f"{pre}{num_b}{rest_b}"
                yield f"{pre}{num}{rest_b}"
            case _:
                if not no_first:
                    yield part_a
                yield part_b

    elif "," in part_name:
        aliases = part_name.split(",")
        base_name = re.sub(r"[A-Z]+$", "", aliases[0])
        yield aliases[0]
        for alias in aliases[1:]:
            if alias.startswith("-"):
                yield re.sub(r"-\S+$", "", aliases[0]) + alias
            else:
                yield base_name + alias
    else:
        yield part_name
