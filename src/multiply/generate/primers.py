import numpy as np
from dataclasses import dataclass, field
from multiply.generate.targets import Target


@dataclass
class Primer:
    seq: str
    direction: str
    start: int
    # end: int
    length: int
    tm: float
    gc: float
    target: Target = None


@dataclass
class PrimerPair:
    """
    Define a pair of primers

    TODO:
    - A nice __repr__?
    - A joint name field?
    - Consider how to implement uniqueness; does target need to be defined?

    """

    F: Primer
    R: Primer
    product_bp: int
    pair_penalty: float
    pair_id: str = field(default="", repr=False)
    target: Target = None

    def __post_init__(self):
        """
        There might be a better way to do this; do I need target
        to ensure uniqueness?

        What about identical primers that anneal to different places?

        """

        F_info = f"{self.F.seq}-{self.F.start}-{self.F.length}"
        R_info = f"{self.R.seq}-{self.R.start}-{self.R.length}"
        self.pair_id = f"{F_info}+{R_info}"

    def get_primer_as_dict(self, direction):
        """
        Get either the forward or reverse primer, as a dictionary
        
        """
        if direction == "F":
            primer_info = self.F.__dict__.copy()
        elif direction == "R":
            primer_info = self.R.__dict__.copy()
        else:
            raise ValueError("Primer direction must be in ['F', 'R'].")

        primer_info.update({
            "product_bp": self.product_bp,
            "pair_penalty": self.pair_penalty,
        })
        
        return primer_info

    # Allow set(), specifically on self.pair_id
    def __hash__(self):
        return hash(self.pair_id)

    def __eq__(self, other):
        if not isinstance(other, PrimerPair):
            return NotImplemented
        return self.pair_id == other.pair_id


def load_primer_pairs_from_primer3_output(primer3_output_path):
    """
    Given an output file from primer3, return a list of
    PrimerPair objects

    """

    # Parse primer3 output
    with open(primer3_output_path, "r") as f:
        # Iterate until determine number of primers returned
        for line in f:
            if line.startswith("PRIMER_PAIR_NUM_RETURNED"):
                n_returned = int(line.strip().split("=")[1])
                break

        # Return an empty list if no primers discovered
        if n_returned == 0:
            return []

        # Store remaining lines in a dictionary
        primer3_dt = {k: v.strip() for k, v in [l.split("=") for l in f]}

    # Define indexes and directions for primer pairs returned
    ixs = np.arange(n_returned)
    directions = ["LEFT", "RIGHT"]

    # Iterate over pairs
    primer_pairs = []
    for ix in ixs:

        # Get information about individual primers
        pair = {}
        for d in directions:

            primer_name = f"PRIMER_{d}_{ix}"
            s, l = primer3_dt[primer_name].split(",")

            pair[d] = Primer(
                seq=primer3_dt[f"{primer_name}_SEQUENCE"],
                direction="F" if d == "LEFT" else "R",
                start=int(s),
                length=int(l),
                tm=float(primer3_dt[f"{primer_name}_TM"]),
                gc=float(primer3_dt[f"{primer_name}_GC_PERCENT"]),
            )

        # Get information about pair
        pair_name = f"PRIMER_PAIR_{ix}"
        primer_pair = PrimerPair(
            F=pair["LEFT"],
            R=pair["RIGHT"],
            product_bp=int(primer3_dt[f"{pair_name}_PRODUCT_SIZE"]),
            pair_penalty=float(primer3_dt[f"{pair_name}_PENALTY"]),
        )

        # Store
        primer_pairs.append(primer_pair)

    return primer_pairs
