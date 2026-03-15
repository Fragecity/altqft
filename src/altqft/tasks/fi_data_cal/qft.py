from dataclasses import dataclass

@dataclass
class FiMetaData:
    pass

@dataclass
class FiData:
    circuit_type: str
    nqubit: int
    nlayer: int
    fi_val: float