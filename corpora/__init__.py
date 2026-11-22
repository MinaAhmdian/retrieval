from corpora.brax import BraxCorpus
from corpora.chexpert import CheXpertCorpus
from corpora.mimic import MimicCorpus
from corpora.nih import NihCorpus
from corpora.openi import OpenICorpus
from corpora.padchest import PadChestCorpus
from corpora.rexgradient import RexGradientCorpus
from corpora.taix import TaixCorpus
from corpora.vindr import VinDrCorpus

CORPORA = {
    "mimic": MimicCorpus,
    "chexpert": CheXpertCorpus,
    "rexgradient": RexGradientCorpus,
    "openi": OpenICorpus,
    "padchest": PadChestCorpus,
    "brax": BraxCorpus,
    "nih": NihCorpus,
    "taix": TaixCorpus,
    "vindr": VinDrCorpus,
}
