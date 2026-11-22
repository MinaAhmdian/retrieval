from config import Config
from controls.controls_retrieval import NO_DEPTH, ControlsRetrieval, EvidenceUnits
from encoders.retrieval import DOCUMENTED, UNDOCUMENTED, Retrieval
from trained_encoder.runs import RUN_GROUPS, run_names
from trained_encoder.trained_retrieval import TWO_STAGE


def group_dir(config: Config, group: str):
    return config.path(RUN_GROUPS[f"{group}.runs"])


class GroupRetrieval(ControlsRetrieval):

    group = ""
    step = ""

    def __init__(self, config: Config, corpus_name: str):
        from controls.depth_selection import DepthSelection
        self.settings = config.get(f"{self.group}.evaluation")
        if corpus_name not in self.settings["test_corpora"]:
            raise ValueError(f"{corpus_name} is not a test corpus of {self.group}: {self.settings['test_corpora']}")
        EvidenceUnits.__init__(self, config, corpus_name, Retrieval(config, corpus_name),
                               group_dir(config, self.group) / self.settings["evidence_folder"] / corpus_name)
        self.depth = DepthSelection(config).depth()
        self.archives = [DOCUMENTED, UNDOCUMENTED] if self.retrieval.documented else [UNDOCUMENTED]
        self.heads = {}
        self.embedders = {}
        self.cache = {}

    def specs(self) -> list[dict]:
        specs = []
        rules = [(rule, self.depth if rule == TWO_STAGE else NO_DEPTH) for rule in self.settings["rules"]]
        for run in run_names(self.config, f"{self.group}.runs"):
            configuration = run.split("__")[1]
            for query_inputs in self.config.get(f"trained_encoder.test_query_inputs.{configuration}"):
                for archive in self.archives:
                    specs += [dict(self.base(run, self.corpus_name, archive, query_inputs), rule=rule, k=k,
                                   source=self.group) for rule, k in rules]
        return specs


class TransferRetrieval(GroupRetrieval):
    group = "transfer"
    step = "transfer-retrieval"
