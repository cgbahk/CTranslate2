# TODO
# - Sentence from actual corpus vs. dummy sentence with same length
# - Measure memory: unloaded / model loaded / runtime max footprint

from pathlib import Path
import yaml
import sentencepiece as spm
import ctranslate2 as ct2
from time import time, sleep
import argparse
from tqdm import tqdm
import random
import re
import GPUtil
from threading import Thread

BATCHES_INFO_PATH = Path("/tmp/batches_info.yml")
GPU_DEVICE = 0
DELAY_SEC = 0.1  # second

SOURCE_SENTENCE_GENERATOR = {}


def get_memory_MiB():
    return GPUtil.getGPUs()[GPU_DEVICE].memoryUsed


class MemFootprintMonitor(Thread):
    """
    Ref:
    https://github.com/OpenNMT/CTranslate2/blob/130d203307/tools/benchmark/benchmark.py#L47
    https://github.com/anderskm/gputil/tree/42ef071dfc#monitor-gpu-in-a-separate-thread
    """

    def __init__(self):
        super(MemFootprintMonitor, self).__init__()

        self.stopped = False
        self._max_mem_in_MiB = 0

    def run(self):
        while not self.stopped:
            cur_mem = get_memory_MiB()
            if self._max_mem_in_MiB < cur_mem:
                self._max_mem_in_MiB = cur_mem
            sleep(DELAY_SEC)

    def stop(self):
        self.stopped = True

    @property
    def max_memory_in_MiB(self):
        return self._max_mem_in_MiB


class SentenceTranslator:

    def __init__(self, config):
        spm_model_path = Path(config["spm_model_path"])
        assert spm_model_path.is_file()

        self.mem_base = get_memory_MiB()
        self._processor = spm.SentencePieceProcessor(str(spm_model_path))
        self._translator = ct2.Translator(**config["ct2_translator_option"])

    def translate(self, *, src_sents, ct2_translate_option):
        response = {}
        response["memory_in_MiB_base"] = self.mem_base  # TODO Where would be correct place?
        response["memory_in_MiB_model_loaded"] = get_memory_MiB()
        response["unix_time_in_second_begin"] = time()

        # TODO How to add documentation for this API
        src_batch = []
        for src_sent in src_sents:
            # Here mimics fairseq behavior
            # Ref: https://github.com/pytorch/fairseq/blob/b5a039c292/fairseq/data/encoders/sentencepiece_bpe.py#L47-L52  # noqa: E501
            src_batch.append(self._processor.Encode(src_sent, out_type=str))

        monitor = MemFootprintMonitor()

        # TODO Use `with` pattern
        # TODO Use try-except, or translate failure will hang

        monitor.start()
        results = self._translator.translate_batch(src_batch, **ct2_translate_option)
        monitor.stop()

        hypotheses = []
        for result in results:
            # This only holds for `num_hypotheses == 1`
            # TODO Support multi hypotheses case
            assert len(result.hypotheses) == 1
            hyp = result.hypotheses[0]

            # Here mimics fairseq behavior
            # Ref: https://github.com/pytorch/fairseq/blob/b5a039c292/fairseq/data/encoders/sentencepiece_bpe.py#L54  # noqa: E501
            detokenized_hyp = "".join(hyp).replace(" ", "").replace("\u2581", " ").strip()

            hypotheses.append(detokenized_hyp)

        response["unix_time_in_second_end"] = time()
        response["ct2_version"] = ct2.__version__
        # TODO Can we get exclusive list of translate option in CT2?
        response["ct2_translate_option"] = ct2_translate_option
        response["hypotheses"] = hypotheses
        response["memory_in_MiB_peak"] = monitor.max_memory_in_MiB
        return response


def make_sentences(*, corpus_path: str, sample: int, shuffle: bool):
    corpus_path = Path(corpus_path)
    assert corpus_path.is_file()

    with open(corpus_path) as corpus_file:
        all_src = corpus_file.read().strip().split("\n")

    if shuffle:
        return random.sample(all_src, sample)
    else:
        return all_src[:sample]


def parse_batches_info_from_raw(lines):
    ret = {
        "count": len(lines),
        "num_examples": [],
        "max": [],
        "sum": [],
    }

    for line in lines:
        match = re.search("num_examples:([0-9]*),max:([0-9]*),sum:([0-9]*),", line)

        assert match

        ret["num_examples"].append(int(match.group(1)))
        ret["max"].append(int(match.group(2)))
        ret["sum"].append(int(match.group(3)))

    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--translator-config", help="[IN] Path to translator configuration yaml")
    parser.add_argument("--base", help="[IN] Directory to base")
    parser.add_argument(
        "--scenario", default="scenario.yml", help="[IN] Relative path from base to scenario yaml"
    )
    parser.add_argument(
        "--workspace", default="WS", help="[OUT] Relative path from base to workspace"
    )

    args = parser.parse_args()

    config_yml_path = Path(args.translator_config)
    assert config_yml_path.is_file()

    with open(config_yml_path) as config_yml_file:
        config = yaml.safe_load(config_yml_file)

    translator = SentenceTranslator(config)

    base_dir = Path(args.base)
    assert base_dir.is_dir()

    scen_yml_path = base_dir / args.scenario
    assert scen_yml_path.is_file()

    with open(scen_yml_path) as scen_yml_file:
        scenarios = yaml.safe_load(scen_yml_file)

    workspace_dir = base_dir / args.workspace
    workspace_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(scenarios)
    for scenario in progress:
        progress.set_description(scenario["name"])

        src_sents = None

        if scenario["source_type"] == "from_corpus":
            src_sents = make_sentences(
                corpus_path=scenario["from_corpus_option"]["corpus_path"],
                sample=scenario["from_corpus_option"]["sample"],
                shuffle=scenario["from_corpus_option"]["shuffle"],
            )

        assert src_sents

        res = translator.translate(
            src_sents=src_sents,
            ct2_translate_option=scenario["ct2_translate_option"],
        )

        scen_workspace_dir = workspace_dir / scenario["name"]
        scen_workspace_dir.mkdir(parents=True, exist_ok=True)

        assert BATCHES_INFO_PATH.is_file()
        BATCHES_INFO_PATH.rename(scen_workspace_dir / BATCHES_INFO_PATH.name)

        hypotheses_path = scen_workspace_dir / "hypotheses.txt"
        with open(hypotheses_path, mode="w") as hypotheses_file:
            hypotheses = res.pop("hypotheses")
            hypotheses_file.write("\n".join(hypotheses))

        manifest_yml_path = scen_workspace_dir / "manifest.yml"
        with open(manifest_yml_path, mode="w") as manifest_yml_file:
            yaml.safe_dump(res, manifest_yml_file, sort_keys=False)


if __name__ == "__main__":
    main()
