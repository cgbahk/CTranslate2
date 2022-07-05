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
import GPUtil
from threading import Thread
import logging

BATCHES_INFO_PATH = Path("/tmp/batches_info.yml")
GPU_DEVICE = 0
DELAY_SEC = 0.1  # second

SOURCE_BATCHES_GENERATOR = {}


def register_src_batches_gen(source_type):

    def decorator(func):

        def wrapper(option):
            return func(option)

        SOURCE_BATCHES_GENERATOR[source_type] = wrapper
        return wrapper

    return decorator


@register_src_batches_gen("from_corpus")
def generate_from_corpus(option):
    corpus_path = Path(option["corpus_path"])
    assert corpus_path.is_file()

    with open(corpus_path) as corpus_file:
        all_src = corpus_file.read().strip().split("\n")

    if option["shuffle"]:
        sents = random.sample(all_src, option["sample"])
    else:
        sents = all_src[:option["sample"]]

    assert sents

    spm_model_path = Path(option["spm_model_path"])
    assert spm_model_path.is_file()

    processor = spm.SentencePieceProcessor(str(spm_model_path))

    batch = []
    for sentence in sents:
        # Here mimics fairseq behavior
        # Ref: https://github.com/pytorch/fairseq/blob/b5a039c292/fairseq/data/encoders/sentencepiece_bpe.py#L47-L52  # noqa: E501
        batch.append(processor.Encode(sentence, out_type=str))

    return [batch]


@register_src_batches_gen("from_batches_info")
def generate_from_batches_info(option):
    spm_model_path = Path(option["spm_model_path"])
    assert spm_model_path.is_file()

    processor = spm.SentencePieceProcessor(str(spm_model_path))

    sample_tokenized_sentence = processor.Encode(option["sample_sentence"], out_type=str)

    batches_info = option["batches_info"]

    batches = []

    assert batches_info["num_batch"] == len(batches_info["batches"])

    for batch_info in batches_info["batches"]:
        assert batch_info["num_example"] == len(batch_info["examples"])

        batch = []
        for sentence_length in batch_info["examples"]:
            assert sentence_length <= len(sample_tokenized_sentence)
            tokenized_sentence = sample_tokenized_sentence[:sentence_length]

            batch.append(tokenized_sentence)

        batches.append(batch)

    return batches


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


# TODO Make as function, to test different `ct2_translator_option`
class TranslateRunner:

    def __init__(self, config):
        self._ct2_translator_option = config["ct2_translator_option"]

    # TODO Take `ct2_translator_option` as well
    def translate(self, *, src_batches, ct2_translate_option):
        response = {}
        response["memory_in_MiB_base"] = get_memory_MiB()

        translator = ct2.Translator(**self._ct2_translator_option)

        response["memory_in_MiB_model_loaded"] = get_memory_MiB()

        monitor = MemFootprintMonitor()

        # TODO Support multi-batch
        assert len(src_batches) == 1

        # TODO Use `with` pattern
        # TODO Use try-except, or translate failure will hang

        monitor.start()
        response["unix_time_in_second_begin"] = time()

        try:
            for src_batch in src_batches:
                results = translator.translate_batch(src_batch, **ct2_translate_option)
        except Exception as err:
            logging.error(err)

            if not monitor.stopped:
                monitor.stop()

            response["status"] = "fail"

        monitor.stop()

        if "status" not in response:
            response["status"] = "success"

        hypotheses = []

        if response["status"] == "success":
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

        # Release resources
        # Ref: https://github.com/OpenNMT/CTranslate2/blob/4b250730e9/docs/memory.md
        del translator

        return response


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

    runner = TranslateRunner(config)

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

        source_type = scenario["source_type"]
        source_type_option = scenario["source_type_option"]

        src_batches = SOURCE_BATCHES_GENERATOR[source_type](source_type_option)

        if source_type == "from_corpus":
            assert len(src_batches) == 1

        if source_type == "from_batches_info":
            assert scenario["ct2_translate_option"].get("max_batch_size", 0) == 0

        assert src_batches

        res = runner.translate(
            src_batches=src_batches,
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
