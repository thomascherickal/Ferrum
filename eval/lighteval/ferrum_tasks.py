"""Custom LightEval tasks for Ferrum SLMs. Data is built by make_task_data.py
into $FERRUM_TASK_DATA (default eval/lighteval/out/data)."""
import os

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

DATA = os.environ.get("FERRUM_TASK_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "data"))


def heldout_prompt(line, task_name=None):
    return Doc(task_name=task_name, query=line["text"], original_query=line["text"], choices=None, gold_index=None)


def cloze_prompt(line, task_name=None):
    return Doc(task_name=task_name, query=line["question"], choices=[" " + c for c in line["choices"]], gold_index=line["gold"])


def _local(sub):
    return dict(
        hf_repo=os.path.join(DATA, sub), hf_subset="default", hf_avail_splits=["test"],
        evaluation_splits=["test"], few_shots_split=None, few_shots_select=None, generation_size=-1,
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="ferrum_heldout", prompt_function=heldout_prompt,
        metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte], **_local("heldout"),
    ),
    LightevalTaskConfig(
        name="tinystories_cloze", prompt_function=cloze_prompt,
        metrics=[Metrics.loglikelihood_acc], **_local("cloze"),
    ),
]
