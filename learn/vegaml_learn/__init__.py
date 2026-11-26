"""Continuous learning for vega: answer typed questions, take the right answer whenever you know it, and keep
learning from it on your own machine.

Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE): free for noncommercial use.
Commercial use needs a licence from Embedding Space (https://embedding.in/licence, nandakishor@embedding.in).

    from vegaml_learn import Learner   # `from vegaml import Learner` works too once this is installed
    learner = Learner(questions, examples=rows, workdir="./vega-state")
    learner.decide(text)["answers"]["team"]["answer"]
    learner.feedback(text, {"team": "billing"})    # learns every 50 confirmed answers, in the background

A teacher for what vega is unsure of (a person, a rules engine, a large model), learned from as it answers:

    desk = Escalator(learner, teacher=vllm_teacher())   # or claude_teacher(), or any function
    desk.decide(text)["answers"]["team"]["source"]          # "vega", or "teacher" while it is still learning

Pictures, recordings, clips and conversations are records too (see vegaml_learn.records):

    Learner(questions).decide({"audio": "call.wav"})                         # learns on the transcript
    Learner(questions, modality="image").decide({"text": "Covered?", "image": "parcel.jpg"})

REST API with JWT: `python -m vegaml_learn serve --questions q.json`
(needs `pip install "vegaml-learn[serve]"`).
"""

__version__ = "0.20.1"

from .learner import Learner, build, read_bundle, starting_store
from .reader import Reader, label_of
from .records import Record, as_record, window
from .teacher import Escalator, claude_teacher, vllm_teacher

__all__ = [
    "Escalator",
    "Learner",
    "Reader",
    "Record",
    "as_record",
    "window",
    "build",
    "claude_teacher",
    "vllm_teacher",
    "label_of",
    "read_bundle",
    "starting_store",
]
