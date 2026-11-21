"""Continuous learning for vega: answer typed questions, take the right answer whenever you know it, and keep
learning from it on your own machine. Apache 2.0.

    from vegaml import Learner                     # or: from vegaml.learn import Learner
    learner = Learner(questions, examples=rows, workdir="./vega-state")
    learner.decide(text)["answers"]["team"]["answer"]
    learner.feedback(text, {"team": "billing"})    # learns every 50 confirmed answers, in the background

A teacher for what vega is unsure of (a person, a rules engine, a large model), learned from as it answers:

    desk = Escalator(learner, teacher=claude_teacher())   # or any function(text, questions) -> answers
    desk.decide(text)["answers"]["team"]["source"]          # "vega", or "teacher" while it is still learning

REST API with JWT: `python -m vegaml.learn serve --questions q.json` (needs `pip install "vegaml[learn]"`).
"""

from .learner import Learner, build, read_bundle, starting_store
from .reader import Reader, label_of
from .teacher import Escalator, claude_teacher

__all__ = [
    "Escalator",
    "Learner",
    "Reader",
    "build",
    "claude_teacher",
    "label_of",
    "read_bundle",
    "starting_store",
]
