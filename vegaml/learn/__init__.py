"""Continuous learning for vega: answer typed questions, take the right answer whenever you know it, and keep
learning from it on your own machine. Apache 2.0.

    from vegaml import Learner                     # or: from vegaml.learn import Learner
    learner = Learner(questions, examples=rows, workdir="./vega-state")
    learner.decide(text)["answers"]["team"]["answer"]
    learner.feedback(text, {"team": "billing"})    # learns every 50 confirmed answers, in the background

REST API with JWT: `python -m vegaml.learn serve --questions q.json` (needs `pip install "vegaml[learn]"`).
"""

from .learner import Learner, build, read_bundle, starting_store
from .reader import Reader, label_of

__all__ = ["Learner", "Reader", "build", "label_of", "read_bundle", "starting_store"]
