# Licensing

The repository holds two packages under two licences.

| Package | What is in it | Licence |
| --- | --- | --- |
| **vegaml** (`pip install vegaml`) | The decision engine, heads fitted on your labels, calibration, policies, images, audio, video, serving | **Apache 2.0.** Free for any use, including commercial. |
| **vegaml-learn** (`pip install vegaml-learn`) | Continuous learning (`Learner`), landscape learning, the teacher loop (`Escalator`), the REST server, the vega Proxy | **Noncommercial only**, [PolyForm Noncommercial 1.0.0](https://github.com/NandhaKishorM/vegaml/blob/main/learn/LICENSE). |

## Who can use vegaml-learn for free

Personal projects, research, study and experiments, and use by charities, schools, public
research bodies and government.

## Who needs a commercial licence

Any use by or for a business: inside a product or a paid service, in a company's own operations
(support routing, sales, risk, back office), or offered to customers as a hosted service. That
covers `Learner`, `Escalator`, `python -m vegaml_learn proxy` and `laya.Learner`, which loads this
package.

A company may evaluate vegaml-learn for 30 days without a licence, as long as it is not used to
serve customers or run the business.

Terms, and a quote for your company: [embedding.in/licence](https://embedding.in/licence), or
write to [nandakishor@embedding.in](mailto:nandakishor@embedding.in).

## What is not affected

vegaml itself, the vega checkpoints, laya, and the learner that shipped inside vegaml 0.17 to
0.19, which keeps the Apache 2.0 licence it was released under. The hosted vega Proxy at
[platform.embedding.in](https://platform.embedding.in) is a separate paid service and needs no
licence from you.

The full statement is in
[LICENSING.md](https://github.com/NandhaKishorM/vegaml/blob/main/LICENSING.md).
