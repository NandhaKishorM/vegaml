# Licensing

This repository holds two packages under two different licences.

| Path | Package | Licence | File |
|---|---|---|---|
| everything except `learn/` | `vegaml` | Apache License 2.0 | [LICENSE](LICENSE) |
| `learn/` | `vegaml-learn` | PolyForm Noncommercial License 1.0.0 | [LICENSE-NONCOMMERCIAL.md](LICENSE-NONCOMMERCIAL.md) (the same text as [learn/LICENSE](learn/LICENSE)) |

## vegaml: Apache 2.0

The decision engine, heads fitted on your labels, calibration, policies, the image, audio and video
paths and the servers. Free for any use, including commercial use.

`vegaml/learn/` inside the `vegaml` package is a small Apache 2.0 shim. It only forwards to
`vegaml-learn` when that package is installed, and contains none of its code.

## vegaml-learn: noncommercial

Continuous learning (`Learner`), landscape learning, the teacher loop (`Escalator`), the REST server
and the vega Proxy. Free for noncommercial use: personal projects, research, study, and use by
charities, schools, public research bodies and government.

Using vegaml-learn in a company, a product or a paid service needs a commercial licence from
Embedding Space:

- Terms, and a quote for your company: https://embedding.in/licence
- Quotes: nandakishor@embedding.in

A company may evaluate vegaml-learn for 30 days without a licence, as long as it is not used to serve
customers or to run the business.

## Earlier versions

The learner that shipped inside `vegaml` 0.17.0 to 0.19.0 was released under Apache 2.0 and stays
under it. Everything in `learn/` from 0.20.0 on is under the PolyForm Noncommercial License.

## Contributions

A contribution to a path is offered under that path's licence. So that Embedding Space can keep
offering commercial licences for `learn/`, a pull request that touches `learn/` also grants
Embedding Space the right to license that contribution commercially.
