## A. No check: what vega learns (final, mean of 3 seeds)

| task | teacher | vega alone | targeted right | targeted answered as the saboteur wants | bad answers taught | teacher share |
|---|---|---|---|---|---|---|
| massive | honest | 78.2 | 79.0 | 2.3 | 0 | 66.0 |
| massive | target | 73.7 | 0.0 | 84.8 | 83 | 64.6 |
| massive | target_half | 75.1 | 20.8 | 62.4 | 62 | 72.8 |
| massive | target_late | 76.1 | 33.6 | 49.8 | 59 | 76.7 |
| massive | rand5 | 77.2 | 76.0 | 3.9 | 103 | 90.5 |
| massive | rand5_late | 78.6 | 78.0 | 3.1 | 45 | 78.9 |
| cb | honest | 98.3 | 96.6 | 3.4 | 0 | 8.5 |
| cb | target | 88.0 | 0.0 | 100.0 | 45 | 14.1 |
| cb | target_half | 93.3 | 50.4 | 49.6 | 53 | 17.8 |
| cb | target_late | 91.1 | 25.8 | 74.2 | 32 | 11.0 |
| cb | rand5 | 97.9 | 96.4 | 3.6 | 10 | 12.4 |
| cb | rand5_late | 98.2 | 95.5 | 4.5 | 4 | 9.1 |
| saas | honest | 78.7 | 82.0 | 18.0 | 0 | 64.5 |
| saas | target | 77.5 | 70.3 | 29.7 | 52 | 80.4 |
| saas | target_half | 78.4 | 83.0 | 17.0 | 26 | 78.1 |
| saas | target_late | 77.5 | 79.4 | 20.6 | 26 | 66.8 |
| saas | rand5 | 78.5 | 84.1 | 15.9 | 42 | 78.0 |
| saas | rand5_late | 79.0 | 83.8 | 16.2 | 18 | 67.4 |

## B. With the check set (200 records, rate 0.25)

| task | teacher | alarms (of 3) | rule | records after onset until alarm | bad answers taught | left after rollback | targeted right | vega alone | extra teacher calls |
|---|---|---|---|---|---|---|---|---|---|
| massive | honest | 0 | - | - | 0 | 0.0 | 79.0 | 78.2 | 501 |
| massive | noisy | 0 | - | - | 39 | 39.3 | 80.4 | 78.8 | 532 |
| massive | target | 3 | option x3 | 1 | 0 | 0.0 | 49.7 | 64.0 | 200 |
| massive | target_half | 2 | option x2 | 1 | 22 | 22.0 | 39.4 | 67.5 | 314 |
| massive | target_late | 3 | turned x3 | 253 | 18 | 0.0 | 65.8 | 72.7 | 425 |
| massive | rand5 | 0 | - | - | 103 | 103.0 | 76.0 | 77.2 | 621 |
| massive | rand5_late | 2 | changed x2 | 843 | 42 | 17.7 | 72.1 | 75.6 | 548 |
| cb | honest | 0 | - | - | 0 | 0.0 | 96.6 | 98.3 | 245 |
| cb | noisy | 0 | - | - | 6 | 5.3 | 95.5 | 98.0 | 255 |
| cb | target | 3 | option x3 | 1 | 0 | 0.0 | 94.6 | 97.5 | 200 |
| cb | target_half | 0 | - | - | 53 | 53.3 | 50.4 | 93.3 | 285 |
| cb | target_late | 2 | turned x2 | 871 | 30 | 14.3 | 62.7 | 94.5 | 255 |
| cb | rand5 | 0 | - | - | 10 | 9.7 | 96.4 | 97.9 | 264 |
| cb | rand5_late | 0 | - | - | 4 | 3.7 | 95.5 | 98.2 | 248 |
| saas | honest | 0 | - | - | 0 | 0.0 | 82.0 | 78.7 | 348 |
| saas | noisy | 0 | - | - | 17 | 16.7 | 80.3 | 78.4 | 341 |
| saas | target | 0 | - | - | 52 | 52.0 | 70.3 | 77.5 | 377 |
| saas | target_half | 0 | - | - | 26 | 26.0 | 83.0 | 78.4 | 374 |
| saas | target_late | 1 | changed x1 | 370 | 24 | 14.3 | 79.4 | 76.4 | 347 |
| saas | rand5 | 0 | - | - | 42 | 41.7 | 84.1 | 78.5 | 373 |
| saas | rand5_late | 0 | - | - | 18 | 18.0 | 83.8 | 79.0 | 355 |

## C. With a floor three points under the honest teacher's score

| task | teacher | alarms (of 3) | rule | targeted right | vega alone |
|---|---|---|---|---|---|
| massive | honest | 0 | - | 79.0 | 78.2 |
| massive | noisy | 0 | - | 80.4 | 78.8 |
| massive | target | 3 | option x3 | 49.7 | 64.0 |
| massive | target_half | 2 | option x2 | 39.4 | 67.5 |
| massive | rand5 | 2 | floor x2 | 59.1 | 68.6 |
| cb | honest | 0 | - | 96.6 | 98.3 |
| cb | noisy | 0 | - | 95.5 | 98.0 |
| cb | target | 3 | option x3 | 94.6 | 97.5 |
| cb | target_half | 2 | floor x2 | 75.2 | 95.8 |
| cb | rand5 | 2 | floor x2 | 94.6 | 97.7 |
| saas | honest | 0 | - | 82.0 | 78.7 |
| saas | noisy | 0 | - | 80.3 | 78.4 |
| saas | target | 3 | floor x3 | 71.4 | 73.8 |
| saas | target_half | 0 | - | 83.0 | 78.4 |
| saas | rand5 | 3 | floor x3 | 71.4 | 73.8 |

## D. Check rate against delay (teacher turns at the middle of the stream)

| task | teacher | rate | alarms (of 3) | records until alarm | bad taught | left | extra calls |
|---|---|---|---|---|---|---|---|
| massive | target_late | 0.25 | 3 | 253 | 18 | 0.0 | 425 |
| massive | target_late | 0.5 | 3 | 87 | 6 | 0.0 | 594 |
| massive | target_late | 1.0 | 3 | 44 | 2 | 0.0 | 995 |
| massive | rand5_late | 0.25 | 2 | 843 | 42 | 17.7 | 548 |
| massive | rand5_late | 0.5 | 2 | 314 | 27 | 17.3 | 773 |
| massive | rand5_late | 1.0 | 2 | 233 | 24 | 17.3 | 1353 |
| cb | target_late | 0.25 | 2 | 871 | 30 | 14.3 | 255 |
| cb | target_late | 0.5 | 3 | 499 | 16 | 0.0 | 280 |
| cb | target_late | 1.0 | 3 | 242 | 5 | 0.0 | 325 |
| cb | rand5_late | 0.25 | 0 | - | 4 | 3.7 | 248 |
| cb | rand5_late | 0.5 | 0 | - | 4 | 3.7 | 292 |
| cb | rand5_late | 1.0 | 0 | - | 4 | 3.7 | 381 |

## E. False alarms: an honest teacher under the check

| task | teacher | rate | runs | runs with an alarm | checks per run |
|---|---|---|---|---|---|
| cb | honest | 0.25 | 3 | 0 | 245 |
| cb | honest | 1.0 | 10 | 0 | 374 |
| cb | noisy | 0.25 | 3 | 0 | 255 |
| cb | noisy | 1.0 | 10 | 0 | 419 |
| massive | honest | 0.25 | 3 | 0 | 501 |
| massive | honest | 1.0 | 10 | 0 | 1510 |
| massive | noisy | 0.25 | 3 | 0 | 532 |
| massive | noisy | 1.0 | 10 | 1 | 1603 |
| saas | honest | 0.25 | 3 | 0 | 348 |
| saas | noisy | 0.25 | 3 | 0 | 341 |

runs: 198
