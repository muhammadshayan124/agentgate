300 replications per row; A/A = unchanged build; the bug is `wrong_refund_amount` at strength 0.35.

| trials per task | 5-pt threshold: fires on unchanged build | agentgate: fires on unchanged build | 5-pt threshold: detects bug | agentgate: detects bug |
|---|---|---|---|---|
| 3 | 14% | 2% | 68% | 22% |
| 5 | 9% | 3% | 72% | 34% |
| 10 | 2% | 3% | 83% | 60% |
| 20 | 0% | 3% | 89% | 85% |
