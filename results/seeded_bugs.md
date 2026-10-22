200 replications, 10 trials per task, 24 tasks. Rates are the share of replications in which the gate fired.

**Unchanged build (A/A): false-positive rate**

| gate | fires on an unchanged build | 95% CI |
|---|---|---|
| naive (any drop) | 43% | 36.3% to 49.9% |
| 5-point threshold | 2% | 1.1% to 5.7% |
| agentgate: quality + cost + critical | 3% | 1.4% to 6.4% |
| agentgate: all four rules | 3% | 1.4% to 6.4% |

**Planted bugs: detection rate**

| bug | strength | mean pass-rate drop | naive | 5-pt threshold | agentgate (3 rules) | agentgate (all 4) | fired via quality / cost / critical / task |
|---|---|---|---|---|---|---|---|
| bad_kb_query | 0.15 | 1.6% | 72% | 12% | 10% | 10% | 10% / 0% / 0% / 1% |
| bad_kb_query | 0.35 | 3.5% | 89% | 32% | 16% | 19% | 16% / 0% / 0% / 5% |
| bad_kb_query | 0.7 | 6.9% | 99% | 72% | 30% | 59% | 30% / 0% / 0% / 44% |
| broken_1_task | 1 task(s) | 4.0% | 92% | 38% | 8% | 100% | 8% / 0% / 0% / 100% |
| broken_3_tasks | 3 task(s) | 11.4% | 100% | 100% | 36% | 100% | 36% / 0% / 0% / 100% |
| hallucinate_status | 0.15 | 3.6% | 88% | 30% | 23% | 23% | 23% / 0% / 0% / 0% |
| hallucinate_status | 0.35 | 8.3% | 100% | 86% | 55% | 56% | 55% / 0% / 0% / 9% |
| hallucinate_status | 0.7 | 16.5% | 100% | 100% | 94% | 100% | 94% / 0% / 0% / 96% |
| missed_escalation | 0.15 | 1.9% | 72% | 12% | 98% | 98% | 8% / 0% / 98% / 0% |
| missed_escalation | 0.35 | 4.3% | 92% | 40% | 100% | 100% | 18% / 0% / 100% / 6% |
| missed_escalation | 0.7 | 8.9% | 100% | 93% | 100% | 100% | 34% / 0% / 100% / 94% |
| no_confirmation_email | 0.15 | 6.7% | 98% | 69% | 56% | 56% | 56% / 0% / 0% / 1% |
| no_confirmation_email | 0.35 | 15.6% | 100% | 100% | 99% | 99% | 99% / 0% / 0% / 25% |
| no_confirmation_email | 0.7 | 30.6% | 100% | 100% | 100% | 100% | 100% / 0% / 0% / 100% |
| redundant_calls | 0.15 | 11.8% | 100% | 99% | 97% | 97% | 96% / 0% / 30% / 2% |
| redundant_calls | 0.35 | 26.8% | 100% | 100% | 100% | 100% | 100% / 0% / 47% / 40% |
| redundant_calls | 0.7 | 54.0% | 100% | 100% | 100% | 100% | 100% / 100% / 74% / 100% |
| refund_forbidden_orders | 0.15 | 3.0% | 86% | 22% | 100% | 100% | 16% / 0% / 100% / 1% |
| refund_forbidden_orders | 0.35 | 7.6% | 100% | 80% | 100% | 100% | 55% / 0% / 100% / 10% |
| refund_forbidden_orders | 0.7 | 15.4% | 100% | 100% | 100% | 100% | 94% / 0% / 100% / 86% |
| runaway_loop | 0.15 | 12.2% | 100% | 98% | 100% | 100% | 96% / 100% / 0% / 2% |
| runaway_loop | 0.35 | 27.7% | 100% | 100% | 100% | 100% | 100% / 100% / 0% / 44% |
| runaway_loop | 0.7 | 55.6% | 100% | 100% | 100% | 100% | 100% / 100% / 0% / 100% |
| skip_policy_check | 0.15 | 6.5% | 97% | 70% | 100% | 100% | 58% / 0% / 100% / 0% |
| skip_policy_check | 0.35 | 15.2% | 100% | 100% | 100% | 100% | 100% / 0% / 100% / 24% |
| skip_policy_check | 0.7 | 30.4% | 100% | 100% | 100% | 100% | 100% / 0% / 100% / 98% |
| wrong_refund_amount | 0.15 | 3.1% | 84% | 25% | 18% | 20% | 18% / 0% / 0% / 2% |
| wrong_refund_amount | 0.35 | 7.8% | 99% | 84% | 59% | 68% | 59% / 0% / 0% / 16% |
| wrong_refund_amount | 0.7 | 15.2% | 100% | 100% | 92% | 100% | 92% / 0% / 0% / 90% |
