# Generator benchmark comparison

- grammars available via `list_grammars()`: arith, dyck, english, fol, regex, tinypy
- algorithms compared: sequential, sequential_opt
- runs per case: 200
- max_steps: 5000
- bushiness: 0.8
- k: 1
- cases: 6->8, 8->8

| Grammar | Depth case | Baseline ms | Candidate ms | Speedup | Baseline success | Candidate success | Baseline valid | Candidate valid | Mean height (base/cand) |
| ------- | ---------- | ----------- | ------------ | ------- | ---------------- | ----------------- | -------------- | --------------- | ----------------------- |
| arith   | 6->8       | 1.460       | 1.138        | 1.283x  | 100.0%           | 100.0%            | 100.0%         | 100.0%          | 7.225 / 7.225           |
| arith   | 8->8       | 2.191       | 1.557        | 1.407x  | 100.0%           | 100.0%            | 100.0%         | 100.0%          | 8.000 / 8.000           |
| dyck    | 6->8       | 1.046       | 0.748        | 1.397x  | 100.0%           | 100.0%            | 100.0%         | 100.0%          | 7.550 / 7.550           |
| dyck    | 8->8       | 0.881       | 0.660        | 1.334x  | 31.0%            | 31.0%             | 31.0%          | 31.0%           | 8.000 / 8.000           |
| english | 6->8       | 0.668       | 0.557        | 1.198x  | 36.0%            | 36.0%             | 36.0%          | 36.0%           | 7.375 / 7.375           |
| english | 8->8       | 0.551       | 0.441        | 1.249x  | 9.5%             | 9.5%              | 9.5%           | 9.5%            | 8.000 / 8.000           |
| fol     | 6->8       | 2.531       | 2.002        | 1.264x  | 43.0%            | 43.0%             | 43.0%          | 43.0%           | 7.977 / 7.977           |
| fol     | 8->8       | 2.310       | 1.767        | 1.307x  | 29.0%            | 29.0%             | 29.0%          | 29.0%           | 8.000 / 8.000           |
| tinypy  | 6->8       | 0.947       | 0.640        | 1.479x  | 51.0%            | 51.0%             | 51.0%          | 51.0%           | 7.716 / 7.716           |
| tinypy  | 8->8       | 0.409       | 0.327        | 1.253x  | 7.5%             | 7.5%              | 7.5%           | 7.5%            | 8.000 / 8.000           |

## Invalid reason summary

| Grammar | Depth case | Algorithm      | Invalid reasons |
| ------- | ---------- | -------------- | --------------- |
| arith   | 6->8       | sequential     | none            |
| arith   | 6->8       | sequential_opt | none            |
| arith   | 8->8       | sequential     | none            |
| arith   | 8->8       | sequential_opt | none            |
| dyck    | 6->8       | sequential     | none            |
| dyck    | 6->8       | sequential_opt | none            |
| dyck    | 8->8       | sequential     | none:138        |
| dyck    | 8->8       | sequential_opt | none:138        |
| english | 6->8       | sequential     | none:128        |
| english | 6->8       | sequential_opt | none:128        |
| english | 8->8       | sequential     | none:181        |
| english | 8->8       | sequential_opt | none:181        |
| fol     | 6->8       | sequential     | none:114        |
| fol     | 6->8       | sequential_opt | none:114        |
| fol     | 8->8       | sequential     | none:142        |
| fol     | 8->8       | sequential_opt | none:142        |
| tinypy  | 6->8       | sequential     | none:98         |
| tinypy  | 6->8       | sequential_opt | none:98         |
| tinypy  | 8->8       | sequential     | none:185        |
| tinypy  | 8->8       | sequential_opt | none:185        |
