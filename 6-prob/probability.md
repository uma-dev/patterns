## Two Dice Probabilities

Two **fair six-sided dice** are rolled. Outcomes are **ordered**: `(die1, die2)`.  
Each elementary outcome has probability `1/36`.

---

### A) Sample Space (Espacio muestral)

The sample space is:

Ω = { (i, j) | i ∈ {1,2,3,4,5,6}, j ∈ {1,2,3,4,5,6} }

Total number of outcomes:

|Ω| = 6 × 6 = 36

---

### B) Probability that die1 = 3 and die2 = 6

There is exactly one favorable outcome: `(3,6)`.

P(die1 = 3 and die2 = 6) = 1 / 36

---

### C) Probability that both dice are even

Even numbers on a die are `{2,4,6}`.

Number of favorable outcomes:
3 × 3 = 9

P(both dice even) = 9 / 36 = 1 / 4

---

### D) Probability that both dice are even, given that the first die is even

Let:

- A = both dice are even
- B = first die is even

Since A ⊆ B:

P(A | B) = P(A) / P(B)

P(A) = 9 / 36  
P(B) = 3 / 6 = 1 / 2

P(A | B) = (9 / 36) ÷ (1 / 2) = 1 / 2

---

### E) Probability that the sum is greater than 5, given that the first die is 3

If die1 = 3, possible values for die2 are `{1,2,3,4,5,6}`.

Sum > 5:

3 + die2 > 5 → die2 > 2

Favorable outcomes: `{3,4,5,6}` → 4 values

P(sum > 5 | die1 = 3) = 4 / 6 = 2 / 3

---

### Final Results

- **A)** |Ω| = 36
- **B)** 1 / 36
- **C)** 1 / 4
- **D)** 1 / 2
- **E)** 2 / 3
