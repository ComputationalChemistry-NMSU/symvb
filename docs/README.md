# symvb documentation

| file | what it is |
|---|---|
| [`api.md`](api.md) | reference for the public surface: `Molecule`, determinants, the `System` facade, the operator DSL, Hückel, spin/symmetry projection, MO↔AO tools |
| [`recipes.md`](recipes.md) | task-oriented snippets for the things people most often hand-roll; every snippet is executed and asserted by [`_recipes_check.py`](_recipes_check.py) |
| [`operators_tutorial.md`](operators_tutorial.md) | a guided tour of the second-quantized operator algebra (`symvb.operators`), from `c`/`c†` up to Reynolds projectors |

Full worked derivations live in the teaching notebooks:
[`../notebooks/`](../notebooks) has one notebook per model system of the paper
(H₂, allyl 3c4e, the (H₂)₂⁺ disphenoid, benzene), plus three topical extras in
[`../notebooks/additional/`](../notebooks/additional).

Before editing `recipes.md`, run the checker and keep it in sync:

```bash
PYTHONPATH=. python3 docs/_recipes_check.py
```
