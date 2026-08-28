# The Elite Upgrade: BlindMind V2

To push this application to its absolute limit as a research tool for combinatorial innovation, we will implement advanced biological and AI mechanics.

## 1. Advanced Biological Mechanics (Tournament Selection)
Currently, parents are selected purely at random. Biological evolution favors fitness. 
- **Implementation:** We will add `get_tournament_concepts` to `src/db.py`. This algorithm selects a random subset (e.g., 5 concepts) and then deterministically chooses the parents with the highest `fitness_score`. This ensures the strongest ideas compound.

## 2. Semantic Diversity (The "Jump" Metric)
To prevent the algorithm from producing "boring" incremental variations, we will introduce a new scoring vector to the AI Critic.
- **Implementation:** Update `CriticScore` in `src/llm_schemas.py` to include `semantic_jump` (1-10). The critic will penalize mutations that are too similar to their parents. The overall survival threshold will require a high semantic jump, ensuring true lateral shifts.

## 3. Adaptive Pressure (Meta-Evolution)
Evolution reacts to its environment. We will simulate this by having the AI Critic generate an `evolutionary_directive` (e.g., "Shift focus away from purely digital solutions towards hardware integration").
- **Implementation:** The `EvolutionRun` will store the latest directive, and the `EvolutionEngine` will inject this directive into the `CROSSOVER` and `POINT_MUTATION` prompts for the *next* generation, creating a self-steering feedback loop.

## 4. Professional Network Visualization (Graphviz)
The terminal tree is nice, but true research requires network analysis.
- **Implementation:** Add a `graph` command to `src/cli.py` that crawls the `Lineage` table and exports the entire evolutionary history as a `.dot` file. This allows researchers to render massive, complex directed acyclic graphs (DAGs) of their latent space using Graphviz or Mermaid.

## Execution Steps:
1. **Update Schemas & Models:** Add `semantic_jump`, `evolutionary_directive`, and update `EvolutionRun`.
2. **Refine Prompts:** Inject the directive into the Variation Prompts; instruct the Critic to measure the Jump.
3. **Upgrade Engine & DB:** Implement Tournament Selection and route the directives through the generation cycles.
4. **Implement CLI Tooling:** Build the `graph` command for `.dot` export and update the Human-in-the-Loop UI to show the new Elite metrics.

This turns the tool from a neat script into a sophisticated, self-steering innovation engine. Shall we proceed with this Elite Upgrade?