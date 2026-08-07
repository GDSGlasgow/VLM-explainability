# VLM explainability

A repository for code associated with the group's IJDE special issue submission.

## Repository layout

- `src/`: masking, VLM analysis, downloading, and analysis code.
- `scripts/`: command-line and SLURM entry points.
- `config_files/`: model and masking configuration.
- `input_data/`: small input tables and semantic label definitions.
- `prompts/`: prompts supplied to the VLM.
- `notebooks/`: exploratory and reporting notebooks.
- `results/`: generated result tables.
- `plots/`: generated analysis plots.
- `paper/figures/`: figures prepared for the paper.
- `paper/source/`: editable figure source files and exports.
- `examples/`: example masking outputs.
- `assets/fonts/`: fonts used when producing figures.
- `logs/`: scheduler output and error logs.
- `tests/`: automated tests.

Run scripts from the repository root, for example:

```bash
sbatch scripts/create_masks.sh
sbatch scripts/run_analysis.sh
```

## Reliability analysis

The reliability analysis selects two unmasked images from each of five sequence
folders, then sends each selected image to InternVL 100 times. Every raw response
is parsed into score, confidence, and rationale columns and written immediately
to `results/reliability_analysis.csv`, together with its sequence ID, image ID,
repetition number, model, and selection seed. Unparsed text is retained when a
response cannot be parsed completely.

Submit the default HPC job from the repository root:

```bash
sbatch scripts/run_reliability.sh
```

The Python entry point also accepts explicit sequence IDs and alternative run counts:

```bash
python src/reliability_analysis.py \
    --sequence-ids SEQUENCE_1 SEQUENCE_2 SEQUENCE_3 SEQUENCE_4 SEQUENCE_5 \
    --runs 100 \
    --seed 42
```

If a run stops, resume it without repeating successful calls:

```bash
python src/reliability_analysis.py --resume
# or through the job script
sbatch scripts/run_reliability.sh --resume
```

Resume mode validates the model, seed, selected image/run set, and CSV schema.
It backs up the original CSV, keeps only completely successful rows in the
active CSV, and retries failed or missing calls.

Summarise score and confidence variability, test the confidence–consistency
association, and generate the reliability figures with:

```bash
python src/reliability_results.py
```

This writes summary tables to `results/reliability/` and PNG/PDF figures to
`plots/reliability/`. Lexical rationale consistency is measured using TF–IDF
cosine similarity. To additionally compare sentence-embedding similarity with a
same-dimension/different-image baseline, run:

```bash
python src/reliability_results.py \
    --semantic-model sentence-transformers/all-MiniLM-L6-v2
```
