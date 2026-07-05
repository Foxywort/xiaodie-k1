# Candidate Data Sources

Use only sources whose license allows training and release use. Keep provenance metadata in JSONL.

## Model Candidates

- `mistralai/Ministral-3-3B-Instruct-2512-BF16`: strict 3B, Apache-2.0, latest target to track.
- `HuggingFaceTB/SmolLM3-3B`: strict 3B, Apache-2.0, stable text-only training target in the current environment.
- `Qwen/Qwen3-4B` or newer Qwen 4B variants: Apache-2.0 and stronger Chinese family, but not strict 3B and heavier for K1.

## Dataset Candidates

- `gofilipa/bedtime_stories`: Apache-2.0, English bedtime stories.
- `garethpaul/children-stories-dataset`: CC-BY-4.0, English children stories.
- `aslicu/fairy_tales`: Unlicense, fairy-tale style data.
- Project Gutenberg public-domain stories: useful for structure and plots, but verify terms and public-domain status per text.

For Chinese production quality, the safest path is to add authorized Chinese story material or write original Chinese training stories, then mix with synthetic instruction variations.
