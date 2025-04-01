raw-gpt-csv
Edited CHATGPT-03-mini csv outputs from prompts (see paper appendix).

prepare_challenge.ipynb
Notebook to prepare the challenge dataset from Edited CHATGPT-03-mini inputs + manual predicate disambiguation input: simply go the notebook.

test_challenge.ipynb
Notebook to test the challenge dataset on standalone functions

helpers.py
Helper functions for standalone functions.

dataset.json
Challenge dataset.

Organized hierarchically under "capabilities" that represent distinct linguistic phenomena (distance, spatial/temporal distinctions, dative alternations, negation, head noun identification, and predicate disambiguation), the dataset employs four testing formats: MFT (Minimal Functional Test) for single-sentence evaluations, DIR (Directional) for paired sentences with different expected labels, INV (Invariant) for paired sentences with consistent labeling, and specialDIR for specialized predicate positioning tests. Each example contains tokenized sentences with specified predicate and argument indices along with their expected semantic role labels (like ARG0, ARG2, ARGM-TMP, ARGM-LOC), allowing for systematic assessment of a model's understanding of semantic roles across syntactic variations and linguistic challenges.

{
    "capabilities": {
        "spacetemp": {
            "DIR": {
                "1": [
                    { example_1 },
                    { example_2 },
                    ...
                ]
            }
        },
        // other capabilities...
    }
}

example_1:
{
    "tokenized1": ["Person", "verb", "the", "noun", "at", "time", "."],
    "tokenized2": ["Person", "verb", "the", "noun", "in/at", "the", "location", "."],
    
    "predicate_index": 1,  // Index of the predicate (verb) in both sentences
    
    "target_index1": 5/6,  // Index of the temporal marker in first sentence
    "target_index2": 6/7,  // Index of the location marker in second sentence
    
    "target_SRL1": "ARGM-TMP",  // Temporal semantic role label
    "target_SRL2": "ARGM-LOC"   // Location semantic role label
}