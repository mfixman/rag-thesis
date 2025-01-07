#!/bin/bash

# List of .bib entries and corresponding authors from the provided .bib file.
declare -A AUTHORS=(
    ["factual_recall"]="Qinan Yu et al."
    ["knowledge_grounding_retrieval_augmented"]="Chenxi Whitehouse et al."
    ["how_can_we_know"]="Zhengbao Jiang et al."
    ["attention_is_all_you_need"]="Ashish Vaswani et al."
    ["gpt3"]="Tom B. Brown et al."
    ["rag"]="Patrick Lewis et al."
    ["atlas_foundational"]="Gautier Izacard et al."
    ["ragged"]="Jennifer Hsia et al."
    ["learning_the_difference"]="Divyansh Kaushik et al."
    ["llama"]="Hugo Touvron et al."
    ["flant5"]="Hyung Won Chung et al."
)

# Loop through all .tex files and apply replacements.
for FILE in *.tex; do
    for CITATION in "${!AUTHORS[@]}"; do
        sed -i -E "s/\\\\citeauthor\{$CITATION\}/${AUTHORS[$CITATION]}\\\\cite\{$CITATION\}/g" "$FILE"
    done
done
