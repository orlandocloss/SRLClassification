from spacy.tokens import Doc
import spacy
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer

nlp = spacy.load("en_core_web_sm")
label_all_tokens=True


def tokenize_and_align_labels(examples, tokenizer):
    #manually making tokenized inputs as must use tokenizer on splits
    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []} #this is output seen from using full tokenisations function (token_classification.ipynb)
    
    for i, tokens in enumerate(examples["tokens"]):
        predicate_pos = examples["predicate_pos"][i]
        tokens_before = tokens[:predicate_pos+1]
        tokens_after = tokens[predicate_pos+1:]

        tokenized_before = tokenizer(tokens_before, is_split_into_words=True, add_special_tokens=False)
        tokenized_after = tokenizer(tokens_after, is_split_into_words=True, add_special_tokens=False)

        input_ids = [tokenizer.cls_token_id] + tokenized_before["input_ids"] + [tokenizer.additional_special_tokens_ids[0]] + tokenized_after["input_ids"] + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        
        tokenized_inputs["input_ids"].append(input_ids)
        tokenized_inputs["attention_mask"].append(attention_mask)

        label = examples["SRL_tags"][i]
        word_ids= make_word_ids(tokens_before, tokens_after, predicate_pos, tokenizer)
        previous_word_idx = None
        label_ids=[]
        
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        tokenized_inputs["labels"].append(label_ids)

    return tokenized_inputs

def make_word_ids(tokens_before, tokens_after, predicate_pos, tokenizer):
    word_ids = [None]  # CLS token

    for i, word in enumerate(tokens_before):
        num_subtokens = len(tokenizer(word, add_special_tokens=False)["input_ids"]) #apply tokenizer to one word to see how much it is split
        for _ in range(num_subtokens):
            word_ids.append(i)

    word_ids.append(None) #Pred token

    for i, word in enumerate(tokens_after):
        num_subtokens = len(tokenizer(word, add_special_tokens=False)["input_ids"])
        for _ in range(num_subtokens):
            word_ids.append(i + predicate_pos + 1)

    word_ids.append(None) #sep token
    return word_ids


def get_complex_feature(tokenized_sentence, predicate_position):
    """
    Gets the dependency path from each token to the predicate using LCA (lowest common ancestor) approach.
    Uses ↑ for going up the tree and ↓ for going down.
    """
    doc = nlp(Doc(nlp.vocab, words=tokenized_sentence)) # process the sentence with spaCy
    
    target = doc[predicate_position] # the predicate token we want paths to
    target_lemma = target.lemma_ # get the lemma of the predicate token (used to create feature)
    
    def find_path_to_root(token):  # Function to find path from token to root 
        path = []
        current = token
        while current.head != current: # Continue until we reach the root (a token that is its own head)
            path.append((current, current.dep_))
            current = current.head
        path.append((current, current.dep_))  # Add the root itself
        return path
    
    results = []
    
    for i, token in enumerate(doc): # For each token in the sentence
        if i == predicate_position: # If this is the predicate token itself
            results.append(f"self | {target_lemma}")
            continue
            
        token_to_root = find_path_to_root(token) # find the path from the token to the root
        target_to_root = find_path_to_root(target) # find the path from the predicate to the root
        
        token_indices = [t[0].i for t in token_to_root] # get the indices of the tokens in the path
        target_indices = [t[0].i for t in target_to_root] # get the indices of the predicate tokens in the path
        
        # Find the LCA (first common ancestor in both paths)
        #https://www.geeksforgeeks.org/lowest-common-ancestor-binary-tree-set-1/
        lca_found = False 
        for i, token_idx in enumerate(token_indices):
            if token_idx in target_indices: #ancestor found (could be the root)
                j = target_indices.index(token_idx) #get the index of the ancestor, will be 0 if ancestor is the predicate
                lca_found = True #set the flag to true
                
                path_up = [f"↑{t[1]}" for t in token_to_root[:i]] #path up from token to ancestor, if ancestor is the root then the path is full (up and down)
                
                path_down = [f"↓{t[1]}" for t in reversed(target_to_root[:j])] #path down from ancestor to predicate, if ancestor is the predicate then the path is empty
                
                final_path = "".join(path_up + path_down) #combine the paths
                results.append(f"{final_path} | {target_lemma}") #add the path and predicate lemma to the results
                break
        if not lca_found:
            results.append(f"nopath | {target_lemma}") #if no path found then add no path to the results (shouldnt happen due to root but may need to debug spacy errors)
    
    return results

def get_lemma_feature(tokenized_sentence):
    """
    Creates a feature using just the lemma (base form) of each token.
    """
    doc = nlp(Doc(nlp.vocab, words=tokenized_sentence))  # Process with spaCy to get lemmas
    
    features = []
    for token in doc:
        features.append(token.lemma_)
    
    return features

def get_pos_feature(tokenized_sentence):
    """
    Creates a feature using just the part-of-speech tag for each token.
    """
    doc = nlp(Doc(nlp.vocab, words=tokenized_sentence))
    
    features = []
    for token in doc:
        features.append(token.pos_)
    
    return features

