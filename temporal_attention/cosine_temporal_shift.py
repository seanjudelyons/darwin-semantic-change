from pathlib import Path
from loguru import logger
from tqdm import tqdm
import torch

import data_utils
import test_bert

class SCORE_METHOD:
    COSINE_DIST = 'cosine_dist'

def get_embedding(model, sentences, word, time_ids, batch_size=None, verbose=False):
    embs = model.embed_word(sentences, word, time_ids, batch_size=batch_size)
    centroid = torch.mean(embs, dim=0)
    if verbose:
        logger.info(f"Embeddings for {word} at time {time_ids}: {embs}")
        logger.info(f"Centroid for {word} at time {time_ids}: {centroid}")
    return centroid


def semantic_change_detection_temporal(time_sentences, model, word, score_method, batch_size=None, verbose=False):
    time_ids = [time_id for time_id, _ in time_sentences.items()]
    embs = [get_embedding(model, sentences, word, time_id, batch_size=batch_size, verbose=verbose)
            for (time_id, sentences) in zip(time_ids, time_sentences.values())]
    embs = [emb for emb in embs if emb.nelement() > 0]
    if not embs:
        return None
    embs = torch.stack(embs)
    score = torch.dist(embs[0], embs[-1]).item()
    if verbose:
        logger.info(f"Distance score between first and last embeddings: {score}")
    return score

def find_sentences_of_word(text_files, word, ignore_case=False):
    word_time_sentences = {}
    for file_path in text_files:
        # Extract the year or numeric part only, assuming the format 'YYYY_something.txt'
        time_id = file_path.stem.split('_')[0]  # Adjust this line based on your actual filename format
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if ignore_case:
                    if word.lower() in line.lower():
                        word_time_sentences.setdefault(time_id, []).append(line.strip())
                else:
                    if word in line:
                        word_time_sentences.setdefault(time_id, []).append(line.strip())
    return word_time_sentences

def semantic_change_detection_wrapper(corpus_name, test_corpus_path, models, word, score_method, batch_size=None, verbose=False):
    logger.info(f"Evaluating {corpus_name} for the word '{word}'...")
    test_corpus_path = Path(test_corpus_path)
    text_files = data_utils.iterdir(test_corpus_path, suffix=".txt")

    for model in models:
        word_time_sentences = find_sentences_of_word(text_files, word, ignore_case=model.tokenizer.do_lower_case)
        if not word_time_sentences:
            logger.info("No relevant sentences found for the target word.")
            continue

        logger.debug(f"Processing word: {word} with sentences across times")
        score = semantic_change_detection_temporal(word_time_sentences, model, word, score_method, batch_size=batch_size, verbose=verbose)
        if score is not None:
            logger.info(f"{word}: {score}")
        else:
            logger.debug(f"No score computed for {word}")


if __name__ == "__main__":
    MODEL_PATH = "output/2024-4-6_22-3-40_5_epochs"
    tester = test_bert.Tester(MODEL_PATH)

    corpus_name = "data"
    test_corpus_path = "data/1840_letters.txt"
    score_method = SCORE_METHOD.COSINE_DIST
    batch_size = 64
    verbose = True
    target_word = "evolution"

    semantic_change_detection_wrapper(corpus_name, test_corpus_path, tester.bert_models, target_word, score_method, batch_size, verbose)
