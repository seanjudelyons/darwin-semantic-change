from pathlib import Path
from loguru import logger
from tqdm import tqdm
import torch
import nltk
nltk.download('punkt')

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
    print(time_sentences)
    time_ids = [time_id for time_id, _ in time_sentences.items()]
    embs = [get_embedding(model, sentences, word, time_id, batch_size=batch_size, verbose=verbose)
            for (time_id, sentences) in zip(time_ids, time_sentences.values())]
    embs = [emb for emb in embs if emb.nelement() > 0]
    if not embs:
        return None, None
    embs = torch.stack(embs)
    score = torch.dist(embs[0], embs[-1]).item()
    return score, embs

def find_sentences_of_word(text_files, word, ignore_case=False):
    word_time_sentences = {}
    for file_path in text_files:
        time_id = file_path.stem.split('_')[0]
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            sentences = nltk.sent_tokenize(text)
            for sentence in sentences:
                if ignore_case:
                    if word.lower() in sentence.lower():
                        word_time_sentences.setdefault(time_id, []).append(sentence.strip())
                else:
                    if word in sentence:
                        word_time_sentences.setdefault(time_id, []).append(sentence.strip())
    return word_time_sentences

def compare_embeddings_between_corpora(test_corpus_path1, test_corpus_path2, models, word, score_method, batch_size=None, verbose=False):
    test_corpus_path1 = Path(test_corpus_path1)
    test_corpus_path2 = Path(test_corpus_path2)
    text_files1 = data_utils.iterdir(test_corpus_path1, suffix=".txt")
    text_files2 = data_utils.iterdir(test_corpus_path2, suffix=".txt")

    results = {}
    for model in models:
        logger.info(f"Evaluating model: {model}")
        sentences1 = find_sentences_of_word(text_files1, word, ignore_case=model.tokenizer.do_lower_case)
        sentences2 = find_sentences_of_word(text_files2, word, ignore_case=model.tokenizer.do_lower_case)

        score1, embs1 = semantic_change_detection_temporal(sentences1, model, word, score_method, batch_size=batch_size, verbose=verbose)
        score2, embs2 = semantic_change_detection_temporal(sentences2, model, word, score_method, batch_size=batch_size, verbose=verbose)

        if score1 is not None and score2 is not None:
            distance = torch.dist(embs1, embs2).item()
            logger.info(f"Semantic change score between corpora for '{word}': {distance}")
            results[model] = distance
        else:
            logger.warning(f"Could not compute scores for {word} in one or both corpora.")

    return results

if __name__ == "__main__":
    MODEL_PATH = "output/2024-4-6_22-3-40_5_epochs"
    tester = test_bert.Tester(MODEL_PATH)

    test_corpus_path1 = "data/1845_letters.txt"
    test_corpus_path2 = "data/1860_letters.txt"
    

    score_method = SCORE_METHOD.COSINE_DIST
    batch_size = 64
    verbose = True
    target_word = "evolution"

    compare_embeddings_between_corpora(test_corpus_path1, test_corpus_path2, tester.bert_models, target_word, score_method, batch_size, verbose)
