from pathlib import Path
from loguru import logger
import torch
from tqdm import tqdm

import data_utils
import test_bert
import utils

from statistics import mean
from loguru import logger


# Assuming SCORE_METHOD is already defined elsewhere in your code.
class SCORE_METHOD:
    TIME_DIFF = "time_diff"
    COSINE_DIST = "cosine_dist"

def get_sentences_containing_word(text_files, word, max_sentences, ignore_case=True):
    sentences = []
    for file_path in text_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if ignore_case and word.lower() in line.lower() or word in line:
                    sentences.append(line.strip())
                    if len(sentences) >= max_sentences:
                        return sentences
    return sentences


def calc_semantic_shift(tester, sentences, word, score_method):
    if score_method == SCORE_METHOD.TIME_DIFF:
        return calc_change_score_time_diff(tester, sentences, word)
    elif score_method == SCORE_METHOD.COSINE_DIST:
        # Collect embeddings for each sentence containing the word.
        embeddings = []
        for sentence in sentences:
            emb = get_embedding(tester, [sentence], word, hidden_layers_number=1)
            if emb.nelement() > 0:  # Only consider valid embeddings
                embeddings.append(emb)
        
        if len(embeddings) > 1:
            embeddings = torch.stack(embeddings)
            centroid = torch.mean(embeddings, dim=0)
            # Calculate the average cosine distance between each embedding and the centroid
            distances = torch.norm(embeddings - centroid, dim=1)
            average_distance = torch.mean(distances).item()
            return average_distance
        else:
            logger.warning("Not enough embeddings for calculation.")
            return None
    else:
        logger.warning(f"Score method {score_method} not implemented.")
        return None


def calc_change_score_time_diff(tester, sentences, word):
    """
    Calculate the semantic change score as the average absolute distance between predicted probabilities for different times
    for all sentences containing the given word.
    This works if there are only 2 different time points and utilizes the Tester's fill_mask_pipelines attribute.
    """
    if not isinstance(tester.fill_mask_pipelines, list):
        fill_mask_pipelines = list(tester.fill_mask_pipelines)
    else:
        fill_mask_pipelines = tester.fill_mask_pipelines

    time_diffs = []
    for sent in sentences:
        # Embed the time tokens directly in the sentence where the word appears, then mask the word for prediction
        modified_sentence = sent.replace(word, "[MASK]")
        result_dict = test_bert.predict_time(modified_sentence, fill_mask_pipelines, print_results=False)
        

        # Assume time tokens are well defined in the configuration of the first pipeline's model
        time_tokens = [f"<{time}>" for time in fill_mask_pipelines[0].model.config.times]
        if len(time_tokens) < 2:
            logger.error("Not enough time tokens available for calculation.")
            return None
        
        first_time_score = result_dict.get(time_tokens[0], 0)
        print("#"*20)
        print(first_time_score)
        print("#"*20)
        last_time_score = result_dict.get(time_tokens[-1], 0)
        print("#"*20)
        print(last_time_score)
        print("#"*20)

        
        # Calculate the absolute difference between the first and last time point scores
        if first_time_score and last_time_score:  # Ensure both scores are present
            time_diff = abs(last_time_score - first_time_score)
            time_diffs.append(time_diff)

    if not time_diffs:
        logger.warning(f"No time differences computed for {word}. Skipping it.")
        return None

    # Calculate and return the average of the time differences
    average_diff = mean(time_diffs)
    return average_diff


def get_embedding(model, sentences, word, time=None, batch_size=None, require_word_in_vocab=False, hidden_layers_number=None):
    if (require_word_in_vocab and word not in model.tokenizer.vocab) or len(sentences) == 0:
        return torch.tensor([])

    if hidden_layers_number is None:
        num_hidden_layers = model.config.num_hidden_layers
        if num_hidden_layers == 12:
            hidden_layers_number = 1
        elif num_hidden_layers == 2:
            hidden_layers_number = 3
        else:
            hidden_layers_number = 1

    embs = model.embed_word(sentences, word, time=time, batch_size=batch_size, hidden_layers_number=hidden_layers_number)
    if embs.ndim == 1:
        # in case of a single sentence, embs is actually the single embedding, not a list
        return embs
    else:
        centroid = torch.mean(embs, dim=0)
        return centroid



def semantic_change_detection_wrapper(corpus_path, word, score_method, max_sentences):
    corpus_path = Path(corpus_path)
    text_files = list(corpus_path.rglob('*.txt'))
    sentences = get_sentences_containing_word(text_files, word, max_sentences)
    logger.info(f"Found {len(sentences)} sentences containing the word '{word}'.")

    MODEL_PATH = "output/2024-4-6_22-3-40_5_epochs"
    tester = test_bert.Tester(MODEL_PATH, device=0)
    shift_score = calc_semantic_shift(tester, sentences, word, score_method)
    if shift_score is not None:
        logger.info(f"Semantic shift score for '{word}': {shift_score}")
    else:
        logger.warning("No valid data to calculate the shift score.")

if __name__ == "__main__":
    main_corpus_path = "data"
    target_word = "selection"
    chosen_score_method = SCORE_METHOD.COSINE_DIST
    sentence_limit = 200

    semantic_change_detection_wrapper(main_corpus_path, target_word, chosen_score_method, sentence_limit)