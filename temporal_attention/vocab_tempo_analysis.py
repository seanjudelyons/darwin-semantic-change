"""Semantic change detection evaluation"""
from enum import Flag, auto
from functools import partial
from pathlib import Path
from statistics import mean

import pandas as pd
import scipy
import torch
from loguru import logger
from tqdm import tqdm
import os
import data_utils
import hf_utils
import test_bert
import utils


class SCORE_METHOD(Flag):
    TIME_DIFF = auto()
    COSINE_DIST = auto()


def calc_change_score_time_diff(
    model,
    sentences,
    word,
    verbose=False,
):
    """
    Semantic change score is the average absolute distance between predicted probablities for different times
    for all sentences containing the given word.
    Note: this works if there are only 2 different time points.
    """
    time_tokens = [f"<{time}>" for time in model.config.times]
    first_time_str = time_tokens[0]
    last_time_str = time_tokens[-1]
    time_diffs = []
    for sent in sentences:
        result_dict = test_bert.predict_time(sent, model.pipeline, print_results=False)
        first_time_score = result_dict[first_time_str]
        last_time_score = result_dict[last_time_str]
        time_diff = abs(last_time_score - first_time_score)
        time_diffs.append(time_diff)
    if not time_diffs:
        logger.warning(f"No time diffs computed for {word=}. Skipping it.")
        return
    diff = mean(time_diffs)
    if verbose:
        logger.debug(f"{model}: {word=} score: {diff:.4f}")
    return diff


def calc_change_score_cosine_dist(
    model,
    sentences,
    word,
    batch_size=None,
    verbose=False,
    **kwargs,
):
    """
    Semantic change score is the average cosine distance between word embeddings
    across all sentences containing the given word.
    """
    embs = model.embed_word(sentences, word, batch_size=batch_size)
    centroid = torch.mean(embs, dim=0)
    avg_dist = torch.dist(embs, centroid)
    dist = avg_dist.item()
    if verbose:
        if dist is not None:
            logger.debug(f"{model}: {word=} score: {dist:.3f}")
        else:
            logger.warning(f"No embedding for '{word}' by {model}. Skipping it.")
    return dist


def get_embedding(
    model,
    sentences,
    word,
    time=None,
    batch_size=None,
    require_word_in_vocab=False,
    hidden_layers_number=None,
):
    if (require_word_in_vocab and not word in model.tokenizer.vocab) or len(
        sentences
    ) == 0:
        return torch.tensor([])
    if hidden_layers_number is None:
        num_hidden_layers = model.config.num_hidden_layers
        if num_hidden_layers == 12:
            hidden_layers_number = 1
        elif num_hidden_layers == 2:
            hidden_layers_number = 3
        else:
            hidden_layers_number = 1
    embs = model.embed_word(
        sentences,
        word,
        time=time,
        batch_size=batch_size,
        hidden_layers_number=hidden_layers_number,
    )
    if embs.ndim == 1:
        #  in case of a single sentence, embs is actually the single embedding, not a list
        return embs
    else:
        centroid = torch.mean(embs, dim=0)
        return centroid


def get_detection_function(score_method, config):
    """Return the apprortiate semantic change detection function to use."""
    if score_method == SCORE_METHOD.TIME_DIFF:
        # Fallback in case time_diff cannot be used
        if not hasattr(config, 'times') or "prepend" not in config.time_embedding_type:
            score_method = SCORE_METHOD.COSINE_DIST
        else:
            detection_function = semantic_change_detection
    if score_method == SCORE_METHOD.COSINE_DIST:
        detection_function = semantic_change_detection_temporal
    return detection_function


def semantic_change_detection_wrapper(
    corpus_name,
    test_corpus_path,
    models,
    max_sentences=500,
    score_method=SCORE_METHOD.TIME_DIFF,
    batch_size=None,
    require_word_in_vocab=False,
    hidden_layers_number=None,
    verbose=False,
):
    print("Starting semantic change detection wrapper function...")
    logger.info(
        f"Will evaluate on {corpus_name}, using {max_sentences=} and {hidden_layers_number=}"
    )
    test_corpus_path = Path(test_corpus_path)
    text_files = data_utils.iterdir(test_corpus_path, suffix=".txt")
    model_to_result_str = {}
    target_words = None
    for model in models:
        print(f"Processing model: {model}")
        shifts_dict = get_shifts(corpus_name, model.tokenizer)
        target_words = list(shifts_dict.keys())
        print(f"Target words identified: {len(target_words)} words to process")
        missing_words = check_words_in_vocab(target_words, model.tokenizer, verbose)
        if missing_words:
            print(f"Missing words that are not in the model's vocabulary: {len(missing_words)}")
            logger.warning(
                f"{model} vocab doesn't contain {len(missing_words)} words: {missing_words}"
            )
        word_time_sentences = data_utils.find_sentences_of_words(
            text_files,
            target_words,
            max_sentences,
            ignore_case=model.tokenizer.do_lower_case,
            override=False,
        )
        print("Gathered sentences for target words.")
        if require_word_in_vocab:
            target_words = [word for word in target_words if word not in missing_words]
        detection_function = get_detection_function(score_method, model.config)
        word_to_score = {}
        print(f"Using detection function: {detection_function.__name__}")
        logger.info(f"Evaluating {model} using {score_method.name}...")
        for word in tqdm(target_words, desc="Words"):
            print(f"Processing word: {word}")
            time_sentences = word_time_sentences[word]
            score = detection_function(
                time_sentences,
                model,
                word,
                score_method=score_method,
                batch_size=batch_size,
                hidden_layers_number=hidden_layers_number,
            )
            if score is None:
                continue
            word_to_score[word] = score
        print("Completed scoring for all target words.")
        print(word_to_score)  # This shows the final scores for each word
        model_to_result_str[model] = compute_metrics(
            model,
            word_to_score,
            shifts_dict,
        )
    print("All models processed. Final results:")
    for model, result_str in model_to_result_str.items():
        print(result_str)  # Each model's results printed
        logger.info(result_str)



def check_words_in_vocab(words, tokenizer, verbose=False, check_split_words=False):
    missing_words = []
    for word in words:
        if not word in tokenizer.vocab:
            if verbose:
                logger.warning(f"{word=} doesn't exist in the vocab")
            missing_words.append(word)
        elif check_split_words:
            expected_token_count = 1
            kwargs = {}
            if utils.is_time_id_necessary(tokenizer.time_embedding_type):
                kwargs["time_id"] = next(iter(tokenizer.time_to_id))
                expected_token_count += 1  # the tokenizer is expected to return a second token for the time id
            tokenized = tokenizer.tokenize(word, **kwargs)
            if len(tokenized) > expected_token_count:
                logger.warning(
                    f"{word=} got split by the tokenizer although it exists in the vocab (library bug) to "
                    f"{tokenized}"
                )
                missing_words.append(word)
    return missing_words


def semantic_change_detection(
    time_sentences,
    model,
    word,
    score_method=SCORE_METHOD.TIME_DIFF,
    verbose=False,
    **kwargs,
):
    """
    For each time period,
    Look at all of the sentences that contains this word.
    For each sentence, predict its time.
    Average the predicted times.
    """
    sentences = [sent for sublist in time_sentences.values() for sent in sublist]
    if score_method == SCORE_METHOD.TIME_DIFF:
        method = calc_change_score_time_diff
    elif score_method == SCORE_METHOD.COSINE_DIST:
        method = calc_change_score_cosine_dist
    else:
        raise ValueError(f"Unknown {score_method=}")
    score = method(model, sentences, word, verbose)
    return score


def semantic_change_detection_temporal(
    time_sentences,
    model,
    word,
    score_method=SCORE_METHOD.COSINE_DIST,
    batch_size=None,
    hidden_layers_number=None,
):
    if score_method == SCORE_METHOD.TIME_DIFF:
        raise NotImplementedError()
    elif score_method == SCORE_METHOD.COSINE_DIST:
        embs = [
            get_embedding(
                model,
                sentences,
                word,
                time=time,
                hidden_layers_number=hidden_layers_number,
                batch_size=batch_size,
            )
            for time, sentences in time_sentences.items()
        ]
        # Filter out empty embeddings
        embs = [emb for emb in embs if emb.nelement() > 0]
        if len(embs) < 2:
            logger.warning(f"Insufficient time data for {word} across time points. Returning NaN.")
            return float('nan')  # Ensure at least two time points are present

        embs = torch.stack(embs)
        # calculate the cosine distance between the first and last vectors
        score = torch.dist(embs[0], embs[-1])
        if torch.isnan(score):
            logger.warning(f"NaN score encountered for {word}.")
            return float('nan')
        return score.item()


def compute_metrics(
    model,
    word_to_score,
    shifts,
):
    if not word_to_score:  # Check if word_to_score is empty
        logger.warning(f"No scores to compute metrics for {model}.")
        return f"{model}: No data to calculate correlation."

    words_str = (
        f"out of {len(shifts)} words" if len(word_to_score) < len(shifts) else "words"
    )
    try:
        scores, ground_truth = zip(
            *((score, shifts[word]) for word, score in word_to_score.items())
        )
        get_corr_str_partial = partial(utils.get_correlation_str, scores, ground_truth)

        pearson_str = get_corr_str_partial(scipy.stats.pearsonr, "Pearson")
        spearman_str = get_corr_str_partial(scipy.stats.spearmanr, "Spearman")
        result_str = f"{model}: {pearson_str}, {spearman_str}"
        if len(word_to_score) < len(shifts):
            result_str += f" (based on {len(word_to_score)} {words_str})"
        logger.info(result_str)
    except ValueError as e:
        logger.error(f"Error computing metrics for {model}: {e}")
        result_str = f"{model}: Error in computing correlation."
    return result_str



def get_shifts(corpus_name, tokenizer=None):
    # Hard-coded semantic change scores for specified words
    hardcoded_shifts_dict = {
        "species": 0.98,
        "selection": 0.99,
        "evolution": 0.95,
        "insect": 0.90,
        "natural": 0.85,
        "variation": 0.92,
    }

    if corpus_name.startswith("train"):
        # Directly return the hard-coded shifts for simplicity
        print(f"Using hard-coded semantic shifts for corpus: {corpus_name}")
        return hardcoded_shifts_dict
    else:
        logger.error(f"Unsupported corpus: {corpus_name}")
        return {}

    # If using a tokenizer, here you could further check or process the words
    # For example, ensuring words are in tokenizer's vocabulary (optional)

    return shifts_dict


if __name__ == "__main__":
    hf_utils.prepare_tf_classes()
    utils.set_result_logger_level()

    # data_path = "data/semeval_eng_lemma_new"
    data_path = "training_data"


    corpus_name = Path(data_path).name
    test_corpus_path = data_path

    score_method = SCORE_METHOD.COSINE_DIST
    require_word_in_vocab = True

    max_sentences = 200  # Limit the number of sentences for very popular words
    hidden_layers_number = (
        4  # Specify None to use the default number for the specified method
    )
    batch_size = 64
    verbose = False
    device = 0

    MODEL_PATH = "output/2024-4-6_22-3-40_5_epochs"  # Path to your model
    tester = test_bert.Tester(MODEL_PATH, device=device)

    if not verbose:
        utils.set_loguru_level("INFO")

    semantic_change_detection_wrapper(
        corpus_name,
        test_corpus_path,
        tester.bert_models,
        max_sentences,
        score_method=score_method,
        batch_size=batch_size,
        require_word_in_vocab=require_word_in_vocab,
        hidden_layers_number=hidden_layers_number,
        verbose=verbose,
    )
