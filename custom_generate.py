import warnings

from torch import nn
from transformers import LogitsProcessorList, StoppingCriteriaList
from transformers.generation_stopping_criteria import validate_stopping_criteria
from transformers.generation_utils import GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, \
    SampleEncoderDecoderOutput, SampleDecoderOnlyOutput
from typing import Optional, Callable, List, Iterable, Union

import torch
import torch.distributed as dist


def decrease_temperature_gradually(logits_warper, decrease_factor):
    if decrease_factor:
        assert 0 < decrease_factor < 1, "decrease_factor must be between 0 to 1"
        logits_warper.temperature = max(1, logits_warper.temperature * decrease_factor)
    return logits_warper


@torch.no_grad()
def custom_sample(
        self,
        inputs: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        transform_logits_warper: Optional[Callable] = None,
        **model_kwargs,
) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
    r"""
    Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
    multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

    Apart from `inputs`, all the arguments below will default to the value of the attribute of the same name
    inside the [`PretrainedConfig`] of the model. The default values indicated are the default
    values of those config.

    Most of these parameters are explained in more detail in [this blog post](https://huggingface.co/blog/how-to-generate).

    Parameters:

        inputs (`torch.Tensor` of shape `(batch_size, sequence_length)`, `(batch_size, sequence_length, feature_dim)` or `(batch_size, num_channels, height, width)`, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models
            `inputs` should of in the format of `input_ids`. For encoder-decoder models *inputs* can
            represent any of `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        max_length (`int`, *optional*, defaults to `model.config.max_length`):
            The maximum length of the sequence to be generated.
        max_new_tokens (`int`, *optional*, defaults to None):
            The maximum numbers of tokens to generate, ignore the current number of tokens. Use either
            `max_new_tokens` or `max_length` but not both, they serve the same purpose.
        min_length (`int`, *optional*, defaults to 10):
            The minimum length of the sequence to be generated.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        early_stopping (`bool`, *optional*, defaults to `False`):
            Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
        num_beams (`int`, *optional*, defaults to 1):
            Number of beams for beam search. 1 means no beam search.
        temperature (`float`, *optional*, defaults to 1.0):
            The value used to module the next token probabilities.
        top_k (`int`, *optional*, defaults to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        no_repeat_ngram_size (`int`, *optional*, defaults to 0):
            If set to int > 0, all ngrams of that size can only occur once.
        encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
            If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
            `decoder_input_ids`.
        bad_words_ids(`List[List[int]]`, *optional*):
            List of token ids that are not allowed to be generated. In order to get the tokens of the words that
            should not appear in the generated text, use `tokenizer(bad_word, add_prefix_space=True).input_ids`.
        num_return_sequences(`int`, *optional*, defaults to 1):
            The number of independently computed returned sequences for each element in the batch.
        max_time(`float`, *optional*, defaults to None):
            The maximum amount of time you allow the computation to run for in seconds. generation will still
            finish the current pass after allocated time has been passed.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, 1 for
            tokens that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same
            shape as `input_ids` that masks the pad token. [What are attention masks?](../glossary#attention-mask)
        decoder_start_token_id (`int`, *optional*):
            If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
        use_cache: (`bool`, *optional*, defaults to `True`):
            Whether or not the model should use the past last key/values attentions (if applicable to the model) to
            speed up decoding.
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
            beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        diversity_penalty (`float`, *optional*, defaults to 0.0):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group
            at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
            enabled.
        prefix_allowed_tokens_fn: (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step
            conditioned on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This
            argument is useful for constrained generation conditioned on the prefix, as described in
            [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904).
        logits_processor (`LogitsProcessorList`, *optional*):
             Custom logits processors that complement the default logits processors built from arguments and a
             model's config. If a logit processor is passed that is already created with the arguments or a model's
             config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
             Custom stopping criteria that complement the default stopping criteria built from arguments and a
             model's config. If a stopping criteria is passed that is already created with the arguments or a
             model's config an error is thrown. This feature is intended for advanced users.
        output_attentions (`bool`, *optional*, defaults to *False*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to *False*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to *False*):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to *False*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        forced_bos_token_id (`int`, *optional*):
            The id of the token to force as the first generated token after the `decoder_start_token_id`.
            Useful for multilingual models like [mBART](../model_doc/mbart) where the first generated token
            needs to be the target language token.
        forced_eos_token_id (`int`, *optional*):
            The id of the token to force as the last generated token when `max_length` is reached.
        remove_invalid_values (`bool`, *optional*):
            Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to
            crash. Note that using `remove_invalid_values` can slow down generation.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)

        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If the
            model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific
            kwargs should be prefixed with *decoder_*.

    Return:
        [`~file_utils.ModelOutput`] or `torch.LongTensor`: A
        [`~file_utils.ModelOutput`] (if `return_dict_in_generate=True` or when
        `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the
            possible [`~file_utils.ModelOutput`] types are:

                - [`~generation_utils.GreedySearchDecoderOnlyOutput`],
                - [`~generation_utils.SampleDecoderOnlyOutput`],
                - [`~generation_utils.BeamSearchDecoderOnlyOutput`],
                - [`~generation_utils.BeamSampleDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~file_utils.ModelOutput`] types are:

                - [`~generation_utils.GreedySearchEncoderDecoderOutput`],
                - [`~generation_utils.SampleEncoderDecoderOutput`],
                - [`~generation_utils.BeamSearchEncoderDecoderOutput`],
                - [`~generation_utils.BeamSampleEncoderDecoderOutput`]

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> # do greedy decoding without providing a prompt
    >>> outputs = model.generate(max_length=40)
    >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    >>> document = (
    ... "at least two people were killed in a suspected bomb attack on a passenger bus "
    ... "in the strife-torn southern philippines on monday , the military said."
    ... )
    >>> # encode input context
    >>> input_ids = tokenizer(document, return_tensors="pt").input_ids
    >>> # generate 3 independent sequences using beam search decoding (5 beams)
    >>> # with T5 encoder-decoder model conditioned on short news article.
    >>> outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
    >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> input_context = "The dog"
    >>> # encode input context
    >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
    >>> # generate 3 candidates using sampling
    >>> outputs = model.generate(input_ids=input_ids, max_length=20, num_return_sequences=3, do_sample=True)
    >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

    >>> tokenizer = AutoTokenizer.from_pretrained("ctrl")
    >>> model = AutoModelForCausalLM.from_pretrained("ctrl")
    >>> # "Legal" is one of the control codes for ctrl
    >>> input_context = "Legal My neighbor is"
    >>> # encode input context
    >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
    >>> outputs = model.generate(input_ids=input_ids, max_length=20, repetition_penalty=1.2)
    >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> input_context = "My cute dog"
    >>> # get tokens of words that should not be generated
    >>> bad_words_ids = tokenizer(["idiot", "stupid", "shut up"], add_prefix_space=True).input_ids
    >>> # encode input context
    >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
    >>> # generate sequences without allowing bad_words to be generated
    >>> outputs = model.generate(input_ids=input_ids, max_length=20, do_sample=True, bad_words_ids=bad_words_ids)
    >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    ```"""

    # **** WE ADDED THIS ****
    assert (transform_logits_warper and temperature) or not (transform_logits_warper or temperature), \
        "Must provide both transform_logits_warper and temperature or none of them"
    # 1. Set generation parameters if not already defined
    bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
    num_beams = num_beams if num_beams is not None else self.config.num_beams
    num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
    )

    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    if pad_token_id is None and eos_token_id is not None:
        # special case if pad_token_id is not defined
        # logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        pad_token_id = eos_token_id

    # 2. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
    batch_size = inputs_tensor.shape[0]

    # 3. Define other model kwargs
    model_kwargs["output_attentions"] = output_attentions
    model_kwargs["output_hidden_states"] = output_hidden_states
    model_kwargs["use_cache"] = use_cache

    if model_kwargs.get("attention_mask", None) is None:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, pad_token_id, eos_token_id
        )

    if self.config.is_encoder_decoder:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 4. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids = self._prepare_decoder_input_ids_for_generation(
            batch_size,
            decoder_start_token_id=decoder_start_token_id,
            bos_token_id=bos_token_id,
            model_kwargs=model_kwargs,
        )
    else:
        # if decoder-only then inputs_tensor has to be `input_ids`
        input_ids = inputs_tensor

    # 5. Prepare `max_length` depending on other stopping criteria
    # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
    if max_length is None and max_new_tokens is not None:
        max_length = max_new_tokens + input_ids.shape[-1]
    elif max_length is not None and max_new_tokens is not None:
        # Both are set, this is odd, raise a warning
        warnings.warn(
            "Both `max_length` and `max_new_tokens` have been set "
            f"but they serve the same purpose. `max_length` {max_length} "
            f"will take priority over `max_new_tokens` {max_new_tokens}.",
            UserWarning,
        )
    # default to config if still None
    max_length = max_length if max_length is not None else self.config.max_length

    if input_ids.shape[-1] >= max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # logger.warning(
        #     f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}. "
        #     "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
        # )

    # 6. determine generation mode
    is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1)
    if not is_sample_gen_mode:
        raise ValueError("Custom sampling doesn't support the use of beams")

    # 7. prepare distribution pre_processing samplers
    logits_processor = self._get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        encoder_input_ids=inputs_tensor,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=forced_bos_token_id,
        forced_eos_token_id=forced_eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        remove_invalid_values=remove_invalid_values,
        logits_processor=logits_processor,
    )

    # 8. prepare stopping criteria
    stopping_criteria = self._get_stopping_criteria(
        max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
    )

    # 9. prepare logits warper
    logits_warper = self._get_logits_warper(
        top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
    )

    # 10. expand input_ids with `num_return_sequences` additional sequences per batch
    input_ids, model_kwargs = self._expand_inputs_for_generation(
        input_ids,
        expand_size=num_return_sequences,
        is_encoder_decoder=self.config.is_encoder_decoder,
        **model_kwargs,
    )

    # 11. run sample
    return sample(
        self,
        input_ids,
        logits_processor=logits_processor,
        logits_warper=logits_warper,
        stopping_criteria=stopping_criteria,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        output_scores=output_scores,
        return_dict_in_generate=return_dict_in_generate,
        synced_gpus=synced_gpus,
        transform_logits_warper=transform_logits_warper,
        **model_kwargs,
    )


def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        transform_logits_warper: Optional[Callable] = None,
        **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    r"""
    Generates sequences for models with a language modeling head using multinomial sampling.

    Parameters:

        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from
            [`LogitsProcessor`] used to modify the prediction scores of the language modeling
            head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from
            [`StoppingCriteria`] used to tell if the generation loop should stop.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from
            [`LogitsWarper`] used to warp the prediction score distribution of the language
            modeling head applied before multinomial sampling at each generation step.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of
            generated tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        output_attentions (`bool`, *optional*, defaults to *False*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to *False*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to *False*):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to *False*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If
            model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation_utils.SampleDecoderOnlyOutput`],
        [`~generation_utils.SampleEncoderDecoderOutput`] or obj:*torch.LongTensor*: A
        `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation_utils.SampleDecoderOnlyOutput`] if
        `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
        [`~generation_utils.SampleEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...    AutoTokenizer,
    ...    AutoModelForCausalLM,
    ...    LogitsProcessorList,
    ...    MinLengthLogitsProcessor,
    ...    TopKLogitsWarper,
    ...    TemperatureLogitsWarper,
    ... )

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    >>> model.config.pad_token_id = model.config.eos_token_id

    >>> input_prompt = "Today is a beautiful day, and"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList([
    ...     MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
    ... ])
    >>> # instantiate logits processors
    >>> logits_warper = LogitsProcessorList([
    ...     TopKLogitsWarper(50),
    ...     TemperatureLogitsWarper(0.7),
    ... ])

    >>> outputs = model.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper)

    >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ```"""

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    cur_len = input_ids.shape[-1]

    this_peer_finished = False  # used by synced_gpus only
    # auto-regressive generation
    while True:

        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # **** THIS IS WHAT WE ADDED ****
        if transform_logits_warper:
            logits_warper = transform_logits_warper(logits_warper)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        cur_len = cur_len + 1

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids
