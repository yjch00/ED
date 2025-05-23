import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import os

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput
from transformers import AutoTokenizer
import matplotlib.pyplot as plt




def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
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
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

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
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need
        
        ed_alpha = model_kwargs.get("ed_alpha") if model_kwargs.get("ed_alpha") is not None else 0.5
        ed_beta = model_kwargs.get("ed_beta") if model_kwargs.get("ed_beta") is not None else 0.5
        ed_tau = model_kwargs.get("ed_tau") if model_kwargs.get("ed_beta") is not None else 4
        
        
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            # output_attentions=output_attentions,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
        )
        next_token_logits = outputs.logits[:, -1, :]
        
        # ED attention map
        avg_head_attention = torch.stack(outputs['attentions'])
        avg_head_attention, _ = torch.topk(avg_head_attention, k=3, dim=0)
        avg_head_attention = avg_head_attention.mean(dim=0)[0]
        avg_head_attention, _ = torch.topk(avg_head_attention, k=3, dim=0)
        avg_head_attention = avg_head_attention.mean(dim=0)
        indices = torch.nonzero(outputs['new_image_attention_mask'][0]).squeeze()
        avg_head_attention = avg_head_attention[-1][indices]
        avg_head_attention = avg_head_attention.view(24, 24)

        # ED attention-guided weight
        half_size = 12
        sum1 = torch.sum(avg_head_attention[:half_size, :half_size])
        sum2 = torch.sum(avg_head_attention[:half_size, half_size:])
        sum3 = torch.sum(avg_head_attention[half_size:, :half_size])
        sum4 = torch.sum(avg_head_attention[half_size:, half_size:])
        sums = torch.tensor([sum1, sum2, sum3, sum4])
        def softmax_with_temperature(logits, temperature=10 ** -ed_tau): # tau
            scaled_logits = logits / temperature
            return torch.nn.functional.softmax(scaled_logits.float(), dim=-1)
        prob1, prob2, prob3, prob4 = softmax_with_temperature(sums)


        use_ed = model_kwargs.get("images_ed1") != None
        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )


        if use_ed:
            if 'past_key_values' not in model_kwargs.keys():
                model_kwargs_ed1 = model_kwargs.copy()
                model_kwargs_ed2 = model_kwargs.copy()
                model_kwargs_ed3 = model_kwargs.copy()
                model_kwargs_ed4 = model_kwargs.copy()
                
            # forward pass with sub-image 1
            model_inputs_ed1 = self.prepare_inputs_for_generation_ed1(input_ids, **model_kwargs_ed1) 
            outputs_ed1 = self(
                **model_inputs_ed1,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
            )
            next_token_logits_ed1 = outputs_ed1.logits[:, -1, :]                                      
            
            # forward pass with sub-image 2
            model_inputs_ed2 = self.prepare_inputs_for_generation_ed2(input_ids, **model_kwargs_ed2)  
            outputs_ed2 = self(
                **model_inputs_ed2,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
            )
            next_token_logits_ed2 = outputs_ed2.logits[:, -1, :]                               
            
            # forward pass with sub-image 3
            model_inputs_ed3 = self.prepare_inputs_for_generation_ed3(input_ids, **model_kwargs_ed3)   
            outputs_ed3 = self(
                **model_inputs_ed3,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
            )
            next_token_logits_ed3 = outputs_ed3.logits[:, -1, :]                                      
            
            # forward pass with sub-image 4
            model_inputs_ed4 = self.prepare_inputs_for_generation_ed4(input_ids, **model_kwargs_ed4)  
            outputs_ed4 = self(
                **model_inputs_ed4,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
            )
            next_token_logits_ed4 = outputs_ed4.logits[:, -1, :]                                    
            
            # ED logit ensemble
            combined_probs = (prob1 * next_token_logits_ed1 + 
                            prob2 * next_token_logits_ed2 + 
                            prob3 * next_token_logits_ed3 + 
                            prob4 * next_token_logits_ed4)
            
            # ED adaptive plausibility constraint
            max_combined_prob = combined_probs.max(dim=-1, keepdim=True).values
            ed_cutoff = torch.log(torch.tensor(ed_beta)) + max_combined_prob
            diffs = (1 - ed_alpha) * next_token_logits + ed_alpha * combined_probs
            ed_logits = diffs.masked_fill(combined_probs < ed_cutoff, -float("inf")) 
            
            
            
            ed_logits = logits_processor(input_ids, ed_logits)
            ed_logits = logits_warper(input_ids, ed_logits)
            next_token_scores = ed_logits
            ed_probs = nn.functional.softmax(ed_logits, dim=-1)
            next_tokens = torch.multinomial(ed_probs, num_samples=1).squeeze(1)
            # next_tokens = torch.argmax(ed_probs, dim=-1)            # make greedy
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)


        # # ED visualize attention map
        # fname = os.path.join("PATH", "attn_map.png")
        # plt.imsave(fname=fname, arr=avg_head_attention.cpu().numpy(), format='png')

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


        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        model_kwargs['new_image_attention_mask'] = outputs['new_image_attention_mask']
        
        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        
        
        # ED update model_kwargs for ensemble decoding
        if use_ed:
            model_kwargs_ed1['new_image_attention_mask'] = outputs['new_image_attention_mask']
            model_kwargs_ed2['new_image_attention_mask'] = outputs['new_image_attention_mask']
            model_kwargs_ed3['new_image_attention_mask'] = outputs['new_image_attention_mask']
            model_kwargs_ed4['new_image_attention_mask'] = outputs['new_image_attention_mask']
            

            model_kwargs_ed1 = self._update_model_kwargs_for_generation(
                outputs_ed1, model_kwargs_ed1, is_encoder_decoder=self.config.is_encoder_decoder
            )
            model_kwargs_ed2 = self._update_model_kwargs_for_generation(
                outputs_ed2, model_kwargs_ed2, is_encoder_decoder=self.config.is_encoder_decoder
            )
            model_kwargs_ed3 = self._update_model_kwargs_for_generation(
                outputs_ed3, model_kwargs_ed3, is_encoder_decoder=self.config.is_encoder_decoder
            )
            model_kwargs_ed4 = self._update_model_kwargs_for_generation(
                outputs_ed4, model_kwargs_ed4, is_encoder_decoder=self.config.is_encoder_decoder
            )
            

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()
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

def evolve_ed_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample