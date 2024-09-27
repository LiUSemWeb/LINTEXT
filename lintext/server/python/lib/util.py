import os.path
from .document import Document
from .fitbert import FitBert

import torch
import pickle
from collections import defaultdict


def read_punctuation(fb):
    punctuation = fb.tokenizer([".,-()[]{}_=+?!@#$%^&*\\/\"'`~;:|…）（•−"], add_special_tokens=False)['input_ids'][0]
    one_hot_punctuation = torch.ones(fb.bert.get_output_embeddings().out_features, dtype=torch.long)
    one_hot_punctuation[punctuation] = 0
    one_hot_punctuation[1232] = 0
    return one_hot_punctuation

def repair_masking(masked_doc):
    masked = ""
    for token in masked_doc:
        if token[:2] == "##":
            masked += token[2:]
        else:
            if masked:
                masked += " "
            masked += token
    return masked


def mask_vectors(self, sent, keep_original=False, add_special_tokens=False, padding=False):
    # tokens = self.tokenize(sent, add_special_tokens=add_special_tokens, padding=padding)
    # print(tokens)
    sent.squeeze_(0)
    # print(sent.shape)
    tlen = len(sent)
    offset = 1 if add_special_tokens else 0
    token_mat = [torch.clone(sent) for i in range(tlen - (2*offset))]
    for i in range(offset, tlen-offset):
        # print("ti B:", token_mat[i-offset][i])
        token_mat[i-offset][i] = self.mask_token_vector
        # print("ti A:", token_mat[i-offset][i])
    if keep_original:
        token_mat = [torch.clone(sent)] + token_mat
    return torch.stack(token_mat)


def score_vectors(self, methods, probs, return_all=True):
    # Enforce evaluation mode
    self.bert.eval()
    with torch.no_grad():
        use_pll = "pll" in [method.label for method in methods]

        assert not use_pll, "PLL cannot be used for raw vectors. Please choose a different metric."

        all_plls = {}
        scores = {}
        
        for method in methods:
            prob, alls = method(probs, return_all=True)
            if return_all:
                all_plls[method.label] = alls
            scores[method.label] = prob

        if self.device == "cuda":
            torch.cuda.empty_cache()
        # for method in scores:
        #     assert len(scores[method]) == len(sents_sorted)
        if return_all:
            return scores, all_plls
        return scores


def run_exp(fb:FitBert = None, resdir='res/', task_name='docred', dset='dev', doc=0, num_blanks=2, num_passes=1, nonlinearity=None, pooling=None, use_ent=True, top_k=0, skip=[], model='bert-large-cased', start_at=0, data_path='data'):
    with torch.no_grad():  # Super enforce no gradients whatsoever.
        torch.cuda.empty_cache()
        if fb is None:
            raise Exception('Need fb to not be None')
            # fb = extend_bert(FitBert(model_name=model), 50, num_blanks)
            # # nls = {None: lambda x:x, "softmax": FitBert.softmax, "relu":torch.relu}
            # global one_hot_punctuation
            # one_hot_punctuation = read_punctuation(fb)
        qx, qy = fb.tokenizer(["?x?y"], add_special_tokens=False)['input_ids'][0]
        # Take a document
        # Find all the entities
        processed_docs = []
        tot = 1000 if dset == 'dev' else 3053
        d: Document = ...  # Silly way to get some IDEs to give better completions.
        for d in Document.read(task_name=task_name, dset=dset, doc=doc, num_blanks=num_blanks, mlm=fb, path=data_path, use_ent=use_ent):
            # docfile = f"{resdir}/{task_name}_{model.model_name}_{dset}_{d.num}_{num_blanks}b_{num_passes}p.pickle"
            # docfile = f"{resdir}/{(task_name + '_') if task_name != 'docred' else ''}{(model + '_') if model != 'bert-large-cased' else ''}{dset}_{d.num}{'_' + str(num_blanks) + 'blanks' if num_blanks != 2 else ''}{'' if use_ent else '_MASK'}{'' if num_passes == 1 else '_'+str(num_passes)}.pickle"
            # print(docfile)
            # if os.path.exists(docfile) or
            if d.num in skip or d.num < start_at:
                # print(f"Pass {p_doc.num}")
                print(f"Document {d.num} skipped.", flush=True)
                continue
            print(f"Document {d.num} started.", flush=True)
            md = d.masked_doc
            if len(md['tokens']) <= 510:
                mask = [False] + md['ment_mask'] + [False]
                # print("mt", len(md['tokens']))
                # ents = md['ents']
                _tokens = fb.tokenizer.convert_tokens_to_ids(md['tokens'])
                _tokens = [fb.tokenizer.cls_token_id] + _tokens + [fb.tokenizer.sep_token_id]
                _tokens = torch.LongTensor([_tokens])
                V = fb.get_vocab_output_dim()  # [tokens, vocab(29028)]

                d.ment_inds = _tokens.squeeze(0)[mask]
                # d.ment_inds_masked = d.ment_inds.clone()
                # d.ment_inds_masked[d.ment_inds >= min(fb.entity_tokens)] = -1
                out = dict()
                # Initial pass: Just one-hot vectors as inputs.
                # print("t", _tokens.shape)
                input_vecs = torch.nn.functional.one_hot(_tokens, V).float()
                out[0] = input_vecs.cpu()
                # if num_passes == 0:
                #     # Then take the input tokens and convert them to one-hot vectors.
                # else:
                #     input_vecs = fb.bert.get_input_embeddings()(_tokens.to(fb.device)).cpu()
                #     out[0] = fb.bert(inputs_embeds=input_vecs.to(fb.device)).logits.cpu()
                #     d.ment_inds = torch.LongTensor([-1]*sum(mask))
                # del _tokens
                np = num_passes
                cp = 1
                while np > 0:
                    # Second and further passes: Run through the MLM.
                    # Step 1: Take original input vecs and sub in the entities.
                    if cp > 1:
                        entities = out[cp - 1].squeeze(0)[mask]
                        if top_k > 0:
                            entities = fb.fuzzy_embed(fb.top_k(entities, k=top_k))
                        elif top_k == 0:
                            entities = fb.fuzzy_embed(fb.softmax(entities))
                        else:
                            entities = fb.fuzzy_embed(entities)
                        input_embeds.squeeze(0)[mask] = entities.squeeze(0)
                    else:
                        # First pass: Just use normal input embeddings.
                        input_embeds = fb.bert.get_input_embeddings()(_tokens.to(fb.device))

                    # else it's the same vectors already.
                    # Then we do a forward pass, gather the new output logits.
                    out[cp] = fb.bert(inputs_embeds=input_embeds).logits.cpu()
                    print(cp, out[cp].shape)
                    cp += 1
                    np -= 1
                del _tokens
                
                d.ment_vecs = dict()
                for p in out:
                    if fb.token_width > 0:
                        d.ment_vecs[p] = out[p].squeeze(0)[mask].view(-1, fb.token_width, V)
                    else:
                        d.ment_vecs[p] = out[p].squeeze(0)[mask].view(-1, V)
                    # print(d.ment_vecs[p].shape)
                del out
                torch.cuda.empty_cache()
                print(f"Document {d.num} preprocessed.", flush=True)
                yield d#, docfile
                # exit(0)
            else:
                print("skipped", d)


def replace_embeddings(x, y, prompt):
    ix = prompt['ix']
    iy = prompt['iy']
    embs = prompt['vecs'].clone()
    # tkns = prompt['input_ids']
    return torch.cat([embs[:,:ix], x, embs[:,ix+1:iy], y, embs[:,iy+1:]], dim=1)


def replace_ids(x, y, prompt):
    ix = prompt['ix']
    iy = prompt['iy']
    embs = prompt['input_ids']
    return torch.cat([embs[:ix], x, embs[ix+1:iy], y, embs[iy+1:]])


def output_to_fuzzy_embeddings(fb, v):
    return (v.to(fb.device)@fb.bert.get_input_embeddings().weight).cpu()


def meminfo():
    f, t = torch.cuda.mem_get_info()
    f = f / (1024 ** 3)
    t = t / (1024 ** 3)
    return f"{f:.2f}g/{t:.2f}g"


def run_many_experiments(task_name, dset, rel_info, nonlin, pooler, scorers, num_blanks, docnum=-1, num_passes=1, max_batch=2000, use_ent=True, skip=[], stopfile='', model=None, start_at=0, data_path='data'):
    # import time
    if num_blanks == 0:
        poolers = [None]
    with torch.no_grad():
        torch.cuda.empty_cache()
        fb = (model or FitBert()).extend_bert(50, num_blanks)
        model_name = model.model_name.split('/')[-1]
        # nls = {None: lambda x:x, "softmax": FitBert.softmax, "relu":torch.relu}
        # global one_hot_punctuation
        # one_hot_punctuation = read_punctuation(fb)
        qx, qy = fb.tokenizer(["?x?y"], add_special_tokens=False)['input_ids'][0]
        prompt_data = {}
        prompts = list(sorted(rel_info.keys()))
        for prompt in prompts:
            pi = dict()
            tkns = fb.tokenizer(rel_info[prompt]['prompt_xy'], return_tensors='pt')['input_ids']
            pi['input_ids'] = tkns[0].cpu()
            pi['ix'] = torch.where(tkns[0] == qx)[0].item()
            pi['iy'] = torch.where(tkns[0] == qy)[0].item()
            # print("Cosine score:", score_batched(fb, [scorer], [prompt])[0]['csd'][0])
            pi['vecs'] = fb.bert.get_input_embeddings()(tkns.to(fb.device)).cpu()
            prompt_data[prompt] = pi
        all_all_scores = dict()
        for p_doc in run_exp(fb, task_name=task_name, dset=dset, doc=docnum, num_blanks=num_blanks, num_passes=num_passes, use_ent=use_ent, skip=skip, model=model_name, start_at=start_at, top_k=0, data_path=data_path):
            all_scores = dict()
            for nps in range(0, num_passes):
                print(f"{nps=}")
                # if os.path.exists(docfile.replace(f'_{num_passes}p', f'_{nps}p')):
                #     print(f"Document {p_doc.num} at {nps} passes skipped.", flush=True)
                #     continue
                all_scores = dict()
                # for nonlin in nonlins:
                print(f"{nonlin=}")
                # print(f"NL: {nonlin}")
                # all_scores[nonlin] = {}
                nl = FitBert.nonlins[nonlin]
                # for pooler in poolers:
                print(f"{pooler=}")
                # print(f"PL: {pooler}")
                # all_scores[nonlin][pooler] = {}
                evs, ev_tkns = p_doc.entity_vecs(nonlinearity=nl, pooling=pooler, passes=nps)
                fuzzy_embeds = {e:[output_to_fuzzy_embeddings(fb, v1.unsqueeze(0)) for v1 in evs[e]] for e in evs}
                e_to_m_map = p_doc.masked_doc['ents']
                # print(f"A: {torch.cuda.mem_get_info()}")
                # times = []
                for prompt_id in prompts: # rel_info:
                    print(f"{prompt_id=}")
                    # if prompt_id.split('|')[0] not in p_doc.relations:
                    #     continue
                    ans = [(a[0], a[2]) for a in p_doc.answers(detailed=False) if a[1] == prompt_id.split('|')[0]]
                    # BioRED explicitly states that all relations are non-directional.
                    # This is honestly false, but we marked the ones that aren't clearly non-directional to avoid issues.
                    # For example, "Conversion" is a one-way process between chemicals.
                    # "Bind" is questionable in this regard. It feels one-directional in some circumstances, but I'm not an expert...
                    # The remaining relations are obviously symmetric:
                    # Association, Positive/Negative Correlation, Comparison, Co-Treament, and Drug Interaction.
                    # The only DocRED relation marked symmetric is "sister city".
                    # "spouse" and "sibling" should also be marked as such, though, so that might get updated.
                    # (Those relations aren't examined in these experiments)
                    if rel_info[prompt_id]["symmetric"] == "true":
                        ans.extend([(a[1], a[0]) for a in ans if (a[1], a[0]) not in ans])
                    print(f"{ans=}")
                    # No sense in setting up a bunch of examples if none are correct.
                    # Maybe a retrieval system (RAG?) can make this selection in the wild?
                    if len(ans) == 0:
                        print("noans")
                        continue
                    all_scores[prompt_id] = {}
                    for sc in scorers:
                        all_scores[prompt_id][sc.label] = []
                    # prompt = rel_info[prompt_id]['prompt_xy']
                    # AAAAAAA
                    # tkns = prompt_data[prompt]['input_ids']
                    # score_len = len(tkns[0]) -2 + (2*num_blanks) - 2 + 1
                    # print(len(tkns[0]), score_len)
                    torch.cuda.empty_cache()
                    # print(f"B: {torch.cuda.mem_get_info()}")
                    res = defaultdict(lambda:-float('inf'))
                    # if num_blanks > 0:
                    # else:
                    #     fuzzy_embeds = {e:[output_to_fuzzy_embeddings(fb, v1.view(1, 1, -1)) for v1 in evs[e]] for e in evs}
                    all_replaced_m = {}
                    all_labels_m = {}
                    for e1 in fuzzy_embeds:
                        for e2 in fuzzy_embeds:
                            if e1 != e2:
                                _seen = set()
                                for v1, m1 in zip(fuzzy_embeds[e1], e_to_m_map[e1]):
                                    for v2, m2 in zip(fuzzy_embeds[e2], e_to_m_map[e2]):
                                        if nps == 0:
                                            vals = tuple(ev_tkns[m1].tolist() + [None] + ev_tkns[m2].tolist())
                                            if vals in _seen:
                                                continue
                                            else:
                                                _seen.add(vals)

                                        # vx = output_to_fuzzy_embeddings(fb, v1.unsqueeze(0))
                                        # vy = output_to_fuzzy_embeddings(fb, v2.unsqueeze(0))
                                        rep_vecs = replace_embeddings(v1, v2, prompt_data[prompt_id])
                                        # assert v1.shape[1] == ev_tkns[m1].shape[0]
                                        # print(rep_vecs.shape)
                                        mv = mask_vectors(fb, rep_vecs, keep_original=True, add_special_tokens=True)
                                        size = mv.shape[0]
                                        if size not in all_replaced_m:
                                            all_replaced_m[size] = []
                                            all_labels_m[size] = []
                                        all_replaced_m[size].append(mv)
                                        all_labels_m[size].append((e1, e2, m1, m2))
                    for size in all_replaced_m:
                        print(f"{size=}")
                        _max_batch_resized = max_batch - (max_batch % size)
                        _sentences_per_batch = _max_batch_resized // size
                        all_labels = all_labels_m[size]
                        print(f"{len(all_labels)} candidate statements.", flush=True)
                        bert_forward = torch.cat(all_replaced_m[size], dim=0).cpu()
                        for v in all_replaced_m[size]:
                            del v
                        # all_replaced_m[size] = None
                        torch.cuda.empty_cache()
                        # print(bert_forward.shape)
                        # fwd_pieces = []
                        print(f"Document {p_doc.num} scoring bert forward ({size}, {len(bert_forward)}) for {nonlin} {pooler} {prompt_id} at {nps} passes", flush=True)
                        while len(bert_forward) > 0:
                            print(f"BF: {len(bert_forward)} ({min(_max_batch_resized, len(bert_forward))//size}/{len(all_labels)})", flush=True)
                            sm_bert = fb.softmax_(fb.bert(inputs_embeds=bert_forward[:_max_batch_resized].to(fb.device)).logits)[:, 1:, :]
                            bert_forward = bert_forward[_max_batch_resized:]
                            torch.cuda.empty_cache()
                            # sm_bert = fb.softmax_(fb.bert(inputs_embeds=bert_forward.to(fb.device)).logits)[:, 1:, :]
                            # print(f"Document {p_doc.num} Past.", flush=True)
                            # This will cause issues.
                            # sm_bert = torch.cat(fwd_pieces) if len(fwd_pieces) > 1 else fwd_pieces[0]
                            # print(len(sm_bert.view(-1, score_len, sm_bert.shape[1], sm_bert.shape[2])), len(all_labels))
                            # print(all_labels)
                            # print(f"SM: {sm_bert.shape}")
                            # print(len(sm_bert.view(-1, score_len, sm_bert.shape[1], sm_bert.shape[2])), len(all_labels))
                            for (e1, e2, m1, m2), s in zip(all_labels[:_sentences_per_batch], sm_bert.view(-1, size, sm_bert.shape[1], sm_bert.shape[2])):
                                # print("S:", s.shape)
                                for scorer in scorers:
                                    origids = None
                                    if scorer.label == "pll":
                                        # Then make the index from its parts, same as the other thing.
                                        origids = replace_ids(ev_tkns[m1], ev_tkns[m2], prompt_data[prompt_id])[1:-1].to(fb.device)
                                    all_scores[prompt_id][scorer.label].append((e1, e2, m1, m2, ev_tkns[m1], ev_tkns[m2], (e1, e2) in ans, origids.cpu(), *scorer(s, origids=origids, return_all=True)))
                            all_labels = all_labels[_sentences_per_batch:]
                            # print(f"Document {p_doc.num} Tick.", flush=True)
                            # for scorer in scorers:
                            #     print(scorer.label, len(all_scores[nonlin][pooler][prompt_id][scorer.label]))
                            del s
                            del sm_bert
                            torch.cuda.empty_cache()
                        del bert_forward
                    # return
                all_all_scores[nps] = all_scores
                # with open(docfile.replace(f'_{num_passes}p', f'_{nps}p'), 'wb') as resfile:
                    # pickle.dump(all_scores, resfile)
                # if os.path.getsize(stopfile) > 0:
                #     break
        return all_all_scores