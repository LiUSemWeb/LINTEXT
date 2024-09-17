import json
from itertools import permutations
import torch
import copy


def prompt(rel, rel_info, xy=True, ensure_period=True):
    if xy:
        prompt = rel_info[rel]['prompt_xy']
    else:
        prompt = rel_info[rel]['prompt_yx']
    if ensure_period and prompt[-1] != '.':
        return prompt + "."
    else:
        return prompt


def candidates(prompt:str, choices, return_ments=False):
    for a, b in permutations(choices, 2):
        if return_ments:
            yield prompt.replace("?x", a, 1).replace("?y", b, 1), (a, b)
        else:
            yield prompt.replace("?x", a, 1).replace("?y", b, 1)


class Document:
    def __init__(self, doc, num, width=1, num_passes=1, mlm=None, use_blanks=True, use_ent=True):
        self.doc = doc
        self.num = num
        self.mlm = mlm
        self.overlaps = {}
        self.mentions, self.mention_types, self.m_to_e  = self.read_mentions()
        self.entities = self.read_entities(self.doc['vertexSet'])
        self.relations = set([a[1] for a in self.answers(detailed=False)])
        self.num_passes = num_passes
        self.use_ent = use_ent
        self.width = width
        self.use_blanks = use_blanks
        self._masked_doc = None

    @property
    def masked_doc(self):
        if not self._masked_doc and (self.mlm and self.use_blanks):
            self._masked_doc = self.apply_blank_width(self.mlm, self.width)
        return self._masked_doc
        
    def apply_blank_width(self, mlm, width):
        if mlm:
            self.mlm = mlm
            self.blank_width = width
            if self.blank_width > 0:
                return self.mask_entities()
            else:
                return self.tokenize_entities()

    def contextualize_doc(self):
        self.unmasked_doc = self.contextualize(mask=False)
        return self.unmasked_doc

    def __getitem__(self, item):
        return self.doc[item]
    
    def __contains__(self, item):
        return item in self.doc

    def sentences(self):
        for sent in self['sents']:
            yield " ".join(sent)

    def text(self):
        return " ".join(self.sentences())

    def read_mentions(self):

        # Accumulate all mentions with their position (s, b, e) = (w, t)
        # avoid duplicates
        # Sort by key, ascending
        # return ordered list of mentions (w), mapping from index to type (t).

        mentions = dict()
        m_to_e = dict()
        for i, v in enumerate(self['vertexSet']):
            for m in v:
                s = m['sent_id']
                b, e = m['pos']
                w = self['sents'][s][b:e]
                t = m['type']
                if (s, b, e) not in mentions:
                    mentions[(s, b, e)] = (w, t, i)

        ments = list()
        types = dict()
        for i, (_, v) in enumerate(sorted(mentions.items())):
            w, t, e = v
            ments.append(w)
            m_to_e[i] = e
            types[i] = t
        return ments, types, m_to_e

    @staticmethod
    def read_entities(vertSet):
        ents = {}
        for i, ent in enumerate(vertSet):
            ents[i] = list(set(e['name'] for e in ent))
        return ents

    def tokenize_entities(self):
        # Step 1: Copy
        sents = copy.deepcopy(self['sents'])
        e_beg = '[ENT_BEG]'
        e_end = '[ENT_END]'

        seen_positions = []
        positions = []
        for i, v in enumerate(self['vertexSet']):
            for m in v:
                s = m['sent_id']
                b, e = m['pos']
                if (s, b, e) not in seen_positions:
                    seen_positions.append((s, b, e))
                    positions.append((s, b, e, i))
                else:
                    print(f"Duplicate at {(s, b, e)}")
        positions = list(sorted(positions, reverse=True))

        for s, b, e, _ in positions:
            sents[s][b:e] = [e_beg] + sents[s][b:e] + [e_end]
        sents = sum(sents, [])
        # print(sents, flush=True)
        tkns = self.mlm.tokenizer.tokenize(sents, add_special_tokens=False, is_split_into_words=True)
        e = 0
        mentions = []  # Handled.
        mention_mask = []  # Handled.
        mapp = {}
        e = -1
        m = len(positions)
        m_types = [None]*m
        ment_lens = [0]*m
        # m -= 1
        for i, w in enumerate(reversed(tkns)):
            if w == e_end:
                en = i
                e = positions.pop(0)[-1]
                if e not in mapp:
                    mapp[e] = []
                m -= 1
                mapp[e].append(m)
            elif w == e_beg:
                en = 0
                e = -1
            else:
                if e > -1:
                    ment_lens[m] += 1
                mention_mask.append(e > -1)
                mentions.append(m if mention_mask[-1] else -1)
        mentions = list(reversed(mentions))
        mention_mask = list(reversed(mention_mask))

        e_count = len(self['vertexSet'])

        tokens = [t for t in tkns if t.upper() not in [e_beg, e_end] ]
        
        
        return {
            "length": len(tokens),
            "tokens": tokens,
            "ments": mentions,
            "ment_mask":mention_mask,
            "ment_types":m_types,
            "ment_lens":ment_lens,
            "ents": mapp,
            "m_count": len(m_types),
            "e_count": e_count,
        }


    # Next step: How do we get entity types read from here?
    def mask_entities(self):
        # Step 1: Copy
        sents = copy.deepcopy(self['sents'])
        
        # Step 2: Replace all mention tokens with a placeholder
        # Note that these tokens are tokenized differently from BERT's tokens.
        for i, v in enumerate(self['vertexSet']):
            ent = f'[ENT_{i}_x]'
            for m in v:
                sid = m['sent_id']

                # Check for overlap here.
                w = sents[sid][m['pos'][0]]
                if (w[0:5].upper() == '[ENT_'):
                    self.overlaps[int(w.split("_")[1])] = i
                for r in range(*m['pos']):
                    sents[sid][r] = ent
        e_count = i + 1
        
        # Step 3: Replace all placeholders with single tokens
        mn = 0
        entities = []
        mentions = []
        mention_mask = []
        mapp = {}
        
        tokens = []
        e = -1
        for w in self.mlm.tokenizer.tokenize(" ".join(" ".join(s) for s in sents)):
            if (w[0:5].upper() == '[ENT_'):
                _e = int(w.split("_")[1])
                if e != _e:
                    e = _e
                    for i in range(self.blank_width):
                        if self.use_ent:
                            tokens.append(w.replace('x', str(i)))
                        else:
                            tokens.append(self.mlm.tokenizer.mask_token)
                        mentions.append(mn)
                        mention_mask.append(True)
                    if e not in mapp:
                        mapp[e] = []
                    mapp[e].append(mn)
                    mn += 1
            else:
                tokens.append(w)
                mentions.append(-1)
                mention_mask.append(False)
                e = -1
        m_types = [None]*mn
        ment_lens = [self.blank_width]*mn
        
        for i, v in enumerate(self['vertexSet']):
            tl = [m['type'] for m in sorted(v, key=lambda x: (x['sent_id'], x['pos'][0]))]
            if i in mapp:
                for m, t in zip(mapp[i], tl):
                    m_types[m] = t
            else:
                # print(f"Missing entity {i} for doc {self.num}")
                # if i in self.overlaps:
                #     print(f"It overlaps with {self.overlaps[i]}")
                # else:
                #     print("It doesn't overlap with anything")
                
                # found = False
                # for ans in self.answers(detailed=False):
                #     # print(ans)
                #     if (ans[0] == i) or (ans[2] == 1):
                #         found=True
                # if found:
                #     print("It WAS an answer entity.")
                pass
        
        return {
            "length": len(tokens),
            "tokens": tokens,
            "ments": mentions,
            "ment_mask":mention_mask,
            "ment_types":m_types,
            "ment_lens":ment_lens,
            "ents": mapp,
            "m_count": mn,
            "e_count": e_count,
        }

    def answers(self, detailed=True):
        ans = []
        for an in self['labels']:
            
            if detailed:
                ents = self.entities
                hs = ents[an['h']]
                ts = ents[an['t']]
                r = an['r']
                trips = []
                for h in hs:
                    for t in ts:
                        trips.append((h, r, t))
                ans.append(trips)
            else:
                ans.append((an['h'], an['r'], an['t']))
        return ans

    def answer_prompts(self):
        ents = self.entities()
        if 'labels' in self:
            ans = []
            for an in self['labels']:
                _ans = []
                pmpt = prompt(an['r'])
                for h in ents[an['h']]:
                    for t in ents[an['t']]:
                        _ans.append(pmpt.replace("?x", h, 1).replace("?y", t, 1))
                ans.append(_ans)
            return ans

    def candidate_maps(self, rel_info, rel:str=None, filt=True):
        if rel:
            rels = [rel]
        else:
            rels = rel_info
        for rel in rels:
            pmpt = prompt(rel, rel_info)
            prompts = {}
            dom = rel_info[rel]['domain']
            ran = rel_info[rel]['range']
            for a, b in permutations(self.mentions, 2):
                ta, tb = self.mention_types[a], self.mention_types[b]
                if not filt or (ta in dom and tb in ran):
                    prompts[pmpt.replace("?x", a, 1).replace("?y", b, 1)] = ((a, ta), (b, tb))
        return prompts

    def entity_vecs(self, nonlinearity=lambda x:x, pooling=None, passes=0):
        ent_vecs = {}
        ment_inds = {}
        if self.blank_width > 0:
            for e, inds in self.masked_doc['ents'].items():
                ent_vecs[e] = self.mlm.augment(self.ment_vecs[passes][inds], nonlinearity, pooling)
            minus_one = torch.LongTensor([-1]*self.blank_width).cpu()
            for i, s in enumerate(self.masked_doc['ment_lens']):
                ment_inds[i] = minus_one.clone()
        else:
            # Each ent_vecs[e] needs to be the correct corresponding set of vectors.
            # Lengths are available:
            ment_vecs = {}
            _s = 0
            for i, s in enumerate(self.masked_doc['ment_lens']):
                ment_vecs[i] = self.mlm.augment(self.ment_vecs[passes][_s:_s + s], nonlinearity, None)  # We can't pool here.
                ment_inds[i] = self.ment_inds[_s:_s + s]
                if passes > 0:
                    # ment_inds[i] = [-1]*len(ment_inds[i])
                    ment_inds[i] = torch.ones_like(ment_inds[i]) * -1
                _s += s
            for e, inds in self.masked_doc['ents'].items():
                ent_vecs[e] = [ment_vecs[m] for m in inds]
        return ent_vecs, ment_inds
    
    @staticmethod
    def read(task_name: str = 'docred', dset: str = 'dev', *, num_blanks=1, num_passes=1, mlm = None, path: str = 'data', doc=-1, verbose=False, use_ent=True):
        if task_name == 'docred' and dset == 'train':
            dset = 'train_annotated'
        with open(f"{path}/{task_name}/{dset}.json") as datafile:
            jfile = json.load(datafile)
            if doc >= 0:
                yield Document(jfile[doc], doc, width=num_blanks, mlm=mlm, num_passes=num_passes, use_ent=use_ent)
            else:
                for i, doc in enumerate(jfile):
                    yield Document(doc, i, width=num_blanks, mlm=mlm, num_passes=num_passes, use_ent=use_ent)
