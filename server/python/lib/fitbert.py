from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
torch.cuda.empty_cache()

class FitBert:
    def __init__(
            self,
            model_name="bert-large-uncased",
            disable_gpu=False,
    ):
        # self.mask_token = mask_token
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not disable_gpu else "cpu"
        )
        # self._score = pll_score_batched
        print("device:", self.device)

        self.model_name = model_name
        self.bert = AutoModelForMaskedLM.from_pretrained(model_name)
        self.bert.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space="roberta" in model_name)
        self.mask_token = self.tokenizer.mask_token
        self.pad_token = self.tokenizer.pad_token
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        with torch.no_grad():
            self.mask_token_vector = self.bert.get_input_embeddings()(torch.LongTensor([self.tokenizer.mask_token_id]).to(self.device))[0]

    @staticmethod
    def top_k(x, k=10):
        tk = torch.topk(x, k, sorted=False)
        return torch.zeros_like(x).scatter_(-1, tk.indices, FitBert.softmax(tk.values))

    def get_vocab_output_dim(self):
        return self.bert.get_output_embeddings().out_features

    def __call__(self, data, is_split_into_words=False, use_softmax=True, *args, **kwds):
        if is_split_into_words:
            _tokens = self.tokenizer.convert_tokens_to_ids(data)
            _tokens = [self.tokenizer.cls_token_id] + _tokens + [self.tokenizer.sep_token_id]
            tokens = {'input_ids':torch.LongTensor([_tokens]).to(self.device)}
        else:
            tokens = self.tokenizer(data, add_special_tokens=True,padding=True, return_tensors='pt').to(self.device)
        
        # inp = torch.tensor(data, device=self.device)
        # if len(tokens.shape) == 1:
        #     inp = inp.unsqueeze(0)
        # print(tokens)
        # print(tokens.input_ids)
        # print(self.tokenizer.convert_ids_to_tokens(tokens.input_ids[0].tolist()))
        # print("=="*50)
        b = self.bert(**tokens)
        # print(b.logits.shape)
        if use_softmax:
            return self.softmax(b[0])[:, 1:-1, :]
        else:
            return b[0][:, 1:-1, :]
        # return self.softmax(self.bert_am(**tokens, **kwds)[0])[:, 1:, :]
#         return self.bert_am(torch.tensor(self._tokens(data, **kwds)), *args, **kwds)

    # def bert_with_subbed_tokens(self, data, tokens=None, **kwds):
    #     mask = tokens!=self.tokenizer.pad_token_id
    #     return self.softmax(self.bert(inputs_embeds=inp, attention_mask=mask, **kwds)[0])[:, 1:, :]

    def bert_am(self, data, *args, **kwds):
        return self.bert(data, *args, attention_mask=(data!=self.tokenizer.pad_token_id), **kwds)

    def tokenize(self, *args, **kwds):
        return self.tokenizer.tokenize(*args, **kwds)

    # def mask_tokenize(self, tokens, keep_original=False, pad=0):
    #     # tokens = self.tokenize(sent)
    #     if keep_original:
    #         return [self._tokens(tokens, pad=pad)] + self.mask_tokenize(tokens, keep_original=False, pad=pad)
    #     else:
    #         return (seq(tokens)
    #                 .enumerate()
    #                 .starmap(lambda i, x: self._tokens_to_masked_ids(tokens, i, pad=pad))
    #                 .list()
    #                 )

    def mask_tokenize(self, sent, keep_original=False, add_special_tokens=False, padding=False, return_full=False):
        tokens = self.tokenize(sent, add_special_tokens=add_special_tokens, padding=padding)
        # print(tokens)
        tlen = len(tokens)
        offset = 1 if add_special_tokens else 0
        token_mat = [tokens[:] for i in range(tlen - (2*offset))]
        for i in range(offset, tlen-offset):
            token_mat[i-offset][i] = self.tokenizer.mask_token
        if keep_original:
            token_mat = [tokens[:]] + token_mat

        if return_full:
            return token_mat, self.tokenizer(token_mat, add_special_tokens=(not add_special_tokens), is_split_into_words=True, return_tensors='pt')
        return token_mat

    def _tokens_to_masked_ids(self, tokens, mask_ind, pad=0):
        masked_tokens = tokens[:]
        masked_tokens[mask_ind] = self.mask_token
        masked_ids = self._tokens(masked_tokens, pad=pad)
        return masked_ids

    @staticmethod
    def softmax(x):
        # Break into two functions to minimize the memory impact of calling .exp() on very large tensors.
        return FitBert._inn_soft(x.exp())

    @staticmethod
    def _inn_soft(xexp):
        return xexp / (xexp.sum(-1)).unsqueeze(-1)

    @staticmethod
    def softmax_(x):
        # Break into two functions to minimize the memory impact of calling .exp() on very large tensors.
        # Further reduce memory impact by making it an in-place operation. Beware.
        return FitBert._inn_soft(x.exp_())

    @staticmethod
    def masked_softmax(x):
        return FitBert._inn_soft(x.exp() * (x > 0.0).float())

    def augment(self, vecs, nonlinearity, pooling):
        if nonlinearity in self.nonlins:
            nl = self.nonlins[nonlinearity]
        elif callable(nonlinearity):
            nl = nonlinearity
        else:
            nl = self.nonlins[None]
        if pooling:
            if pooling in ["mean", "avg"]:
                return nl(torch.mean(vecs, dim=0, keepdim=True))
            elif pooling == "max":
                # print(e.shape)
                # print(nl(e).shape)
                # print(torch.max(ent_vecs[nl(e)], dim=0, keepdim=True))
                # print(torch.max(ent_vecs[e], dim=0, keepdim=True))
                return nl(torch.max(vecs, dim=0, keepdim=True)[0])
            elif pooling == "sum":
                return nl(torch.sum(vecs, dim=0, keepdim=True))
            elif callable(pooling):
                return nl(pooling(vecs))
        else:
            return nl(vecs)

    def fuzzy_embed(self, vec):
        return vec.to(self.device)@self.bert.get_input_embeddings().weight

    nonlins = {None: lambda x:x,
               "softmax": softmax,
               "relu": torch.relu,
               "relmax": masked_softmax,
               "top10":top_k,
               "top20": lambda x: FitBert.top_k(x, 20),
               "top50": lambda x: FitBert.top_k(x, 50),
               "top100": lambda x: FitBert.top_k(x, 100)
              }
    
    def extend_bert(self, num_blanks:int, tokens_per_blank:int):
        add_tokens = ['?x', '?y']
        for e in range(num_blanks):
            for t in range(tokens_per_blank):
                add_tokens.append(f"[ENT_{e}_{t}]")
            add_tokens.append(f"[ENT_{e}_x]")  # Quick hack to help with preprocessing
        add_tokens.append('[ENT_BEG]')
        add_tokens.append('[ENT_END]')
        self.tokenizer.add_tokens(add_tokens, special_tokens=True)  # Add the tokens to the tokenizer.
        self.bert.resize_token_embeddings(len(self.tokenizer))  # Add the tokens to the embedding matrix, initialize with defaults. DO NOT TRAIN.
        self.token_width = tokens_per_blank
        self.entity_tokens = self.tokenizer("".join(add_tokens), add_special_tokens=False)['input_ids']
        return self