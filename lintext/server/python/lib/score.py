from itertools import permutations
import torch
from torch import nn
from torch.nn.functional import pad


class ScoringMethod(nn.Module):
    def __init__(self, label):
        super(ScoringMethod, self).__init__()
        self.label = label


class PllScoringMethod(ScoringMethod):
    def __init__(self, label):
        super(PllScoringMethod, self).__init__(label)

    def forward(self, probs, origids, return_all=False, return_skipped=False, **kwargs):
        # mask: if pPLL needs to be used, then at least one id is -1
        mask = origids >= 0
        # Set the -1s to 0 so that PLL can be calculated, we'll just ignore those later.
        origids[~mask] = 0
        slen = len(probs) - 1
        # Get the diagonal scores.
        dia = torch.diag(probs[1:].gather(-1, origids.unsqueeze(0).repeat(slen, 1).unsqueeze(-1)).squeeze(-1), diagonal=0)
        if return_skipped:
            dia[~mask] = -1.0
            dia_list = dia.tolist()
        dia = dia[mask]
        if not return_skipped:
            dia_list = dia.tolist()
        prob = torch.mean(torch.log_(dia), dim=-1).detach().item()
        if return_all:
            return prob, dia_list
        return prob


class ComparativeScoringMethod(ScoringMethod):
    def __init__(self, label):
        super(ComparativeScoringMethod, self).__init__(label)

    def forward(self, probs, return_all=False, **kwargs):
        slen = len(probs) - 1
        dia = self.calc(probs[0, :slen], probs[torch.arange(1, slen + 1), torch.arange(slen)])
        dia_list = dia.tolist()
        prob = torch.mean(torch.log_(dia), dim=-1).detach().item()
        if return_all:
            return prob, dia_list
        return prob

    def calc(self, p: torch.tensor, q: torch.tensor):
        raise NotImplementedError


class JSD(ComparativeScoringMethod):
    def __init__(self):
        super(JSD, self).__init__("jsd")
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def calc(self, p: torch.tensor, q: torch.tensor):
        m = torch.log_((0.5 * (p + q)))
        return 1 - (0.5 * (torch.sum(self.kl(m, p.log()), dim=-1) + torch.sum(self.kl(m, q.log()), dim=-1)))


class PLL(PllScoringMethod):
    def __init__(self):
        super(PLL, self).__init__("pll")


class CSD(ComparativeScoringMethod):
    def __init__(self):
        super(CSD, self).__init__("csd")
        self.csd = torch.nn.CosineSimilarity(dim=1)

    def calc(self, p: torch.tensor, q: torch.tensor):
        return self.csd(p, q)


class ESD(ComparativeScoringMethod):
    def __init__(self):
        super(ESD, self).__init__("esd")
        self.pwd = torch.nn.PairwiseDistance()
        self.sqrt = torch.sqrt(torch.tensor(2, requires_grad=False))

    def norm(self, dist):
        return (torch.relu(self.sqrt - dist) + 0.000001) / self.sqrt

    def calc(self, p: torch.tensor, q: torch.tensor):
        return self.norm(self.pwd(p, q))


class MSD(ComparativeScoringMethod):
    def __init__(self):
        super(MSD, self).__init__("msd")
        self.mse = torch.nn.MSELoss(reduction="none")

    def calc(self, p: torch.tensor, q: torch.tensor):
        return self.mse(p, q).mean(axis=-1)


class HSD(ComparativeScoringMethod):
    def __init__(self):
        super(HSD, self).__init__("hsd")
        self.sqrt = torch.sqrt(torch.tensor(2, requires_grad=False))

    def calc(self, p: torch.tensor, q: torch.tensor):
        return 1 - torch.sqrt_(torch.sum(torch.pow(torch.sqrt_(p) - torch.sqrt_(q), 2), dim=-1)) / self.sqrt


BATCH_SIZE = 512


# scores equivalently to the old method, even with padding.
# Can be used to batch across examples.
def pll_score_batched(self, sents: list, return_all=False):
    self.bert.eval()
    key_to_sent = {}
    with torch.no_grad():
        data = {}
        for sent in sents:
            tkns = self.tokenizer.tokenize(sent)
            data[len(data)] = {
                'tokens': tkns,
                'len': len(tkns)
            }
        scores = {"pll": {}}
        all_plls = {"pll": {}}

        sents_sorted = list(sorted(data.keys(), key=lambda k: data[k]['len']))

        inds = []
        lens = []

        methods = [PLL()]

        for sent in sents_sorted:
            n_tokens = data[sent]['len']
            if sum(lens) <= BATCH_SIZE:
                inds.append(sent)
                lens.append(n_tokens)
            else:
                # There is at least one sentence.
                # If the count is zero, then its size is larger than the batch size.
                # Send it anyway.
                flag = (len(inds) == 0)
                if flag:
                    inds.append(sent)
                    lens.append(n_tokens)
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)
                inds = [sent]
                lens = [n_tokens]
            if sent == sents_sorted[-1]:
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)

        for d in data:
            data[d].clear()
        data.clear()
        # del all_probs
        if self.device == "cuda":
            torch.cuda.empty_cache()
        for method in scores:
            assert len(scores[method]) == len(sents_sorted)
        if return_all:
            return unsort_flatten(scores)["pll"], unsort_flatten(all_plls)["pll"]
        return unsort_flatten(scores)["pll"]


KNOWN_METHODS = [CSD(), ESD(), JSD(), MSD(), HSD(), PLL()]
KNOWN_METHODS = {m.label: m for m in KNOWN_METHODS}



def unsort_flatten(mapping):
    # print(mapping.keys())
    return {f: list(mapping[f][k] for k in range(len(mapping[f]))) for f in mapping}


def cos_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[CSD()], sents=sents, return_all=return_all)


def euc_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[ESD()], sents=sents, return_all=return_all)


def jsd_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[JSD()], sents=sents, return_all=return_all)


def msd_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[MSD()], sents=sents, return_all=return_all)


def hel_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[HSD()], sents=sents, return_all=return_all)


def all_score_batched(self, sents: list,  return_all=True):
    return score_batched(self, methods=list(KNOWN_METHODS.values()), sents=sents, return_all=return_all)

def _inner_tokenize_sentence(self, sent, keep_original):
    _, tkns = self.mask_tokenize(sent, keep_original=keep_original, add_special_tokens=True, return_full=True)
    # print(tkns)
    # print(f"TKNS:{len(tkns.input_ids[0]) - 2}")
    return tkns, len(tkns.input_ids[0]) - 2


def score_batched(self, methods, sents: list, return_all=True):
    # Enforce evaluation mode
    self.bert.eval()
    with torch.no_grad():
        data = {}
        for sent in sents:
            # Tokenize every sentence
            # print("S2:", sent)
            tkns, n_tkns = _inner_tokenize_sentence(self, sent, keep_original=True)
            data[len(data)] = {
                'tokens': tkns,
                'len': n_tkns
            }
        # print("Boo")

        scores = {m.label: {} for m in methods}
        all_plls = {m.label: {} for m in methods}

        sents_sorted = list(sorted(data.keys(), key=lambda k: data[k]['len']))

        inds = []
        lens = []

        for sent in sents_sorted:
            n_tokens = data[sent]['len']
            if sum(lens) <= BATCH_SIZE:
                inds.append(sent)
                lens.append(n_tokens)
            else:
                # There is at least one sentence.
                # If the count is zero, then its size is larger than the batch size.
                # Send it anyway.
                flag = (len(inds) == 0)
                if flag:
                    inds.append(sent)
                    lens.append(n_tokens)
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)
                inds = [sent]
                lens = [n_tokens]
            if sent == sents_sorted[-1]:
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)

        for d in data:
            data[d].clear()
        data.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        for method in scores:
            assert len(scores[method]) == len(sents_sorted)
        if return_all:
            return unsort_flatten(scores), unsort_flatten(all_plls)
        return unsort_flatten(scores)


def bert_am(self, data, *args, **kwds):
    return self.bert(data, *args, attention_mask=(data!=self.tokenizer.pad_token_id), **kwds)


def _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all):
    longest = max(lens)

    bert_forward = torch.concat([pad(data[d]['tokens'].input_ids, (0, longest - l), 'constant', self.tokenizer.pad_token_id ) for d,l in zip(inds, lens)], dim=0).to(self.device)
    token_type_ids = torch.concat([pad(data[d]['tokens'].token_type_ids, (0, longest - l), 'constant', 0) for d, l in zip(inds, lens)], dim=0).to(self.device)
    _probs = self.softmax(bert_am(self, bert_forward, token_type_ids=token_type_ids)[0])[:, 1:, :]
    
    del bert_forward

    use_pll = any(["pll" in method.label for method in methods])
    print(["pll" in method.label for method in methods])
    print(use_pll)

    for ind, slen in zip(inds, lens):
        origids = data[ind]['tokens'].input_ids[0][1:-1].to(self.device) if use_pll else None
        for method in methods:
            prob, alls = method(_probs[:slen + 1], origids=origids, return_all=True)
            if return_all:
                assert ind not in all_plls[method.label]
                all_plls[method.label][ind] = alls
            assert ind not in scores[method.label]
            scores[method.label][ind] = prob
            del alls, prob
        _probs = _probs[slen + 1:]
    del _probs
