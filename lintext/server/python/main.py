
import asyncio
import os
from pathlib import Path
import random

import uvicorn
import argparse
import json
import typing
# from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
from fastapi import Request
from lib.score import *
from lib.document import Document
from lib.fitbert import FitBert
from lib.util import run_many_experiments

from huggingface_hub import repo_exists

from typing import Dict, Any

from transformers import AutoTokenizer

__def_model = 'bert-large-uncased'
# __def_model = 'dmis-lab/biobert-large-cased-v1.1'
# __def_model = 'Charangan/MedBERT'


class CustomJSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(content, ensure_ascii=False).encode("utf-8")

OK = {'status':'OK'}
tokenizer:AutoTokenizer = None

fb = None

def getBERT(model_name:str=__def_model):
    global fb
    if fb is None or fb.model_name != model_name:
        # model_name = 'bert-large-cased'
        if repo_exists(repo_id=model_name, repo_type="model", token=False):
            fb = FitBert(model_name=model_name).extend_bert(0, 1)
        else:
            return None
    return fb


# Prepare HTTPS server
app = FastAPI()
# app.add_middleware(HTTPSRedirectMiddleware)

# app.add_middleware(CORSMiddleware)

origins = [
    "http://localhost",
    "http://localhost:13679",
    "http://magicant",
    "http://magicant:13679"
    "*",
]

verbose = False

# app.mount(
#     "/static",
#     StaticFiles(directory=Path(__file__).parent.absolute() / "assets/static"),
#     name="static",
# )

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/", response_class=CustomJSONResponse)
async def respond_post(data: Request) -> Dict[str, Any]:
    print("Received request!")
    out_json = {'tokens': []}
    try:
        in_json = await data.json()
        method = in_json['method']
        body = in_json['body']
        if method == 'fetch':
            out_json['tokens'].extend(getDocument(**body))
            print('Fetch schema?', in_json['schema'])
            if in_json['schema']:
                out_json['schema'] = getRelInfo(body['dataset'].lower())
        elif method == 'parse':
            tokenized = await asyncio.create_task(tokenize(body['text']))
            print(tokenized)
            for token in tokenized:
                i = random.randint(-3, 2)
                out_json['tokens'].append({'text': token, 'type': str(i) if i >= 0 else '', 'ent': i, 'ment': i})
        elif method == 'analyze':

            out_json['results'] = analyze(**body)
    except json.decoder.JSONDecodeError:
        print(f"[ERROR] Received an unexpected POST request:\n{data}")

    return out_json

# This MUST be after the above line enabling the POST API.
interface_path = 'lintext/build/web/'
app.mount("/", StaticFiles(directory=interface_path, html=True), name="/")

async def tokenize(text:str):
    return getBERT().tokenizer.tokenize(text)

def getEntity(ment, masked_doc):
    if ment > -1:
        for e, ms in masked_doc['ents'].items():
            if ment in ms:
                return e
    return -1

def getDocument(dataset, subset, docnum, model='', num_blanks=0, **kwargs):
    # read_document(task_name=task_name, dset=dset, doc=doc, num_blanks=num_blanks, mlm=fb, path='data', use_ent=use_ent)
    mlm = getBERT(model) if model else getBERT()
    mlm.extend_bert(50, 1)
    d = next(Document.read(task_name=dataset.lower(), dset=subset, doc=int(docnum), path=data_path, num_blanks=num_blanks, use_ent=False, mlm=mlm))

    for a, b in zip(d.masked_doc['tokens'], d.masked_doc['ments']):
        yield {'text': a, 'type':d.mention_types[b] if b >= 0 else '', 'ent': getEntity(b, d.masked_doc), 'ment': b}
    # print(d.masked_doc)

def getRelInfo(dataset):
    with open(f'{data_path}/{dataset}/rel_info_full.json', 'r') as rel_info_file:

        j = json.load(rel_info_file)
        out_list = []
        for a, b in j.items():
            b['rel_id'] = a
            out_list.append(b)
        return out_list
    
def error(text):
    return {'error': text}

def analyze(text='', schema={}, dataset='', doc=-1, subset='', model=__def_model, nonlin='top50', pooler=None, scorer='pll', num_passes=0, num_blanks=0, **kwargs):
    print(f"Shall we work?\n{dataset}_{subset} {doc} {model}\n{schema}")
    if not schema:
        return error('Schema must include at least one relation.')
    if doc > -1:
        if not dataset:
            return error(f'Must specify dataset if specifying document number ({doc=})')
        elif not subset:
            error(f'Must specify subset of the dataset if specifying document number ({doc=})')
        else:
            # return error('No more work.')
            # print('after')
            fb = getBERT(model)
            if fb:
                res = run_many_experiments(task_name=dataset.lower(), dset=subset.lower(), rel_info=schema, nonlin=nonlin,
                                    pooler=pooler, scorers=[KNOWN_METHODS[scorer]], num_blanks=num_blanks, num_passes=num_passes+1, docnum=doc,
                                    max_batch=1000, model=fb, data_path=data_path, use_ent=False)
                # print(res.keys())
                # print(num_passes, res[1])
                res = res[num_passes]
                print("RESULTS:")
                # lis_a = res[doc]
                lis_b = []
                seen = set()
                for rel in res:
                    # (3, 2, 0, 1, tensor([2002, 4502, 3406, 5666, 2618, 4517, 5387, 1006, 1044, 2078, 2546, 1007, 1011, 1020]),
                    # tensor([16736,  1011, 14447, 14671]), True, -12.33619499206543)
                    lis_b.extend(res[rel][scorer])
                lis_c = []
                for r in sorted(lis_b, key=lambda x: -x[-2]):
                    e1, e2, m1, m2, i1, i2, truth, tokens, score, allscores = r
                    if (e1, e2) not in seen:
                        # t1 = fb.tokenizer.convert_ids_to_tokens(i1)
                        if num_blanks == 1:
                            tokens_2 = tokens[:]
                            blanks = (tokens_2 == -1).nonzero(as_tuple=True)[0]
                            if len(blanks) > 0:
                                tokens_2[blanks[0]] = fb.tokenizer.convert_tokens_to_ids(f"[E{e1}]")
                                tokens_2[blanks[1]] = fb.tokenizer.convert_tokens_to_ids(f"[E{e2}]")
                        else:
                            tokens_2 = tokens
                        # t2 = fb.tokenizer.convert_ids_to_tokens(i2)
                        statement = " ".join(fb.tokenizer.convert_ids_to_tokens([1 if t == -1 else t for t in tokens_2]))
                        # statement = schema[rel]['prompt_xy'].replace('?x', fb.tokenizer.convert_tokens_to_string(t1)).replace('?y', fb.tokenizer.convert_tokens_to_string(t2))
                        # seen.add((e1, e2))
                        lis_c.append([e1, e2, m1, m2, statement, truth, allscores, score])
                return lis_c
            else:
                return error(f'Unknown MLM model name: f{model}')
            pass
    elif not text:
        return error('Must specify either dataset+document pair or raw text to be analyzed. Received neither.')
    else:
        return error('Text-only mode not yet implemented.')
        # pass
    return error("Well that didn't go well.")
    { 'text': ...,  # raw text of the document, before tokenization
      'schema': ...,  # full schema to be applied. all relations will be analyzed.
      'dataset': ...,  # If applicable, name of the dataset (docred, biored, etc.)
      'doc': ...,  # If applicable, document number from known dataset
      'subset': ...,  # If applicable, which subset of the data (train, test, dev)
      'model': ...,  # The MLM model name to be used.
    }


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Starts a server for parsing text and doing relation extraction.')
parser.add_argument("-p", "--port", type=int, default=13679)
parser.add_argument('-v', '--verbose', action='store_true', default=False)
parser.add_argument('-d', '--data', type=str, default='/data/git/understanding-pll/data')
args = parser.parse_args()
verbose = args.verbose
data_path = args.data
# print(f"Data path is {data_path}")

# print(__name__)

if __name__ == '__main__':

    print(f"Spinning up the server on port {args.port}!")
    # Spin up the HTTPS server
    uvicorn.run(
        "main:app",
        port=args.port,
        host='0.0.0.0',
        # ssl_keyfile=os.getenv('API_PRIVKEY_PATH'),
        # ssl_certfile=os.getenv('API_FULLCHAIN_PATH'),
        reload=True,
        workers=1,
        log_level='warning')
elif __name__ == 'main': # Uvicorn main
    getBERT()
#     analyze(text='', schema={
#   "Association": {
#     "name": "Association",
#     "desc": "",
#     "prompt_xy": "There is an association between ?x and ?y.",
#     "prompt_yx": "There is an association between ?y and ?x.",
#     "domain": [
#       "ChemicalEntity",
#       "GeneOrGeneProduct",
#       "DiseaseOrPhenotypicFeature",
#       "SequenceVariant"
#     ],
#     "range": [
#       "DiseaseOrPhenotypicFeature",
#       "GeneOrGeneProduct",
#       "ChemicalEntity",
#       "SequenceVariant"
#     ],
#     "reflexive": "false",
#     "irreflexive": "true",
#     "symmetric": "true",
#     "antisymmetric": "false",
#     "transitive": "false",
#     "implied_by": [],
#     "tokens": [],
#     "verb": 0
#   },}, dataset='biored', doc=0, subset='train', model=__def_model)