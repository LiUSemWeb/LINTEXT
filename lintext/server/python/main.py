# Scree! w/ Me: Switchboard
# Copyright (C) 2022 Reyncke Lab - All Rights Reserved
#
import asyncio
import os
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
    "http://localhost:8080",
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


@app.get("/", status_code=501, response_class=HTMLResponse)
def respond_get():
    return """
    <html>
        <head>
            <title>501 Not Implemented</title>
        </head>
        <body>
            <h1>501 Not Implemented</h1>
        </body>
    </html>
    """

async def tokenize(text:str):
    return getBERT().tokenizer.tokenize(text)

def getEntity(ment, masked_doc):
    if ment > -1:
        for e, ms in masked_doc['ents'].items():
            if ment in ms:
                return e
    return -1

def getDocument(dataset, subset, docnum, model='', **kwargs):
    # read_document(task_name=task_name, dset=dset, doc=doc, num_blanks=num_blanks, mlm=fb, path='data', use_ent=use_ent)
    mlm = getBERT(model) if model else getBERT()
    d = next(Document.read(task_name=dataset.lower(), dset=subset, doc=int(docnum), path=data_path, num_blanks=0, use_ent=True, mlm=mlm))

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

def analyze(text='', schema={}, dataset='', doc=-1, subset='', model=__def_model, **kwargs):
    if not schema:
        return error('Schema must include at least one relation.')
    if doc > -1:
        if not dataset:
            return error(f'Must specify dataset if specifying document number ({doc=})')
        elif not subset:
            error(f'Must specify subset of the dataset if specifying document number ({doc=})')
        else:
            fb = getBERT(model)
            if fb:
                res = run_many_experiments(task_name=dataset, dset=subset, rel_info=schema, nonlins=[None],
                                    poolers=[None], scorers=['pll'], num_blanks=0, num_passes=1, docnum=doc,
                                    max_batch=1000, model=fb)
                print(res)
            else:
                return error(f'Unknown MLM model name: f{model}')
            pass
    elif not text:
        return error('Must specify either dataset+document pair or raw text to be analyzed. Received neither.')
    else:
        return error('Text-only mode not yet implemented.')
        # pass
    
    { 'text': ...,  # raw text of the document, before tokenization
      'schema': ...,  # full schema to be applied. all relations will be analyzed.
      'dataset': ...,  # If applicable, name of the dataset (docred, biored, etc.)
      'doc': ...,  # If applicable, document number from known dataset
      'subset': ...,  # If applicable, which subset of the data (train, test, dev)
      'model': ...,  # The MLM model name to be used.
    }




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
    # Parse command-line arguments

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