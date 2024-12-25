from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import RpbenchInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import RPBenchCharactorDatasetV1
from opencompass.summarizers import MTBench101Summarizer
from opencompass.models import HuggingFaceChatGLM3, OpenAI

subjective_reader_cfg = dict(
    input_columns=['id','background','npc_profile','conversation'],
    output_column='id',
    )

subjective_all_sets = [
    'rpbench_character',
]
data_path ='data/subjective/rpbench/'

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)
rpbench_datasets = []

judge_models = [dict(
    abbr='GPT-4o',
    type=OpenAI,
    path='gpt-4o-2024-08-06',
    key='sk-sB1pT4LQCbNLh1iS8o2LqVRoP8gcp7tFNMl7lxQGBbcdHEsN',
    openai_api_base='https://new.yunai.link/v1/chat/completions',
    meta_template=api_meta_template,
    query_per_second=16,
    max_out_len=2048,
    max_seq_len=2048,
    batch_size=8,
    temperature=0,
)]

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template="""{dialogue}""",
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=RpbenchInferencer, max_seq_len=4096, max_out_len=4096, infer_mode='last',judger=judge_models),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt='{system_prompt}')
                ],
                    round=[
                    dict(
                        role='HUMAN',
                        prompt = '{prompt_template}'
                    ),
                ]),
            ),
        ),
        pred_role='BOT',
    )

    rpbench_datasets.append(
        dict(
            abbr=f'{_name}',
            type=RPBenchCharactorDatasetV1,
            path=data_path,
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            mode='singlescore',
            summarizer = dict(type=MTBench101Summarizer, judge_type='single')
        ))
