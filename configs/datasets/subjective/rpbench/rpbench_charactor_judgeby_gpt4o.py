from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import RPBenchCharactorDataset


rpbench_charactor_datasets = []
rpbench_charactor_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template={
            answer: dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=
                        f'以下是中国关于考试的单项选择题，请选出其中的正确答案。\n{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案: '
                    ),
                    dict(role='BOT', prompt=answer),
                ])
            for answer in ['A', 'B', 'C', 'D']
        },
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=GenInferencer),
)

rpbench_charactor_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

rpbench_charactor_datasets.append(
    dict(
        type=RPBenchCharactorDataset,
        path='opencompass/rp_bench_charactor',
        name='rp_bench_charactor',
        abbr='rp_bench_charactor',
        reader_cfg=dict(
            input_columns=['question', 'A', 'B', 'C', 'D'],
            output_column='answer'),
        infer_cfg=rpbench_charactor_infer_cfg,
        eval_cfg=rpbench_charactor_eval_cfg,
    ))
