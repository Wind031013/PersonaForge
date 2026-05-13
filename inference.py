import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)


def generate_response(scene, input_text):
    prompt = f"场景：{scene}\n输入：{input_text}\n输出："

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        repetition_penalty=1.1,
        temperature=0.7,
        top_p=0.9,
    )

    return


if __name__ == "__main__":
    print("孙悟空角色扮演模型推理测试")
    print("=" * 50)

    test_cases = [
        {"scene": "取经路上遇到妖怪拦路", "input": "猴哥，前面有个妖怪挡路！"},
        {"scene": "有人怀疑孙悟空的能力", "input": "你这猴子有什么本事，也敢来送死？"},
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"场景：{case['scene']}")
        print(f"输入：{case['input']}")
        print("输出：")
        generate_response(case["scene"], case["input"])
        print("=" * 50)
