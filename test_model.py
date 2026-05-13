import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download
import json


class RolePlayTester:
    """角色扮演模型测试器"""
    
    def __init__(self, model_path, device="auto"):
        """
        初始化测试器
        
        Args:
            model_path: 微调后的模型路径
            device: 设备，'auto'自动选择
        """
        print("=" * 80)
        print("正在加载模型和tokenizer...")
        print("=" * 80)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("模型加载成功！")
        print("=" * 80)
    
    def generate_response(self, user_input, max_new_tokens=256, temperature=0.8, top_p=0.9):
        """
        生成回复
        
        Args:
            user_input: 用户输入
            max_new_tokens: 最大生成长度
            temperature: 温度参数
            top_p: top_p采样
            
        Returns:
            生成的回复文本
        """
        # 构造对话格式
        messages = [
            {"role": "system", "content": "你是一个角色扮演助手，请根据用户的输入，给出符合角色特点的回复。"},
            {"role": "user", "content": user_input},
        ]
        
        # 转换为文本格式
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取assistant的回复
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        
        return response
    
    def test_single_query(self, query):
        """测试单个查询"""
        print(f"\n{'=' * 80}")
        print(f"用户输入: {query}")
        print(f"{'=' * 80}")
        
        response = self.generate_response(query)
        
        print(f"\n模型回复:\n{response}")
        print(f"{'=' * 80}\n")
        
        return response
    
    def test_batch(self, test_cases):
        """批量测试"""
        print(f"\n{'=' * 80}")
        print(f"开始批量测试，共 {len(test_cases)} 个测试用例")
        print(f"{'=' * 80}\n")
        
        results = []
        for idx, test_case in enumerate(test_cases, 1):
            print(f"\n[测试 {idx}/{len(test_cases)}]")
            response = self.test_single_query(test_case)
            results.append({
                "query": test_case,
                "response": response
            })
        
        return results
    
    def interactive_mode(self):
        """交互式测试模式"""
        print("\n" + "=" * 80)
        print("进入交互式测试模式")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 80)
        
        while True:
            user_input = input("\n请输入: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("\n退出交互模式")
                break
            
            if not user_input:
                print("输入不能为空，请重新输入")
                continue
            
            self.generate_response(user_input)
    
    def save_results(self, results, output_file="test_results.json"):
        """保存测试结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n测试结果已保存到: {output_file}")


def main():
    # 配置
    config = {
        "model_path": "./outputs/qwen_roleplay_merged",  # 微调后的模型路径
        "test_cases": [
            # 孙悟空相关的测试
            "猴哥，师父让我来请你回去，他说知道错了，请你原谅他，咱们师兄弟继续西行取经吧！",
            "大师兄，这妖怪太厉害了，我们还是绕道走吧！俺老猪的钉耙都怕它！",
            "师父，这去西天一路上的妖怪常会变化，肉眼看不出来。刚才俺老孙见那霞光中隐隐透出一股妖气，所以知道凶险。师父，快赶路吧！",
            
            # 更多测试用例
            "大师兄，我肚子疼得厉害，实在走不动了，不如让我休息一天，明天再去巡山吧！",
            "猴哥，你这火眼金睛也太厉害了，连我躲在树后偷吃你都看得见。",
            "猴哥，这水流太急了，咱们还是绕道走吧，听说前面有座桥呢！",
            "去不去？",
        ],
        "output_file": "test_results.json"
    }
    
    print("=" * 80)
    print("角色扮演模型测试工具")
    print("=" * 80)
    print(f"模型路径: {config['model_path']}")
    print("=" * 80)
    
    # 初始化测试器
    tester = RolePlayTester(model_path=config["model_path"])
    
    # 测试选项
    print("\n请选择测试模式:")
    print("1. 批量测试 (使用预设测试用例)")
    print("2. 交互式测试 (手动输入)")
    print("3. 单次测试")
    print("4. 退出")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    if choice == "1":
        # 批量测试
        print("\n" + "=" * 80)
        print("开始批量测试...")
        print("=" * 80)
        results = tester.test_batch(config["test_cases"])
        tester.save_results(results, config["output_file"])
        
    elif choice == "2":
        # 交互式测试
        tester.interactive_mode()
        
    elif choice == "3":
        # 单次测试
        query = input("\n请输入测试问题: ").strip()
        if query:
            tester.test_single_query(query)
        else:
            print("输入不能为空")
            
    elif choice == "4":
        print("退出程序")
        
    else:
        print("无效选项，退出程序")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
