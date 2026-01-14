from importlib.metadata import version
import tiktoken


print("tiktoken version:", version("tiktoken"))
#实例化
tokenizer = tiktoken.get_encoding("gpt2")
#分词器
text = (
"Hello, do you like tea? <|endoftext|> In the sunlit terraces" 
"of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
#使用 decode 方法将词元 ID 转换回文本
strings = tokenizer.decode(integers)
print(strings)
#使用 BPE 分词器对短篇小说 The Verdict 的全文进行分词
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
#移除前 50 个词元
enc_sample = enc_text[50:]
print(len(enc_text))
print(len(enc_sample))
#上下文大小决定了输入中包含多少个词元
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
#下一单词预测
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
print(f"x: {x}")
print(f"y: {y}")