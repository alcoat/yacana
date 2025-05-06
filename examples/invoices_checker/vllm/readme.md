Not the best example but it works ^^  
You'll probably get some loops in GroupChat. It's normal. The prompt engineering is not great and will be reworked at some point.  
It will stop when reaching a max_iteration counter (around 5 iterations) so let the script go. It will end.  

You'll need VLLM and a LLM: "meta-llama/Llama-3.1-8B-Instruct". You can replace it with your own inside the script.  

To run vllm do:
```
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192 --guided-decoding-backend outlines --enable-auto-tool-choice --tool-call-parser llama3_json
```

Install the PDF parser dependency:
```
pip install pypdf
```
...And the last version of Yacana obviously!

Then run it like this:
```
python3 quick_demo.py
```

To get better examples, read the doc. There are many code snippets to look at and play with.  
HF  