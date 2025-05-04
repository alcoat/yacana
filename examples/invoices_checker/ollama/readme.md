Not the best example but it works ^^  
You'll probably get some loops in GroupChat. It's normal. The prompt engineering is not great and will be reworked at some point.  
It will stop when reaching a max_iteration counter (around 5 iterations) so let the script go. It will end.  

You'll need Ollama and a LLM: "llama3.1:8b". You can replace it with your own inside the script.  

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