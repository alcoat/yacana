Not the best example but it works ^^  
You'll probably get some loops in GroupChat. It's normal. The prompt engineering is not great and will be reworked at some point.  
It will stop when reaching a max_iteration counter (around 5 iterations) so let the script go. It will end.  

You'll need a ChatGPT account with a token and a LLM: gpt-4o-mini". You can replace it with another inside the script.  

You must set your OpenAi token as ENV variable:
```
export OPENAI_API_TOKEN=<your_token>
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