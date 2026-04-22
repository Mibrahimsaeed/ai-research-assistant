import google.generativeai as genai

genai.configure(api_key="AIzaSyAVmKQX9dYsA557_a1qQlr66YMlPjbZqP4")

for m in genai.list_models():
    print(m.name, m.supported_generation_methods)