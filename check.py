from llm_factory import get_llm

providers = ["ollama", "groq", "mistral"]   # test whichever you want

for p in providers:
    print(f"\nTesting {p}...")

    try:
        llm = get_llm(p)

        response = llm.invoke("Explain what SEBI is in one sentence.")

        # handle response format
        if hasattr(response, "content"):
            print("Response:", response.content)
        else:
            print("Response:", response)

    except Exception as e:
        print(f"❌ Error in {p}:", e)