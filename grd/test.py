from openai import OpenAI
import os

client = OpenAI(
            api_key=os.getenv("SILICON_API_KEY"),
            base_url="https://api.siliconflow.cn/v1/",
        )


text = """First paragraph about a specific topic.
Second paragraph continuing the same topic.
Third paragraph switching to a different topic.
Fourth paragraph expanding on the new topic."""

response = client.embeddings.create(
            model="BAAI/bge-m3",
            input=text,
        )

print(response)