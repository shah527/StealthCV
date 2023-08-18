import openai

openai_api_key = '#'

openai.api_key = openai_api_key

response = openai.Completion.create(
  engine="text-davinci-003",
  prompt="Translate the following English text to French: 'Hello, World!'",
  max_tokens=60
)

print(response.choices[0].text)
