import litellm
import os
import base64
import json

# response = litellm.completion(
#   model = "ollama/qwen2.5vl",
#   messages=[
#       {
#           "role": "user",
#           "content": [
#                           {
#                               "type": "text",
#                               "text": "Whats in this image?"
#                           },
#                           {
#                               "type": "image_url",
#                               "image_url": {
#                               "url": "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"
#                               }
#                           }
#                       ]
#       }
#   ],
# )
# print(response)

"""
{
  "id": "chatcmpl-6c0a11af-bef5-4cb2-8e4f-986d557b588c",
  "created": 1751365505,
  "model": "ollama/qwen2.5vl",
  "object": "chat.completion",
  "system_fingerprint": None,
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": (
          "The image shows a simple, cartoon-style drawing of a llama or alpaca. "
          "The character has a round head with a single eye and a small mouth, "
          "and it appears to be waving with one hand. The drawing is minimalistic "
          "and uses only black lines on a white background."
        ),
        "tool_calls": None,
        "function_call": None,
        "provider_specific_fields": None
      }
    }
  ],
  "usage": {
    "completion_tokens": 58,
    "prompt_tokens": 53,
    "total_tokens": 111,
    "completion_tokens_details": None,
    "prompt_tokens_details": None
  }
}
"""


# response = litellm.completion(
#   model = "ollama/qwen2.5vl",
#   messages=[
#       {
#           "role": "user",
#           "content": [
#                           {
#                             "type": "text",
#                             "text": "Whats in this image?"
#                           },
#                           {
#                             "type": "image_url",
#                             "image_url": {
#                                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
#                             }
#                           }
#                       ]
#       }
#   ],
# )
# print(response)

"""
{
  "id": "chatcmpl-86c37e2d-12c5-4cd4-9d28-92acc4a81528",
  "created": 1751365349,
  "model": "ollama/qwen2.5vl",
  "object": "chat.completion",
  "system_fingerprint": None,
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": (
          "The image shows a serene beach scene during what appears to be either sunrise or sunset, "
          "as indicated by the warm, golden light. A woman is sitting on the sand, smiling, and "
          "interacting with a light-colored dog. The dog is wearing a colorful harness and is sitting "
          "on the sand, extending its paw towards the woman, who is also extending her hand to give "
          "the dog a high-five. The ocean is visible in the background, with gentle waves rolling onto "
          "the shore. The overall atmosphere of the image is peaceful and joyful, capturing a moment "
          "of connection between the woman and her dog."
        ),
        "tool_calls": None,
        "function_call": None,
        "provider_specific_fields": None
      }
    }
  ],
  "usage": {
    "completion_tokens": 119,
    "prompt_tokens": 1284,
    "total_tokens": 1403,
    "completion_tokens_details": None,
    "prompt_tokens_details": None
  }
}
"""

COUNT = 100
BASE_FOLDER="../.data/flaticon.com/target/train"

# Get all image files from subfolders
image_files = []
for root, dirs, files in os.walk(BASE_FOLDER):
    for file in files:
        #if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')):
        image_files.append(os.path.join(root, file))

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image 

def image_path_to_prompt(image_path):
    # Extract collection number and name from path
    # e.g., '../.data/flaticon.com/target/train/1171527-blockchain/png/001-blockchain.png'
    parts = image_path.split('/')
    collection_folder = parts[-3]  # '1171527-blockchain'
    filename = parts[-1]  # '001-blockchain.png'
    
    # Extract collection number and name
    collection_parts = collection_folder.split('-', 1)
    collection_num = collection_parts[0]
    collection_name = collection_parts[1] if len(collection_parts) > 1 else ""
    
    # Extract filename without extension
    file_base = filename.rsplit('.', 1)[0]
    file_ext = filename.rsplit('.', 1)[1]
    
    return f"""
This is from Flaticon.com Collection #{collection_num} \"{collection_name}\", {file_base} {file_ext}.
Describe what's in this image in a very very concise way, pay attention to the deisigner's original intention for a {file_base} logo,
under the colleciton of \"{collection_name}\"), only mention if details like color scheme, lines, layout, style, etc warrant mentioning. 
Don't use full sentence, more like a descriptoin from an art gallery description for a painting.
"""


for i in range(min(COUNT, len(image_files))):
    image_path = image_files[i]
    base64_image = image_to_base64(image_path)
    prompt_text = image_path_to_prompt(image_path)

    response = litellm.completion(
    model = "ollama/qwen2.5vl",
    messages=[
        {
            "role": "user",
            "content": [
              {
                  "type": "text",
                  "text": prompt_text
              },
              {
                  "type": "image_url",
                  "image_url": {
                    "url": base64_image
                  }
              }
          ]
        }
    ],

    )

    content = response.choices[0].message.content # response.choiceresponse.json()["choices"][0]["message"]["content"]
    print(content)

"""
{
  "id": "chatcmpl-97d5eb33-05fb-463e-b74f-24b9786fcf5c",
  "created": 1751367351,
  "model": "ollama/qwen2.5vl",
  "object": "chat.completion",
  "system_fingerprint": null,
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The image depicts a network or system architecture with four interconnected nodes, each represented by a cube. The cubes are connected by lines, suggesting a network or system where data or information can flow between the nodes. The color scheme is primarily shades of blue, with the cubes having a gradient effect. The lines connecting the cubes are also blue, with a lighter blue section in the middle, possibly indicating a different type of connection or a different level of data flow. The overall design suggests a modern, digital, or technological theme.",
        "tool_calls": null,
        "function_call": null,
        "provider_specific_fields": null
      }
    }
  ],
  "usage": {
    "completion_tokens": 105,
    "prompt_tokens": 361,
    "total_tokens": 466,
    "completion_tokens_details": null,
    "prompt_tokens_details": null
  }
}
"""


"""Ollama OpenAI Compatible API Example

 curl http://localhost:11434/v1/chat/completions     -H "Content-Type: application/json"     -d '{
        "model": "qwen2.5vl",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello!"
            }
        ]
    }'
"""