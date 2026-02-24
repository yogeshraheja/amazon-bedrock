import boto3
import json

def textToTextFnc(prompt_data):

    bedrock = boto3.client("bedrock-runtime")

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt_data
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 512,
            "temperature": 0.8,
            "topP": 0.8
        }
    }

    response = bedrock.invoke_model(
        modelId="<PUT_COMPLETE_ARN>:inference-profile/global.amazon.nova-2-lite-v1:0",
        body=json.dumps(payload)
    )

    result = json.loads(response["body"].read())

    # Extract generated text
    output_text = result["output"]["message"]["content"][0]["text"]

    return output_text
