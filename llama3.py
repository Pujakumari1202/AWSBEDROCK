import boto3
import json


prompt_data="""
Act as a shakespeare and write a poem on Machine Learning

"""

AWS_REGION = "us-east-1"

# this client we will use and the service name is bedrock-runtime
bedrock=boto3.client(service_name="bedrock-runtime",region_name=AWS_REGION)


# structure of the payload and this will be key value pair
payload={
    "prompt":"[INST]" + prompt_data + "[/INST]",
    "max_gen_len":512,
    "temperature":0.5,
    "top_p":0.9
}


## convert into json
body=json.dumps(payload)
INFERENCE_PROFILE_ARN = "arn:aws:bedrock:us-east-1:194722404467:inference-profile/us.meta.llama3-3-70b-instruct-v1:0"




model_id="meta.llama3-3-70b-instruct-v1:0"


response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
    inferenceProfileArn=INFERENCE_PROFILE_ARN
)

response_body=json.loads(response.get("body").read())
response_text=response_body['generation']
print(response_text)


