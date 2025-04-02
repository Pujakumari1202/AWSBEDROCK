import boto3
import json
import base64
import os


prompt_data="""
Provide me an 4k hd image of a beach ,also use a blue sky rainy season and cinematic display


"""

prompt_template=[{"text":prompt_data,"weight":1}]

AWS_REGION = "us-east-1"

bedrock=boto3.client(service_name="bedrock-runtime",region_name=AWS_REGION)


payload={
    "text_prompts":prompt_template,
    "cfg_scale":10,
    "seed":0,
    "steps":50,
    "width":512,
    "height":512
}


body=json.dumps(payload)
model_id="arn:aws:bedrock:us-east-1:194722404467:inference-profile/stability.stable-diffusion-x1-v0"
response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",

)

# we get artifact
response_body=json.loads(response.get("body").read())
print(response_body)
# take the first artifact
artifact=response_body.get("artifacts")[0]
# encoded into base64 
image_encoded=artifact.get("base64").encode("utf-8")
# converted into bytes
image_bytes=base64.b64decode(image_encoded)


# save image to a file in the output directory
output_dir="output"  # save image in this directory
os.makedirs(output_dir,exits_ok=True)
file_name=f"{output_dir}/generated-img.png"
with open(file_name,"wb") as f:
        f.write(image_bytes)



# all things are same just converting the bytes info into image and saving into a file

## not work because of the access denied