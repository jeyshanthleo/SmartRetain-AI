import boto3
import json

def generate_retention_strategy(customer_info, risk_level):

    prompt = f"""
You are a telecom customer retention expert.

Customer profile (JSON):
{json.dumps(customer_info, indent=2)}

Predicted churn risk level: {risk_level}

Return STRICT JSON only (no markdown, no extra text) in this exact schema:
{{
  "churn_reasons": ["...", "...", "..."],
  "recommended_strategy": ["...", "...", "..."],
  "personalized_offer": {{
    "title": "...",
    "details": "...",
    "estimated_cost": "...",
    "expected_impact": "..."
  }},
  "customer_message": {{
    "subject": "...",
    "body": "..."
  }}
}}
"""
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )

    body = json.dumps({
    "messages": [
        {
            "role": "user",
            "content": [
                {"text": prompt}
            ]
        }
    ],
    "inferenceConfig": {
        "maxTokens": 500,
        "temperature": 0.7
    }
})

    response = client.invoke_model(
        modelId="amazon.nova-pro-v1:0",
        body=body
    )

    result = json.loads(response["body"].read())

    return result["output"]["message"]["content"][0]["text"]
import os
from dotenv import load_dotenv

load_dotenv()
import boto3

client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION")
)
if __name__ == "__main__":

    customer_info = {
        "name": "John Doe",
        "tenure": 5,
        "monthly_charges": 90,
        "contract": "Month-to-month"
    }

    risk_level = "High"

    output = generate_retention_strategy(customer_info, risk_level)

    print("\nAI Retention Strategy:\n")
    print(output)