"""
Login into your AWS and call Bedrock models in Python.

Classes:

    Bedrock

Functions:

    aws_login_mfa(string, string) -> boto3.Session

"""
import warnings
import os
from pandas import DataFrame
import boto3
from botocore.exceptions import ClientError, ProfileNotFound
from botocore.config import Config
from typing import Union


class Bedrock:
    """
    A class used to manage AWS Bedrock calls.
    This class is intentionally limited to a few harmful commands compared with correspondent API to avoid unintended disruptive actions.
    ...
    Attributes
    ----------
    _bedrock_client_ : object
        low-level service Bedrock client
    _runtime_client_ : object
        low-level service BedrockRuntime client

    Methods
    -------
    list_models(available,inference_type):
        Returns a list of all the (available) models.
    invoke_model(prompt, model_id, **kwargs):
        Call the aws model and retrieve the generated answer.
    """
    def __init__(self, session: boto3.Session, region="us-west-2"):
        """
        Log into AWS and start a Bedrock session in the target region.

        :param session: the established MFA session
        :param region: the AWS region where the target bedrock models are located (default: us-west-2)
        """
        self._bedrock_client_ = session.client(service_name = "bedrock",
                                               region_name=region,
                                               config=Config(read_timeout=320, retries={'max_attempts': 5}))
        self._runtime_client_ = session.client(service_name = "bedrock-runtime",
                                               region_name=region,
                                               config=Config(read_timeout=320, retries={'max_attempts': 5}))

    def list_models(self, available:bool = False, inference_type:str ='on_demand'):
        """
        List the available models.

        :param available: if True, only the models you actually gained access to are returned. WARNING: this will result in minimal yet nonzero invocation costs. Default: False.
        :param inference_type: either on_demand or provisioned. Default: on_demand.
        :return: DataFrame including all the models
        """
        df = DataFrame(self._bedrock_client_.list_foundation_models(byInferenceType=inference_type.upper())['modelSummaries'])
        if available:
            is_available = []
            warnings.warn("WARNING: Calling this method with available = True will result in minimal yet nonzero invocation costs.\nSince AWS does not provide official methods to filter out inaccessible models, this function loops over the model list trying to invoke each model with minimal input (\"\") and minimal output (maxTokens=1), filtering out all the models triggering AccessDeniedException.")
            for index,row in df.iterrows():
                try:
                    self._runtime_client_.converse(modelId=row['modelId'],
                                                   messages=[{"role": "user","content": [{"text": ""}]}],
                                                   inferenceConfig={"maxTokens":1})
                except ClientError:
                    #should be AccessDeniedException
                    is_available.append(False)
                else:
                    is_available.append(True)
            df = df[is_available]
        return df

    def invoke_model(self, prompt: str, model_id: str, **kwargs) -> Union[str,None]:
        """
        Invoke the AWS model and retrieve the generated answer.

        :param prompt: the text prompt
        :param model_id:
        :param kwargs: additional AWS configuration parameters to be passed when invoking the model (default: maxTokens=20, temperature=0.5)
        :return: answer: the generated text
        """


        inference_config = kwargs
        if kwargs.get('maxTokens') is None:
            inference_config['maxTokens'] = 20
        if kwargs.get('temperature') is None:
            inference_config['temperature'] = 0.5

        conversation = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]

        try:
            # Send the message to the model, using a basic inference configuration.
            response = self._runtime_client_.converse(
                modelId=model_id,
                messages=conversation,
                inferenceConfig=inference_config,
                additionalModelRequestFields={}
            )
            return response["output"]["message"]["content"][0]["text"]
        except (ClientError, Exception) as e:
            warnings.warn(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            return None



def aws_login_mfa(arn: str,
                  token: str,
                  profile: str = None,
                  aws_access_key_id: str = None,
                  aws_secret_access_key: str = None,
                  duration: int = 3600) -> boto3.Session:
    """
    Login to AWS Bedrock using MFA.
    First, the user authenticates to AWS using his long-term credentials. Credentials can be explicitly provided through an id-secret couple or by profile name (stored in ~/.aws/credentials). If no credentials are provided explicitly, the function tries to recover the "default" profile or by looking in the environment variables.
    If long-term connection is established, then the device ARN and the generated secret token are used to establish temporary MFA session to access all the AWS services requiring this security level.


    :param duration: the duration in seconds of the MFA session (default: 1 hour)
    :param aws_secret_access_key: the access key ID (es: AKIAIOSFODNN7EXAMPLE)
    :param aws_access_key_id: the secret access key (es: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY)
    :param profile: the name of profile storing the target credentials (es: "default")
    :param arn: the ARN of your MFA app/device (es: "arn:aws:iam::012345678910:mfa/example_username")
    :param token: the MFA token generated by the authenticator app/device (es: "123456")
    :return: session: the esablished MFA session
    """

    if profile is not None:
        session = boto3.Session(profile_name=profile)
        sts_client = session.client("sts")
    elif aws_access_key_id is not None and aws_secret_access_key is not None:
        # Initialize session using AWS Security Token Service(STS)
        sts_client = boto3.client('sts',
                            aws_access_key_id=aws_access_key_id,
                            aws_secret_access_key=aws_secret_access_key)
    else:
        # Attempt with 'default' profile
        try:
            session = boto3.Session(profile_name="default")
            sts_client = session.client("sts")
        except ProfileNotFound:
            # Attempt by looking for AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env variables
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            if aws_access_key_id is not None and aws_secret_access_key is not None:
                sts_client = boto3.client('sts', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
            else:
                raise ValueError("Provide long-term credentials either with a valid profile name or with id-secret pair.")

    # Request temporary credentials using MFA
    response = sts_client.get_session_token(DurationSeconds=duration, SerialNumber=arn, TokenCode=token)

    # Extract and use the temporary credentials to create a new session
    temp_credentials = response['Credentials']

    temp_session = boto3.Session(
        aws_access_key_id = temp_credentials['AccessKeyId'],
        aws_secret_access_key = temp_credentials['SecretAccessKey'],
        aws_session_token = temp_credentials['SessionToken']
    )

    return temp_session