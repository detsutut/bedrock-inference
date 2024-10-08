# bedrock-inference

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


[![Status][status-shield]][status-url] 


<sub> **Keywords**: *generative AI, Amazon AWS, Bedrock, API, wrapper, LLM, inference, NLP*. </sub>

------------------------------

Bedrock Inference is a simple python package to handle 2FA and on-demand calls to AWS Bedrock foundation models.

<!-- REQUIREMENTS -->
## Requirements

To invoke Amazon Bedrock models, you need to authenticate with your AWS IAM account and establish a connection through MFA.
This requires to retrieve some codes 

- An **`Access key ID`** (*es: AKIAIOSFODNN7EXAMPLE*) and its correspondent **`secret access key`** (*es: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY*) for API requests
    - These credentials can be found created at Home -> Security credentials -> Access Keys.
        - If you don't have an access key yet: Access Keys -> Create access key
          
    - If you have AWS CLI/SDK installed and set up on your machine, chances are your credentials are already stored at `~/.aws/credentials` and/or in the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables. In this case, the package will retrieve them automatically.
        - *Note: AWS CLI and AWS SDks are <ins>not required</ins>  to use bedrock-inference.*

- A **`Multi-factor authentication device ARN`** (*es: arn:aws:iam::012345678910:mfa/example_username*)
    - The ARN of your MFA device can be located at Home -> Security credentials -> Multi-factor authentication
      
- The 6 digits **`temp token`** generated by the MDA device (*es: 366 712*)

### Tip

Even if you don't have any AWS service set up on your machine, you can still create the `~/.aws/credential` file to store your access key credentials for later reuse.
The credential file should look like this:

```
[bedrock_profile]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

The `profile` name you choose can be used later in `aws_login_mfa` in place of `aws_access_key_id` and `aws_secret_access_key` arguments.

<!-- INSTALLATION -->
## Installation

1. Make sure you have the latest version of pip installed
   ```sh
   pip install --upgrade pip
    ```
2. Install araucanaxai through pip
    ```sh
    pip install bedrock_inference
    ```

<!-- USAGE EXAMPLES -->
## Usage

Here is a minimal example of how the AWS Bedrock models can be invoked with `bedrock-inference`.

```python
from bedrock_inference import bedrock

# Authenticates through MFA and establish a session
session = bedrock.aws_login_mfa(arn="MFA_device_arn", token="MFA_token", aws_access_key_id="your_ID", aws_secret_access_key="your_key")
#session =  bedrock.aws_login_mfa(arn="MFA_device_arn", token="MFA_token") #use this if you already have AWS credentials set up for API requests on your machine

# Instantiate youre Bedrock caller
caller = bedrock.Bedrock(session=session, region="your_aws_region")

#Check available models in your region
print(caller.list_models())

#Invoke the target AWS model and retrieve the generated answer
answer = caller.invoke_model(prompt="What's the conceptual opposite of 'Hello World'?", model_id="target_aws_model_id")
print(answer)
```

A more extensive example, including advanced usage, can be found in [this notebook](bedrock_inference_example.ipynb).

<!-- CONTACTS AND USEFUL LINKS -->
## Contacts and Useful Links

*   **Repository maintainer**: Tommaso M. Buonocore  [![Gmail][gmail-shield]][gmail-url] [![LinkedIn][linkedin-shield]][linkedin-url]  

*   **Project Link**: [https://github.com/detsutut/bedrock-inference](https://github.com/detsutut/bedrock-inference)

*   **Package Link**: [https://pypi.org/project/bedrock-inference/](https://pypi.org/project/bedrock-inference/)

<!-- LICENSE -->
## License

Distributed under MIT License. See `LICENSE` for more information.


<!-- MARKDOWN LINKS -->
[contributors-shield]: https://img.shields.io/github/contributors/detsutut/bedrock-inference.svg?style=for-the-badge
[contributors-url]: https://github.com/detsutut/bedrock-inference/graphs/contributors
[status-shield]: https://img.shields.io/badge/Status-pre--release-blue
[status-url]: https://github.com/detsutut/bedrock-inference/releases
[forks-shield]: https://img.shields.io/github/forks/detsutut/bedrock-inference.svg?style=for-the-badge
[forks-url]: https://github.com/detsutut/bedrock-inference/network/members
[stars-shield]: https://img.shields.io/github/stars/detsutut/bedrock-inference.svg?style=for-the-badge
[stars-url]: https://github.com/detsutut/bedrock-inference/stargazers
[issues-shield]: https://img.shields.io/github/issues/detsutut/bedrock-inference.svg?style=for-the-badge
[issues-url]: https://github.com/detsutut/bedrock-inference/issues
[license-shield]: https://img.shields.io/github/license/detsutut/bedrock-inference.svg?style=for-the-badge
[license-url]: https://github.com/detsutut/bedrock-inference/blob/master/araucanaxai/LICENSE
[linkedin-shield]: 	https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url]: https://linkedin.com/in/tbuonocore
[gmail-shield]: https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white
[gmail-url]: mailto:buonocore.tms@gmail.com
