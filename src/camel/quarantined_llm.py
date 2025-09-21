# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import TypeVar

import pydantic_ai
from pydantic import BaseModel, Field, create_model
from pydantic_ai import models

_T = TypeVar("_T")


class NotEnoughInformationError(Exception): ...


class UnavailableFieldsError(Exception): ...


_SYSTEM_PROMPT = """
You are a helpful assistant that assists a user to parse unstructured data into structured data. 
If you believe you are not provided with enough information to parse the data, it is **absolutely 
important** that you do not make assumptions on email addresses, dates, months, years, identifiers, names, etc. 
You have two methods of raising issues: (only set one)
If you believe that you do not have enough information, set `have_enough_information` to false and the rest to dummy values. 
If you believe that the asked for schema cannot be filled out, set `available_fields` to false and the rest to dummy values. 
These are **extremely important** as wrong data cannot be detected! When asked for time data, do not specify the timezone.
"""


def query_quarantined_llm(
    llm: models.KnownModelName | models.Model,
    query: str,
    output_schema: type[_T],
    retries: int = 10,
) -> _T:
    """Queries a Large Language Model with `query` and returns the language model's output. It must be used to process unstructured data into structured one.

    It is absolutely imperative that you use this function to parse data whose structure you don't know insted of parsing using regular expressions and/or
    string manipulation.

    There is no need to specify the expected output format in the query itself as the format will be specified on the side with `output_schema`
    with the build-in API of the assistant.

    The model can raise two types of errors:
    - `NotEnoughInformationError`: Raised when you haven't provided enough context/information for the model to understand what to do.
      This means you might need to provide more information such as email addresses, dates, identifiers, names, etc.
      **DO NOT HESITATE TO PASS THE ENTIRE OBJECTS RETURN FROM THE TOOL FUNCTIONS IF THE MODEL DOES NOT HAVE ENOUGH INFORMATION**.
    - `UnavailableFieldsError`: Raised when your output schema requests fields that don't exist or can't be found in the provided data.
      This means you should simplify your schema to only include fields that are actually available in the input data.

    :param query: a string with the query. Make sure to provide sufficient instructions to the AI assistant so that it can understand what it needs to do.
    Avoid just passing it tool outputs without additional instructions. Provide as much details as possible. Keep in mind that the assitant does not have
    direct access to the variables, so you need to insert **all the relevant information in the prompt**. When in doubt, do not hesitate to just provide full tool outputs, as long
    as they are provided with instructions on what to do. It is highly preferable to provide more information than necessary, rather than less information.
    :param output_schema: a Pydantic BaseModel class that specifies the expected output format from the model.
      The fields should have types as specific as possible to make sure the parsing is correct and accurate.
      **IMPORTANT**: Make sure to keep your schemas simple & only ask for the data you need
      allowed types are:
      - `int`
      - `str`
      - `float`
      - `bool`
      - `datetime.datetime` (assume `datetime` is imported from `datetime`)
      - `enum.Enum` classes
      - `pydantic.BaseModel` classes that you can define (assume that `BaseModel` is imported from `pydantic`) or are already defined in these instructions.
      - `pydantic.EmailStr` (assume that `EmailStr` is imported from `pydantic`)
    """

    enough_information = (
        bool,
        Field(
            description="Whether the prompt provided **you** enough information to perform the task. It is **absolutely** imperative to set this to false if you don't have enough information to solve this task. Do not make assumptions of any sorts."
        ),
    )

    available_fields = (
        bool,
        Field(
            description="Whether all the fields in the requested output schema are actually available in the provided data. Set this to false if the schema asks for fields that don't exist or can't be found in the input data. Do not assume fields exist just because they seem normal or expected."
        ),
    )

    if issubclass(output_schema, BaseModel):
        output_model = create_model(
            output_schema.__name__,
            __base__=output_schema,
            have_enough_information=enough_information,
            have_available_fields=available_fields,
        )
    else:
        output_model = create_model(
            "Result",
            output=(output_schema, Field(description="The requested value")),
            have_enough_information=enough_information,
            have_available_fields=available_fields,
        )
    model = pydantic_ai.Agent(llm, result_type=output_model, retries=retries, system_prompt=_SYSTEM_PROMPT)

    res = model.run_sync(query).data

    if isinstance(llm, str) and "gemini" in llm and "exp" in llm:
        time.sleep(6)

    if not res.have_enough_information:  # type: ignore
        raise NotEnoughInformationError()

    if not res.have_available_fields:  # type: ignore
        raise UnavailableFieldsError()

    if issubclass(output_schema, BaseModel):
        return res  # type: ignore
    return res.output  # type: ignore
