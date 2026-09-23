import json
import os
import glob
import time
 
import anthropic

TASKS_FILE = "matching_tasks.json"
MODEL = "claude-sonnet-5"

client = anthropic.Anthropic()

def main():
    with open(TASKS_FILE) as f:
        data = json.load(f)

    tasks = data["code_generation"]


    with open("baseline_results.txt", "w") as output_file:

        for task in tasks:
            prompt = (
                "Solve the following coding task.\n\n"
                + task
                + "\n\nProvide only the code solution. There should be no other explanation."
            )

            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}]
                )

                for block in response.content:
                    if block.type == "thinking":
                        output_file.write(block.thinking)
                    elif block.type == "text":
                        output_file.write(block.text)

                output_file.write("\n\n")

            except Exception as e:
                print(f"Error on {task}: {e}")


if __name__ == "__main__":
    main()