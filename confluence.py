import requests
import json
import os
from requests.auth import HTTPBasicAuth
import openai
import time
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Set OPENAI_API_KEY as an environment variable
openai.api_key  = os.getenv('OPENAI_API_KEY')

# Function to retrieve Confluence page content
def get_confluence_page(page_id):
    url = "https://<confluence-cloud-base-url>/wiki/api/v2/pages/{}?body-format=storage".format(page_id)

    auth = HTTPBasicAuth("confluence_email", "confluence_token")

    headers = {
        "Accept": "application/json"
    }

    response = requests.request("GET", url, headers=headers, auth=auth)

    if response.status_code == 200:
        return response.json()
    else:
        return None


# Function to extract text from Confluence page JSON
def extract_text_from_page(page):
    if "body" in page and "storage" in page["body"]:
        return page["body"]["storage"]["value"]
    else:
        return ""


# Function to save Confluence page data to a JSON file
def save_confluence_page(page_id, data, output_directory):
    filename = os.path.join(output_directory, "confluence_page_{}.json".format(page_id))
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


# Main script
if __name__ == "__main__":
    # Set up variables
    page_ids = [65538, 98411, 164008]  # List of Confluence page IDs to retrieve
    output_directory = "confluence_data"  # Directory to save Confluence page data
    output_file = "confluence_data.jsonl"  # Output JSONL file path

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create an empty list to store the processed data
    output_data = []

    for page_id in page_ids:
        page_data = get_confluence_page(page_id)
        if page_data:
            title = page_data["title"]
            body = extract_text_from_page(page_data)

            # Create a dictionary entry with prompt and completion
            entry = {
                "prompt": title,
                "completion": body
            }

            # Append the entry to the output data list
            output_data.append(entry)

            # Save the page data to an individual JSON file
            save_confluence_page(page_id, page_data, output_directory)

            print("Confluence page {} saved.".format(page_id))
        else:
            print("Failed to retrieve Confluence page {}.".format(page_id))

    # Save the output data to a JSONL file
    with open(output_file, "w", encoding="utf-8") as file:
        for entry in output_data:
            file.write(json.dumps(entry) + "\n")

    # Delete the individual JSON files
    for page_id in page_ids:
        filename = os.path.join(output_directory, "confluence_page_{}.json".format(page_id))
        os.remove(filename)

    print("All data saved to {} and individual JSON files deleted.".format(output_file))


# Prepare the dataset
dataset_path = 'confluence_data.jsonl'  # Path to your JSONL dataset file

# Fine-tune the model
def fine_tune_model():
    # Define fine-tuning parameters
    model = 'text-davinci-003'  # Pre-trained model to fine-tune
    steps = 1000  # Number of fine-tuning steps
    batch_size = 4  # Batch size for fine-tuning

    # Create a fine-tuning session
    response = openai.ChatCompletion.create(
        model=model,
        training_configuration={
            "dataset": dataset_path,
            "max_new_tokens": 100,
            "num_steps": steps,
            "batch_size": batch_size,
        },
    )

    # Monitor the fine-tuning progress
    model_id = response['id']
    print(f"Fine-tuning started. Model ID: {model_id}")

    while True:
        model_info = openai.ChatCompletion.retrieve(model_id)
        if model_info['status'] == 'ready':
            print("Fine-tuning completed.")
            break
        print("Fine-tuning in progress...")
        time.sleep(30)  # Wait for 30 seconds before checking again

    # Step 4: Save the fine-tuned model
    # Retrieve the fine-tuned model
    fine_tuned_model = openai.ChatCompletion.retrieve(model_id)

    # Save the fine-tuned model for later use
    with open('fine_tuned_model.json', 'w') as file:
        json.dump(fine_tuned_model, file)

# Start the fine-tuning process
fine_tune_model()


