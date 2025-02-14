import os
import requests
import shutil

# Replace with your Dolby.io API key
api_key = os.environ.get("DOLBYIO_API_KEY")

# File paths
input_file_path = "E:/Coding_stuff/jobb_code/transcription/preprocessing/dolby/input_converted.wav"  # Replace with your local audio file path
output_file_path = "E:/Coding_stuff/jobb_code/transcription/preprocessing/dolby/processed.wav"  # Replace with desired output path

# Step 1: Upload the audio file to Dolby.io temporary storage
def upload_to_dolby(file_path):
    upload_url = "https://api.dolby.com/media/input"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {"url": f"dlb://in/{os.path.basename(file_path)}"}
    
    # Request pre-signed URL
    response = requests.post(upload_url, json=body, headers=headers)
    response.raise_for_status()
    presigned_url = response.json()["url"]
    
    # Upload file using pre-signed URL
    with open(file_path, "rb") as file:
        upload_response = requests.put(presigned_url, data=file)
        upload_response.raise_for_status()
    
    return body["url"]

# Step 2: Enhance the audio file
def enhance_audio(input_url):
    enhance_url = "https://api.dolby.com/media/enhance"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "input": input_url,
        "output": f"dlb://out/{os.path.basename(input_url)}",
        "content": {"type": "podcast"}  # Adjust content type as needed
    }
    
    response = requests.post(enhance_url, json=body, headers=headers)
    response.raise_for_status()
    
    return response.json()["job_id"], body["output"]

# Step 3: Check enhancement job status
def check_job_status(job_id):
    status_url = f"https://api.dolby.com/media/enhance"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    while True:
        response = requests.get(status_url, params={"job_id": job_id}, headers=headers)
        response.raise_for_status()
        status_data = response.json()
        
        if status_data["status"] == "Success":
            return status_data["result"]["output"]
        elif status_data["status"] == "Failed":
            raise Exception("Enhancement job failed.")
        
# Step 4: Download enhanced audio file
def download_from_dolby(output_url, save_path):
    download_url = f"https://api.dolby.com/media/output"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    params = {"url": output_url}
    with requests.get(download_url, params=params, headers=headers, stream=True) as response:
        response.raise_for_status()
        with open(save_path, "wb") as output_file:
            shutil.copyfileobj(response.raw, output_file)

# Main execution flow
try:
    print("Uploading audio file...")
    input_dlb_url = upload_to_dolby(input_file_path)
    
    print("Starting enhancement process...")
    job_id, output_dlb_url = enhance_audio(input_dlb_url)
    
    print("Checking enhancement job status...")
    enhanced_output_url = check_job_status(job_id)
    
    print("Downloading enhanced audio file...")
    download_from_dolby(enhanced_output_url, output_file_path)
    
    print(f"Enhanced audio saved at: {output_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
