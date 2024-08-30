# Question-Answer Generation from Documents

## Setup
1. Install the required packages using the following command:
```
python3 -m venv doc-env
source doc-env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2. Create an `.env` file under the root directory and add the following environment variables:
```
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
```
You can get the Google API key from [here](https://aistudio.google.com/app/apikey).

3. Update the config.yaml file with the desired prompt.

## Example Usage
````
python qa_generation.py --input_file data/tes.pdf --output_file data/sample_output.json --chunk_size 4096 --chunk_overlap 200
````