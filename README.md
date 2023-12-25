# Ancient Chinese NER with LLM-Generated Dataset
Final Project for ADL 2023 Fall<br/>
R11922101 Chia-Hung Huang

### Project Structure
- `data/`
  - `variant.csv`: Chinese variant character list.
  - `train_data.json`: Training data.
  - `val_data.json`: Validation data.
- `model/`: Saved checkpoint.
- `utils/`
  - `clean_variant.py`: Replace Chinese variant characters with standard characters.
  - `opanai_ner.py`: Call the OpenAI GPT-4-turbo API and generate the NER data in Python dictionary format.
  - `labeling.py`: Label all the tokens based on the dictionary data generated in the previous step.
- `generate_dataset.py`: Use functions in `utils/` to generate the training and validation dataset.
- `run_ner.py`: The training script.
- `run_ner_test.py`: The testing (predicting) script.
- `plot.py`: Plot the training curve (loss, f1 score) on the validation set.
- `app.py`: UI.

### How to Run the code
- Download the Model
  ```bash
  bash download.sh
  ```
- Train
  ```bash
  bash run.sh
  ```
- Run the App
  ```bash
  streamlit run app.py
  ```
