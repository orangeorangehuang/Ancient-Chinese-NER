from utils.clean_variant import clean
from utils.openai_ner import get_ner_dict
from utils.labeling import labeling
from google.oauth2.service_account import Credentials
import gspread
import pandas as pd
import time
import json

scope = ['https://www.googleapis.com/auth/spreadsheets'];
creds = Credentials.from_service_account_file("client_secret.json", scopes=scope);
gs = gspread.authorize(creds);
sheet = gs.open_by_url('https://docs.google.com/spreadsheets/d/19baEk071FIExai8AskmWXnbYV6yKjVwheA1-GVomYN8/edit?usp=sharing');
worksheet = sheet.get_worksheet(0)
df = pd.DataFrame(worksheet.get_all_records())

def generate_dict_with_openai():
  text = df['text']

  for i, row in enumerate(text):
    if i < 220:
      continue
    print(i)
    row = clean(row)
    worksheet.update_cell(i + 2, 3, row)
    res_dict = str(get_ner_dict(row))
    worksheet.update_cell(i + 2, 4, res_dict)
    time.sleep(5);

def generate_entity_list():
  text = df['text']
  entity_dict = df['dict']
  print(len(text))

  for i, row in enumerate(text):
    if (i < 357):
      continue
    obj_str = entity_dict[i]
    obj_str = obj_str.replace("'", '"')
    obj = json.loads(obj_str)
    token_labels = labeling(obj, row)
    worksheet.update_cell(i + 2, 5, str(token_labels))
    time.sleep(1);

def generate_training_sets():
  text = df['text']
  label = df['token_label']
  train_json = []
  val_json = []

  for i in range(400):
    lst = []
    for letter in text[i]:
      lst.append(letter)
    if (i % 10 == 1):
      val_json.append({"text": lst, "token_label": label[i].strip('][').split(', ')})
    else:
      train_json.append({"text": lst, "token_label": label[i].strip('][').split(', ')})

  with open('train_data.json', 'w', encoding='utf8') as json_file:
    json.dump(train_json, json_file, ensure_ascii=False)
  with open('val_data.json', 'w', encoding='utf8') as json_file:
    json.dump(val_json, json_file, ensure_ascii=False)

# generate_entity_list()
generate_training_sets()