from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_prompt():
  persona = "想像你是一個中國史專家，精通文言文的語法結構，非常了解中國歷史上的人名、地名、書名、官職。\n"

  entity_header = "Entity Definition:\n"
  entity_1 = "1. PERSON: 人名。\n"
  entity_2 = "2. LOC: 地名。地名(LOC)為地理位置的名稱，如城市、國家、區域、省份、州郡。\n"
  entity_3 = "3. BOOK: 書名。書名(BOOK)為中國歷史上出現過的書籍的名稱。\n"
  entity_4 = "4. OFFICIAL: 官職名(OFFICIAL)作為中國歷史上政府機構中的特定職位或稱號，用於指代具有特定職責和權限的人員。\n"
  entity_5 = "5. DYNASTY: 朝代名(DYNASTY)作為中國歷史上出現過的朝代的名字。\n"
  entity_definition = entity_header + entity_1 + entity_2 + entity_3 + entity_4 + entity_5

  output_format = "Output Format: {'PERSON': [list of entities present], 'LOC': [list of entities present], 'BOOK': [list of entities present], 'OFFICIAL': [list of entities present], 'DYNASTY': [list of entities present]}\n"

  example_header = "Examples:\n"
  example_1 = "1. Sentence: 自此之後，魯周霸、孔安國，雒陽賈嘉，頗能言尚書事。孔氏有古文尚書，而安國以今文讀之，因以起其家。逸書得十餘篇，蓋尚書滋多於是矣。\nOutput: {'PERSON': ['周霸',  '孔安國', '賈嘉', '孔氏', '安國'], 'LOC': ['魯', '雒陽'], 'BOOK': ['尚書', '古文尚書'], 'OFFICIAL': [], 'DYNASTY': []}\n"
  example_2 = "2. Sentence: 蘭陵王臧既受詩，以事孝景帝為太子少傅，免去。今上初即位，臧乃上書宿衛上，累遷，一歲中為郎中令。及代趙綰亦嘗受詩申公，綰為御史大夫。\nOutput: {'PERSON': ['蘭陵王臧',  '孝景帝',  '臧', '趙綰', '申公', '綰'], 'LOC': [], 'BOOK': [], 'OFFICIAL': ['太子少傅', '郎中令', '御史大夫'], 'DYNASTY': []}\n"
  example_3 = "3. Sentence: 文王在位五十年崩〈《呂氏春秋》曰四十一年。〉年九十七，葬于畢〈今長安縣西畢陌是也。〉。大姒十子：長伯邑考，次武王發，次管叔鮮，次周公旦，次蔡叔度，次曹叔振鐸，次郕叔武，次霍叔處，次康叔封；季曰聃季載〈皇甫謐曰：「文王生伯邑考，次武王，次管叔，次蔡叔，次郕叔，次霍叔，次周公，次曹叔，次康叔，次聃季。」\nOutput: {'PERSON': ['文王', '大姒', '伯邑考', '武王發', '管叔鮮', '周公旦', '蔡叔度', '曹叔振鐸', '郕叔武', '霍叔處', '康叔封', '聃季載', '皇甫謐', '武王', '管叔', '蔡叔', '郕叔', '霍叔', '周公', '曹叔', '康叔', '聃季'], 'LOC': ['畢', '長安縣', '畢陌'], 'BOOK': ['呂氏春秋'], 'OFFICIAL': [], 'DYNASTY': []}\n"
  example_4 = "4. Sentence: 其年七月，始皇帝至沙丘，病甚，令趙高為書賜公子扶蘇曰：「以兵屬蒙恬，與喪會咸陽而葬。」書已封，未授使者，始皇崩。書及璽皆在趙高所，獨子胡亥、丞相李斯、趙高及幸宦者五六人知始皇崩，餘群臣皆莫知也。\nOutput: {'PERSON': ['始皇帝', '趙高', '公子扶蘇', '蒙恬', '始皇', '胡亥', '李斯'], 'LOC': ['沙丘', '咸陽'], 'BOOK': [], 'OFFICIAL': ['丞相'], 'DYNASTY': []}\n"
  example_5 = "Sentence: 厥初，生民穴居野處，聖人教之結巢以避蟲豸之害而食草木之實，故號有巢氏，亦曰大巢氏，亦謂之始君，言君臣之道於是乎始也。臣每疑此在伏羲時為諸侯，如夏、商、周在唐、虞之世是也。\nOutput: {'PERSON': ['有巢氏',  '大巢氏', '始君', '伏羲', '唐', '虞'], 'LOC': [], 'BOOK': [], 'OFFICIAL': [], 'DYNASTY': ['夏', '商', '周']}\n"
  examples = example_header + example_1 + example_2 + example_3 + example_4 + example_5

  prompt = persona + "\n" + entity_definition + "\n" + output_format + "\n" + examples
  # print(prompt)
  return prompt


def get_ner_dict(sent):
  prompt_instruct = get_prompt()
  print('total tokens:', len(prompt_instruct) + len(sent))

  completion = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
      {"role": "system", "content": prompt_instruct},
      {"role": "user", "content": "Sentence: " + sent + "\nOutput: ?"}
    ],
    temperature=1,
    max_tokens=512,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  raw = completion.choices[0].message.content
  raw = raw.replace('\'', '"')
  raw = raw.replace('\n', '')
  raw = raw.replace('```json', '')
  raw = raw.replace('```', '')
  raw = raw.replace('Output:', '')
  raw = raw.replace(' ', '')
  print(raw)
  result = json.loads(raw)
  # print(result)
  return result