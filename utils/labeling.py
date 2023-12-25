import re

def labeling_occurrence(class_start, class_span, start, end, token_labels, text):
  print(start, end, text[start: end])
  for i in range(start, end):
    if token_labels[i] != 'O':
      return token_labels
  token_labels[start] = class_start
  for i in range(start + 1, end):
    token_labels[i] = class_span
  return token_labels

def labeling(obj, text):
  entities = []
  for i, item in enumerate(obj['PERSON']):
    entities.append((item, 'PER'))
  for i, item in enumerate(obj['LOC']):
    entities.append((item, 'LOC'))
  for i, item in enumerate(obj['BOOK']):
    entities.append((item, 'BOO'))
  for i, item in enumerate(obj['OFFICIAL']):
    entities.append((item, 'OFF'))
  for i, item in enumerate(obj['DYNASTY']):
    entities.append((item, 'DYN'))
  
  entities = sorted(entities, key=lambda tup: tup[0], reverse=True)
  entities = sorted(entities, key=lambda tup: len(tup[0]), reverse=True)

  """Class of Tokens
  O
  B-PER
  I-PER
  B-LOC
  I-LOC
  B-BOO
  I-BOO
  B-OFF
  I-OFF
  B-DYN
  I-DYN
  """
  token_labels = ['O'] * len(text)
  for ent in entities:
    for match in re.finditer(ent[0], text):
      if ent[1] == 'PER':
        class_start = 'B-PER'
        class_span = 'I-PER'
      elif ent[1] == 'LOC':
        class_start = 'B-LOC'
        class_span = 'I-LOC'
      elif ent[1] == 'BOO':
        class_start = 'B-BOO'
        class_span = 'I-BOO'
      elif ent[1] == 'OFF':
        class_start = 'B-OFF'
        class_span = 'I-OFF'
      elif ent[1] == 'DYN':
        class_start = 'B-DYN'
        class_span = 'I-DYN'
      # More class...

      token_labels = labeling_occurrence(class_start, class_span, match.start(), match.end(), token_labels, text)

  return token_labels