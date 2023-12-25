# Copyright 2020 The HuggingFace Team. All rights reserved.
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

python run_ner.py \
  --model_name_or_path output-final/checkpoint-160 \
  --train_file data/train_data.json \
  --validation_file data/val_data.json \
  \
  --text_column_name text \
  --label_column_name token_label \
  --output_dir output \
  --max_seq_length 512 \
  --logging_steps 10 \
  --num_train_epochs 20 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --evaluation_strategy steps \
  --eval_steps 10 \
  --save_steps 10 \
  --overwrite_output_dir \
  --do_train \
  --do_eval