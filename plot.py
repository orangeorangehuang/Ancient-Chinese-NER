import matplotlib.pyplot as plt
import json

f = open('output/trainer_state.json')
data = json.load(f)

x = []
f1 = []
loss = []
for item in data["log_history"]:
  if 'eval_f1' in item:
    x.append(item['step'])
    f1.append(item['eval_f1'])
    loss.append(item['eval_loss'])


plt.plot(x, f1) 
plt.xlabel("steps") 
plt.ylabel("f1 Score") 
plt.savefig("eval_f1.png") 
plt.show() 

plt.plot(x, loss) 
plt.xlabel("steps") 
plt.ylabel("Loss") 
plt.savefig("eval_loss.png") 
plt.show() 

f.close()