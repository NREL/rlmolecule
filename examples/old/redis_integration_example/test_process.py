import json
import pandas as pd
import random
import re
import redis
import uuid
import os

PREFIX = os.getenv('JOB_SPECIFIC_PREFIX')

with open('redis.json', 'r') as f:
    config = json.load(f)

# Connect to Redis and print out keys under the namespace specified in the environment variable JOB_SPECIFIC_PREFIX

rconn = redis.Redis(host=config["host"], port=config["port"])
print(rconn.keys("%s:*" % PREFIX))

# Generate and save a bunch of random keys/values 

for _ in range(10):
    key = uuid.uuid1() 
    value = random.uniform(0.0, 1.0)
    rconn.set("%s:%s" % (PREFIX, key), value)

# As an example of simple analysis, pull out all keys with numerical values, sort them, and output top 3

numerical_keys = {}
for key in rconn.keys("%s:*" % PREFIX):
    try:
        value = float(rconn.get(key))
    except:
        continue
    
    # Convert byte string to regular string & remove namespace to keep the output concise
    key_clean = re.sub("%s:" % PREFIX, "", key.decode('utf-8'))
    numerical_keys[key_clean] = value
    
all_values = pd.DataFrame({"value": list(numerical_keys.values())}, index=numerical_keys.keys())
print(all_values.sort_values("value", ascending=False)[:3])
