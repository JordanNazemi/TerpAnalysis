import request
from bs4 import BeautifulSoup
import pandas as pd
import regex as re

url = "https://www.leafly.com/strains/"

# soup = BeautifulSoup(page.content, 'html.parser')
# strain_effects = soup.find_all(class_="font-bold font-headers text-sm")

lab_data = pd.read_csv("results.csv")
strain_name_set = set()
num_found = 0

for i, j in lab_data.iterrows():
    name = j["Sample Name"].lower()
    name = re.sub(r'\([^()]*\)', '', name)
    split_string = name.split("-", 1)
    name = split_string[0]
    name = name.replace(" ", "-")
    if(name[-1] == "-"):
        name = name[:-1]

    # if name not in strain_name_set:
    #     strain_name_set.add(name)
    #     page_url = url + "name"
    #     page = requests.get(url)
    #     if (str(page) == "<Response [200]>"):
    #         num_found += 1

print(num_found)

# for effect in strain_effects:
#     print(effect)