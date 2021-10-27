import requests
from bs4 import BeautifulSoup
import pandas as pd
import regex as re


def url_scraper():
    url = "https://www.allbud.com/marijuana-strains/search?results=6000"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    strain_list = soup.find_all(class_="object-title")

    lab_data = pd.read_csv("results.csv")
    lab_data["URL"] = ""
    del lab_data["Provider"], lab_data["Receipt Time"], lab_data["Test Time"], lab_data["Moisture Content"]
    lab_data.URL = lab_data.URL.astype(str)
    lab_data['URL'].replace('', "NULL", inplace=True)

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)

    strain_name_set = set()
    num_found = 0
    counter = 0

    for entry in strain_list:
        counter += 1
        children = entry.findChildren("a")
        for x in children:

            strain_url = "https://www.allbud.com" + x["href"]
            split_string = x["href"].split("/")
            strain_name = split_string[3]
            print(f"Searching for {strain_name} ({counter}/6000)")

            for i, j in lab_data.iterrows():
                name = j["Sample Name"].lower()
                name = re.sub(r'\([^()]*\)', '', name)
                split_string = name.split("-", 1)
                name = split_string[0]
                name = name.replace(" ", "-")
                name = name.replace("#", "")
                name = name.replace("Trim", "")
                name = name.replace("Vape", "")
                name = name.replace("Shake", "")
                name = name.replace(r"\(.*\)", "")

                if (name[-1] == "-"):
                    name = name[:-1]

                if name not in strain_name_set:
                    if (strain_name == name):
                        strain_name_set.add(name)
                        num_found += 1
                        print(f"Found {strain_name} at {strain_url} - {len(strain_name_set)}")
                        lab_data.at[i, "URL"] = strain_url
                        break

    final_data = lab_data[~lab_data.URL.str.contains("NULL")]
    final_data.to_csv('results_url.csv', index=False)


def page_scraper():
    lab_data = pd.read_csv("results_url.csv")
    lab_data["Aromas"] = ""
    lab_data["Flavors"] = ""
    lab_data["Effects"] = ""

    for i, j in lab_data.iterrows():
        page_url = j["URL"]
        page = requests.get(page_url)
        soup = BeautifulSoup(page.content, 'html.parser')

        effect_result = soup.find_all(id="positive-effects")
        flavor_result = soup.find_all(id="flavors")
        aroma_result = soup.find_all(id="aromas")

        effect_array = []
        for y in effect_result:
            children = y.findChildren("a")
            for x in children:
                effect_array.append(x.text)

        aroma_array = []
        for y in aroma_result:
            children = y.findChildren("a")
            for x in children:
                aroma_array.append(x.text)

        flavor_array = []
        for y in flavor_result:
            children = y.findChildren("a")
            for x in children:
                flavor_array.append(x.text)

        print(j['Sample Name'])
        print(effect_array)
        print(aroma_array)
        print(flavor_array)

        lab_data.at[i, "Effects"] = effect_array
        lab_data.at[i, "Aromas"] = aroma_array
        lab_data.at[i, "Flavors"] = flavor_array

    lab_data.to_csv('full_results.csv', index=False)


url_scraper()
page_scraper()
