# TerpAnalysis

# Description
The purpose of this project is analyze the terpene makeup of weed strains and see
if they can be used with several analytics algorithms to predict aroma profiles.

We sourced our data from a now-deleted GitHub repository that skimmed reports from
SClabs, a lab that uses gas chromatography to identify the terpene contents of
cannabis strains with over 5,000 samples. The dataset contained the composition of
strains using 35 different terpenes (so values are between 0 and 1 representing percent
composition). We removed 6 terpene features containing sparse data, so we ultimately
had 29 features to our dataset. We then scraped AllBud, a website containing reviews
for different cannabis strains on subjective measures such as aroma, flavor, and effect,
for matching strains that were contained in the SClabs report. From this we were able to
find around 500 matching strains, which we used to append the subjective labels to the
terpene contents of each respective strain. We decided to focus on classifying strictly
using the aroma data, which we simplified from 47 different aromas down to 10: “diesel”,
“earthy”, “pine”, “citrus”, “fruity”, “skunky”, “nutty”, “sweet”, “spicy”, and “herbal”.

The AllBud website has changed format but the code originally used to skim the data can be found
under the "scraping" folder.

# Install
Install using ````"pip install -r requirements.txt"````

You can then run the individual models withing the /Models directory to run an analysis. The results of
which are summarized in the Terpene Report
