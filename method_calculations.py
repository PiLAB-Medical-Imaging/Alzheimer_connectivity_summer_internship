import pandas as pd
from datetime import date
import dateutil 

data = pd.read_csv("/Users/sam/Desktop/TAU_Dg_neuro_complet_DATA.csv", sep=";", decimal=",", encoding="latin-1")
long_data = pd.read_csv("/Users/sam/Desktop/long_form_combined.csv", sep=";", decimal=",", encoding="latin-1")
birth_dates = pd.to_datetime(long_data["DOB"], format="mixed", errors="coerce")
consultation_dates =  pd.to_datetime(long_data['Date_Cognitive_Assessment'], format="mixed", errors="coerce")
age = consultation_dates - birth_dates
days = age.dt.days
years = days/365.25

data["age"] = years



print(years)

print(data.columns)

data_minus_duplicate_IDs = data.drop_duplicates(subset="ID")



summarised_data = data_minus_duplicate_IDs.describe()

print(summarised_data["ID"])

group_numbers = data_minus_duplicate_IDs["GROUP"].value_counts()
sex_numbers = data_minus_duplicate_IDs["Sex"].value_counts()
visit_numbers = data["Visit_number"].value_counts()
age_numbers = data["age"].describe()
print(group_numbers)
print(sex_numbers)
print(age_numbers)

print(f"Visits \n {visit_numbers}")

# Play with the each visit statistics
subset = data[["Visit_number", "age", "Demented"]]
print(subset)
group_by_visit = subset.groupby(["Visit_number"]).describe()
print(group_by_visit)

