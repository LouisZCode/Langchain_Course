
import pandas as pd

APPLICATIONS = "applications.csv"

company_name  = input("name of the company:")

df = pd.read_csv(APPLICATIONS)
#align wthe value to the one the databas would have:
company_name = company_name.casefold()
print(company_name)
#We need to first filter the row, if it exist
if company_name in df["company"].values:
    filtered_row = df[df["company"] == company_name]
    df.loc[filtered_row, df["company"]] = "new_status"
    print(f"The application to the {company_name} has been updated to NEW STATUS")

else:
    print("This company does not exist in the database")
